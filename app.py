import os
import sys
import logging
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func):
            return func

import gradio as gr

# Setup python path so imports work cleanly locally and on Hugging Face
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from backend.core.vector_store import VectorDBManager
from backend.core.llm_service import LLMService, build_search_query, decompose_query
from backend.core.reranker import ReRanker

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize core services.
# No HybridRetriever here: sparse (keyword) search now lives inside Qdrant as a
# native sparse vector, so there is no in-process BM25 index to build at startup.
# The old version pulled the entire corpus into RAM on every boot to do that.
logger.info("Initializing core RAG components...")
vector_db = VectorDBManager()
llm_service = LLMService()
reranker = ReRanker()

# Force the lazy embedding/reranker singletons to actually load now, not on
# whichever request happens to arrive first. Gradio Spaces have no separate
# build step (no Dockerfile) -- this module runs once at container start, so
# this is the only place to warm them ahead of real traffic.
logger.info("Warming models at startup...")
vector_db.embedding
reranker.encoder
vector_db.collection
logger.info("Models warmed -- ready to serve requests.")

def get_document_list():
    """
    Returns a predefined list of the most common Cross Validated tags
    to avoid scrolling a 218k-document collection over the network.
    """
    common_tags = [
        "machine-learning",
        "regression",
        "time-series",
        "probability",
        "hypothesis-testing",
        "bayesian",
        "distributions",
        "self-study",
        "neural-networks",
        "classification",
        "clustering",
        "anova"
    ]
    return ["🔍 All Topics"] + sorted(common_tags)



# NOTE: PDF upload has been removed.
# The corpus is now the Stack Exchange dump, loaded offline by
# `backend/scripts/parse_dump.py` -> `backend/scripts/seed_corpus.py`.
# Ingestion is deliberately NOT a runtime path: it is a one-off batch job run
# locally, so a Space restart never re-embeds the corpus

def get_message_role_and_content(msg):
    """Safely extracts role and content regardless of Gradio version (dict, ChatMessage, or list/tuple)."""
    if isinstance(msg, dict):
        return msg.get("role"), msg.get("content")
    if hasattr(msg, "role") and hasattr(msg, "content"):
        return msg.role, msg.content
    if isinstance(msg, (list, tuple)) and len(msg) >= 2:
        if msg[0] is not None:
            return "user", msg[0]
        return "assistant", msg[1]
    return None, None


def set_message_content(msg, content):
    """Safely sets the content of a message slot in-place."""
    if isinstance(msg, dict):
        msg["content"] = content
    elif hasattr(msg, "content"):
        msg.content = content
    elif isinstance(msg, list) and len(msg) >= 2:
        msg[1] = content
    return msg


def add_user_message(user_message, history):
    """Instantly appends the user's query to the chatbot UI and clears the input box."""
    if not user_message or not user_message.strip():
        return history, ""
    
    is_dict_format = hasattr(gr, "ChatMessage")
    if is_dict_format:
        if history and isinstance(history[0], dict):
            history = history + [{"role": "user", "content": user_message}]
        else:
            history = history + [gr.ChatMessage(role="user", content=user_message)]
    else:
        history = history + [[user_message, None]]
    return history, ""


def get_text_content(content) -> str:
    """Helper to extract string content from potentially list-based Gradio 6 multi-modal messages."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif hasattr(p, "text"):
                parts.append(p.text)
        return " ".join(parts)
    return str(content)


def needs_query_rewrite(query: str, chat_history: list) -> bool:
    """
    Determines if a query requires LLM-based rewriting and typo normalization.
    Returns True if:
    1. Chat history exists (to resolve conversational referents and pronouns).
    2. Query contains negation terms (to prevent distractor term contamination).
    3. Standalone queries to perform fast typo and grammar normalization.
    """
    return True


@spaces.GPU
def chat_stream(history, selected_doc):
    """Processes the last message in history through RAG and streams the answer."""
    import time
    start_time = time.time()
    
    if not history:
        yield history
        return

    # Get the last user message from history
    is_dict_format = hasattr(gr, "ChatMessage")
    role, user_message_raw = get_message_role_and_content(history[-1])
    user_message = get_text_content(user_message_raw)

    # Convert Gradio chat history format (excluding the last user message) to standard dictionary list
    chat_history_dicts = []
    for msg in history[:-1]:
        r, c = get_message_role_and_content(msg)
        if r and c:
            chat_history_dicts.append({"role": r, "content": get_text_content(c)})

    # Append the assistant message slot FIRST to give immediate UI feedback
    if is_dict_format:
        if history and isinstance(history[0], dict):
            history = history + [{"role": "assistant", "content": "*(Thinking & searching documents...)*"}]
        else:
            history = history + [gr.ChatMessage(role="assistant", content="*(Thinking & searching documents...)*")]
    else:
        if isinstance(history[-1], list):
            history[-1][1] = "*(Thinking & searching documents...)*"
        else:
            history = history + [[None, "*(Thinking & searching documents...)*"]]
            
    yield list(history)

    # Step 1: Rewrite Query ONLY if history exists and query contains referential pronouns or is ultra-short
    if needs_query_rewrite(user_message, chat_history_dicts):
        rewritten_query = llm_service.rewrite_query(user_message, chat_history_dicts)
    else:
        # Standalone questions skip external LLM rewriter calls to eliminate latency
        rewritten_query = user_message

    # Step 2: Build the SEARCH query. Falls back to concatenating the last two
    # conversation turns when the rewriter did nothing -- a follow-up like
    # "how do I prevent it?" is unsearchable on its own. See build_search_query.
    search_query = build_search_query(user_message, chat_history_dicts, rewritten_query)
    decomposed_queries = decompose_query(search_query, llm_service=llm_service)

    # Candidate pool tuned to 10 for low-latency interactive chat
    CANDIDATE_K = 10
    filter_source = None if selected_doc in ["🔍 All Topics", None] else selected_doc

    # Stage 1: Native Hybrid Search via Qdrant RRF Fusion (with multi-query decomposition)
    if len(decomposed_queries) > 1:
        fused_candidates = vector_db.search_multi_query(queries=decomposed_queries, n_results=CANDIDATE_K, source_file=filter_source)
    else:
        fused_candidates = vector_db.search_hybrid(query=search_query, n_results=CANDIDATE_K, source_file=filter_source)

    # Stage 2: Cross-Encoder Re-Ranking. Scored against what the user actually
    # typed, not the history-padded search string -- the padding is there to
    # find candidates, and it would otherwise skew relevance toward the old turn.
    reranked_results = reranker.rerank(query=user_message, chunks=fused_candidates, top_k=3)

    # Format citations. Stack Exchange answers have no page numbers, so a citation
    # is the question title plus the vote score and accepted flag.
    citations_list = [
        {
            'source_file': res['metadata'].get('question_title', 'Untitled question'),
            'score': res['metadata'].get('score', 0),
            'is_accepted': res['metadata'].get('is_accepted', False),
            'url': res['metadata'].get('url', ''),
            # display_text (plain chunk, no title/overlap prefix) if present;
            # falls back to the full embedded text for chunks indexed before
            # this field existed.
            'text_snippet': res['metadata'].get('display_text', res["text"])
        } for res in reranked_results
    ]

    full_text = ""

    try:
        # Stream answer text with ~50ms batch throttling to prevent Gradio UI queue backpressure
        last_yield_time = time.time()
        for token in llm_service.stream_answer_sync(query=user_message, citations=citations_list, chat_history=chat_history_dicts):
            full_text += token
            now = time.time()
            if now - last_yield_time > 0.05:
                history[-1] = set_message_content(history[-1], full_text)
                yield list(history)
                last_yield_time = now
            
        history[-1] = set_message_content(history[-1], full_text)
        yield list(history)
            
        # -------------------------------------------------------------------
        # Append Citations & Latency Footer (Done ONLY ONCE at the end)
        # -------------------------------------------------------------------
        display_text = full_text
        is_fallback = "out-of-boundary" in full_text.lower() or "do not have enough information" in full_text.lower()
        if citations_list and not is_fallback:
            display_text += "\n\n---\n### 📚 Sources\n"
            for i, cite in enumerate(citations_list, 1):
                badge = " ✅ accepted" if cite.get("is_accepted") else ""
                title = f"[{cite['source_file']}]({cite['url']})" if cite.get("url") else cite["source_file"]
                display_text += (
                    f"**{i}. {title}** — {cite.get('score', 0)} votes{badge}\n"
                    # Snippet capped: enough to verify the answer, not a reproduction
                    f"> {cite['text_snippet'][:300]}...\n\n"
                )

        # Append latency footer
        latency = time.time() - start_time
        display_text += f"\n*(⏱️ Generated in {latency:.1f}s)*"
        
        # Final UI update
        history[-1] = set_message_content(history[-1], display_text)
        yield list(history)
        
    except Exception as e:
        logger.error(f"Error during streaming answer: {e}")
        error_msg = full_text + f"\n\n[An error occurred: {str(e)}]"
        history[-1] = set_message_content(history[-1], error_msg)
        yield list(history)

# Custom CSS Theme & Glassmorphism Styling
custom_css = """
.container { max-width: 1200px; margin: auto; }
.header-box {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.4);
}
.header-box h1 { margin: 0 0 8px 0; font-size: 2.2rem; font-weight: 700; color: #ffffff; }
.header-box p { margin: 0; opacity: 0.9; font-size: 1.05rem; }
.badge {
    background: rgba(255, 255, 255, 0.2);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-left: 8px;
}
"""

with gr.Blocks(title="⚡ Portfolio RAG Assistant", css=custom_css, theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")) as demo:
    gr.HTML("""
        <div class="header-box">
            <h1>⚡ Qdrant Rerank RAG Assistant <span class="badge">Hybrid RAG + Cross-Encoder</span></h1>
            <p>Dual-Stage Vector & Sparse Search with Re-Ranking and Multi-Turn Conversational Memory</p>
        </div>
    """)

    with gr.Row():
        # Sidebar Controls
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Filter by topic")
            doc_dropdown = gr.Dropdown(
                label="Tag",
                choices=get_document_list(),
                value="🔍 All Topics",
                interactive=True
            )
            refresh_btn = gr.Button("🔄 Refresh tags", size="sm")

            gr.Markdown("---")
            gr.Markdown(
                "### ℹ️ About\n"
                "Answers are grounded in Cross Validated "
                "(stats.stackexchange.com) posts, licensed CC-BY-SA. "
                "Every answer links to its source thread."
            )

        # Main Chatbot Interface
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                type="messages",
                label="Conversational AI Assistant",
                height=520,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=PortfolioRAG"),
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                    {"left": "\\(", "right": "\\)", "display": False},
                    {"left": "\\[", "right": "\\]", "display": True}
                ]
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a statistics or machine-learning question...",
                    show_label=False,
                    scale=4,
                    container=False
                )
                submit_btn = gr.Button("Send 🚀", variant="primary", scale=1)
                clear_btn = gr.Button("Clear 🗑️", scale=1)

    # Event Handlers
    refresh_btn.click(
        fn=get_document_list,
        outputs=[doc_dropdown]
    )

    # Chain user message insertion -> bot response generation
    submit_btn.click(
        fn=add_user_message,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
        queue=False
    ).then(
        fn=chat_stream,
        inputs=[chatbot, doc_dropdown],
        outputs=[chatbot]
    )

    msg_input.submit(
        fn=add_user_message,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
        queue=False
    ).then(
        fn=chat_stream,
        inputs=[chatbot, doc_dropdown],
        outputs=[chatbot]
    )

    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg_input])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
