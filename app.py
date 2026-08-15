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
from backend.core.llm_service import LLMService
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
    Tag filter for the corpus. Reads distinct tags from the indexed answers.

    Note: this scrolls the whole collection, which is fine at a few thousand
    documents but should become a Qdrant facet query once the corpus is large.
    """
    try:
        chunks = vector_db.get_all_chunks()
        tags = set()
        for chunk in chunks:
            for tag in (chunk.get("metadata", {}).get("tags") or []):
                tags.add(tag)
        return ["🔍 All Topics"] + sorted(tags)
    except Exception as e:
        logger.error(f"Error fetching tag list: {e}")
        return ["🔍 All Topics"]


# NOTE: PDF upload has been removed.
# The corpus is now the Stack Exchange dump, loaded offline by
# `backend/scripts/parse_dump.py` -> `backend/scripts/seed_corpus.py`.
# Ingestion is deliberately NOT a runtime path: it is a one-off batch job run
# locally, so a Space restart never re-embeds the corpus.

@spaces.GPU
def chat_stream(user_message, history, selected_doc):
    """Processes user query through Query Rewriting, Dual Retrieval (Dense + Sparse), RRF Fusion, Cross-Encoder Re-Ranking, and LLM Streaming."""
    if not user_message or not user_message.strip():
        yield history, ""
        return

    # Convert Gradio chat history format to standard dictionary list
    chat_history_dicts = []
    for user_msg, bot_msg in history:
        if user_msg:
            chat_history_dicts.append({"role": "user", "content": user_msg})
        if bot_msg:
            chat_history_dicts.append({"role": "assistant", "content": bot_msg})

    # Step 1: Rewrite Query if chat history exists
    rewritten_query = llm_service.rewrite_query(user_message, chat_history_dicts)

    CANDIDATE_K = 15
    filter_source = None if selected_doc in ["🔍 All Topics", None] else selected_doc

    # Stage 1: Native Hybrid Search via Qdrant RRF Fusion
    fused_candidates = vector_db.search_hybrid(query=rewritten_query, n_results=CANDIDATE_K, source_file=filter_source)

    # Stage 2: Cross-Encoder Re-Ranking
    reranked_results = reranker.rerank(query=rewritten_query, chunks=fused_candidates, top_k=3)

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

    history = history + [[user_message, ""]]
    full_text = ""

    try:
        for token in llm_service.stream_answer_sync(query=user_message, citations=citations_list, chat_history=chat_history_dicts):
            full_text += token
            
            # Format text + citations if relevant
            not_found = "do not have enough information" in full_text.lower() or not citations_list
            display_text = full_text
            if citations_list and not not_found:
                display_text += "\n\n---\n### 📚 Sources\n"
                for i, cite in enumerate(citations_list, 1):
                    badge = " ✅ accepted" if cite.get("is_accepted") else ""
                    title = f"[{cite['source_file']}]({cite['url']})" if cite.get("url") else cite["source_file"]
                    display_text += (
                        f"**{i}. {title}** — {cite.get('score', 0)} votes{badge}\n"
                        # Snippet capped: enough to verify the answer, not a reproduction
                        f"> {cite['text_snippet'][:300]}...\n\n"
                    )

            history[-1][1] = display_text
            yield history, ""
    except Exception as e:
        logger.error(f"Error during streaming answer: {e}")
        history[-1][1] = full_text + f"\n\n[An error occurred: {str(e)}]"
        yield history, ""

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
                label="Conversational AI Assistant",
                height=520,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=PortfolioRAG")
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

    submit_btn.click(
        fn=chat_stream,
        inputs=[msg_input, chatbot, doc_dropdown],
        outputs=[chatbot, msg_input]
    )

    msg_input.submit(
        fn=chat_stream,
        inputs=[msg_input, chatbot, doc_dropdown],
        outputs=[chatbot, msg_input]
    )

    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg_input])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
