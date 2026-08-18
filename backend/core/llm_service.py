import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from langchain_groq import ChatGroq

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.core.config import SETTINGS
from backend.core import prompts

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(
        self,
        answer_variant: Optional[str] = None,
        rewrite_variant: Optional[str] = None
    ):
        """
        Args:
            answer_variant: Prompt variant for answer generation. Defaults to the
                'active' variant in system_prompts.yaml. The evaluation harness
                passes explicit names to sweep variants.
            rewrite_variant: Prompt variant for query rewriting. Same defaulting.
        """
        # Model settings live alongside their prompts in system_prompts.yaml,
        # so a prompt and the model it was tuned against stay in sync.
        self.answer_variant = answer_variant or prompts.active("answer")
        self.rewrite_variant = rewrite_variant or prompts.active("query_rewrite")

        answer_cfg = prompts.get("answer", self.answer_variant)
        rewrite_cfg = prompts.get("query_rewrite", self.rewrite_variant)

        logger.info(
            f"LLMService using prompts answer={self.answer_variant}"
            f"[{prompts.fingerprint('answer', self.answer_variant)}] "
            f"query_rewrite={self.rewrite_variant}"
            f"[{prompts.fingerprint('query_rewrite', self.rewrite_variant)}]"
        )

        groq_key = SETTINGS.GROQ_API_KEY
        if groq_key:
            # extra_body carries provider-specific params the LangChain wrapper
            # has no first-class argument for (e.g. reasoning_effort on gpt-oss).
            # Variants that don't set it are unaffected.
            client_kwargs = {}
            if answer_cfg.get("extra_body"):
                client_kwargs["model_kwargs"] = {"extra_body": answer_cfg["extra_body"]}

            self.client = ChatGroq(
                api_key=groq_key,
                model_name=answer_cfg["model"],
                temperature=answer_cfg["temperature"],
                max_tokens=answer_cfg["max_tokens"],
                streaming=True,
                **client_kwargs
            )
        else:
            self.client = None
            logger.warning("GROQ_API_KEY missing in SETTINGS. LLMService disabled.")

        openrouter_key = SETTINGS.OPENROUTER_API_KEY
        if openrouter_key and ChatOpenAI:
            self.openrouter_client = ChatOpenAI(
                openai_api_key=openrouter_key,
                openai_api_base="https://openrouter.ai/api/v1",
                model_name=rewrite_cfg["model"],
                temperature=rewrite_cfg["temperature"],
                default_headers={
                    "HTTP-Referer": "https://github.com/ksrinivasarao2012/qdrant-rerank-rag",
                    "X-Title": "RAG Portfolio Evaluation"
                }
            )
        else:
            self.openrouter_client = None

        gemini_key = SETTINGS.GEMINI_API_KEY
        if gemini_key and gemini_key.startswith("AIzaSy") and ChatGoogleGenerativeAI:
            self.gemini_client = ChatGoogleGenerativeAI(
                api_key=gemini_key,
                model="gemini-1.5-flash",
                temperature=rewrite_cfg["temperature"]
            )
        else:
            self.gemini_client = None

        github_token = SETTINGS.GITHUB_TOKEN
        if github_token and ChatOpenAI:
            # GitHub Models API uses standard OpenAI compatibility layer
            # Qwen-2.5-7B-Instruct is a great small model supported on GitHub Models
            model_to_use = "Qwen-2.5-7B-Instruct"
            self.github_client = ChatOpenAI(
                openai_api_key=github_token,
                openai_api_base="https://models.inference.ai.azure.com",
                model_name=model_to_use,
                temperature=rewrite_cfg["temperature"]
            )
        else:
            self.github_client = None

        hf_token = SETTINGS.HF_TOKEN
        if hf_token and ChatOpenAI:
            # Hugging Face Serverless Inference API (OpenAI compatible wrapper)
            self.hf_client = ChatOpenAI(
                openai_api_key=hf_token,
                openai_api_base="https://api-inference.huggingface.co/v1/",
                model_name="Qwen/Qwen2.5-7B-Instruct",
                temperature=rewrite_cfg["temperature"]
            )
        else:
            self.hf_client = None

        # Local rewriter model (only for dev/eval, avoids API limits completely)
        from pathlib import Path
        import os
        local_model_path = Path(__file__).resolve().parents[2] / "data" / "models" / "qwen2.5-7b-instruct-q4_k_m.gguf"
        if not local_model_path.exists():
            local_model_path = Path(__file__).resolve().parents[2] / "data" / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        self._local_rewriter = None
        if local_model_path.exists():
            # Limit thread count to max 8 to prevent high CPU utilization/OOM risk
            threads = min(8, os.cpu_count() or 4)
            try:
                from llama_cpp import Llama
                # 1. Try loading with GPU offloading enabled (all layers to GPU)
                logger.info(f"Initializing local query rewriter from {local_model_path.name} with GPU acceleration...")
                self._local_rewriter = Llama(
                    model_path=str(local_model_path),
                    n_ctx=2048,
                    n_threads=threads,
                    n_gpu_layers=-1,
                    chat_format="chatml",
                    verbose=False
                )
            except Exception as gpu_err:
                logger.warning(f"Failed to load local model with GPU: {gpu_err}. Falling back to CPU.")
                try:
                    from llama_cpp import Llama
                    # 2. Fall back to CPU only (n_gpu_layers=0)
                    self._local_rewriter = Llama(
                        model_path=str(local_model_path),
                        n_ctx=2048,
                        n_threads=threads,
                        n_gpu_layers=0,
                        chat_format="chatml",
                        verbose=False
                    )
                except Exception as cpu_err:
                    logger.warning(f"Failed to load local query rewriter on CPU: {cpu_err}. Falling back to API.")

        self.model_name = answer_cfg["model"]

    # ------------------------------------------------------------------
    # Shared prompt construction
    # ------------------------------------------------------------------
    # stream_answer, stream_answer_sync and generate_answer all send the exact
    # same prompt; only the transport differs (async stream / sync stream /
    # single call). These two helpers are the single place that prompt is built.

    @staticmethod
    def _format_history(chat_history: Optional[List[Any]]) -> List[Dict[str, str]]:
        """Normalizes dicts or Pydantic models into plain {role, content} pairs."""
        normalized = []
        for msg in chat_history or []:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            
            # Extract plain text from Gradio 6 list-based content structures
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict) and "text" in p:
                        parts.append(p["text"])
                    elif hasattr(p, "text"):
                        parts.append(p.text)
                content = " ".join(parts)
                
            if role and content:
                normalized.append({"role": role, "content": str(content)})
        return normalized

    def _format_context(self, citations: List[Dict[str, Any]]) -> str:
        """Renders retrieved chunks into the numbered citation blocks the prompt expects."""
        formatted_blocks = []
        for idx, cite in enumerate(citations, start=1):
            snippet_text = cite.get("text_snippet", "").strip()
            # Cap snippet to max 1500 characters (~350-400 words) for fast LLM TTFT
            if len(snippet_text) > 1500:
                snippet_text = snippet_text[:1500] + "..."
            formatted_blocks.append(
                prompts.render(
                    "answer", "citation_block", variant=self.answer_variant,
                    index=idx,
                    source=cite.get("source_file", "Unknown"),
                    page=cite.get("page_number", "N/A"),
                    snippet=snippet_text
                )
            )
        return "".join(formatted_blocks)

    def _build_messages(
        self,
        query: str,
        citations: List[Dict[str, Any]],
        chat_history: Optional[List[Any]] = None
    ) -> List[Any]:
        """
        Assembles the full message stack: grounding system prompt, conversational
        memory, then the final question with retrieved context attached.
        """
        messages = [SystemMessage(
            content=prompts.get("answer", self.answer_variant)["system"]
        )]

        for msg in self._format_history(chat_history):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=prompts.render(
            "answer", "user", variant=self.answer_variant,
            context=self._format_context(citations),
            query=query
        )))
        return messages

    def rewrite_query(self, query: str, chat_history: Optional[List[Any]] = None) -> str:
        """
        Uses an LLM (Groq first for speed, falling back to Gemini / HF / Local / GitHub / OpenRouter)
        to check the query for clarity, resolve history references, and generate a search-friendly query.
        """
        # Format history if present, otherwise pass an empty placeholder
        history_text = ""
        if chat_history:
            history_text = "".join(
                f"{msg['role'].upper()}: {msg['content']}\n"
                for msg in self._format_history(chat_history)
            )
        else:
            history_text = "(No previous conversation history)"

        prompt = prompts.render(
            "query_rewrite", "user", variant=self.rewrite_variant,
            history=history_text, query=query
        )

        # 1. Primary Rewriter: Groq (Ultra-fast, < 300ms, active API key)
        if self.client is not None:
            try:
                response = self.client.invoke([HumanMessage(content=prompt)])
                rewritten = response.content.strip()
                logger.info(f"Groq query analysis for '{query}' returned: '{rewritten}'")
                return rewritten
            except Exception as e:
                logger.warning(f"Failed to analyze query with Groq: {e}. Falling back to Gemini/HF.")

        # 2. Fall back to Gemini Flash first (with 3s timeout)
        if self.gemini_client is not None:
            try:
                response = self.gemini_client.invoke([HumanMessage(content=prompt)], config={"timeout": 3.0})
                rewritten = response.content.strip()
                logger.info(f"Gemini query analysis for '{query}' returned: '{rewritten}'")
                return rewritten
            except Exception as e:
                logger.warning(f"Failed to analyze query with Gemini: {e}. Falling back to HF/Local/GitHub.")

        # 3. Fall back to Hugging Face Serverless Inference API (with 3s timeout)
        if self.hf_client is not None:
            try:
                response = self.hf_client.invoke([HumanMessage(content=prompt)], config={"timeout": 3.0})
                rewritten = response.content.strip()
                logger.info(f"Hugging Face Serverless query analysis for '{query}' returned: '{rewritten}'")
                return rewritten
            except Exception as e:
                logger.warning(f"Failed to analyze query with Hugging Face Serverless: {e}. Falling back to Local/GitHub.")

        # 3. Prioritize Local GGUF Rewriter (Offline, avoids API rate limits completely)
        if self._local_rewriter is not None:
            try:
                system_prompt = prompts.get("query_rewrite", self.rewrite_variant).get("system", "")
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                res = self._local_rewriter.create_chat_completion(
                    messages=messages,
                    max_tokens=256,
                    temperature=0.0
                )
                rewritten = res["choices"][0]["message"]["content"].strip()
                logger.info(f"Local query analysis for '{query}' returned: '{rewritten}'")
                return rewritten
            except Exception as e:
                logger.warning(f"Local query rewriter failed: {e}. Falling back to GitHub/OpenRouter.")

        # 4. Fall back to GitHub Models API (High limit, 100% free)
        if self.github_client is not None:
            try:
                response = self.github_client.invoke([HumanMessage(content=prompt)])
                rewritten = response.content.strip()
                logger.info(f"GitHub Models query analysis for '{query}' returned: '{rewritten}'")
                return rewritten
            except Exception as e:
                if "429" in str(e):
                    logger.warning(f"GitHub Models rate limit hit. Sleeping 5s before retry...")
                    time.sleep(5)
                    try:
                        response = self.github_client.invoke([HumanMessage(content=prompt)])
                        rewritten = response.content.strip()
                        return rewritten
                    except Exception as retry_err:
                        logger.warning(f"GitHub Models retry failed: {retry_err}. Falling back to OpenRouter.")
                else:
                    logger.warning(f"Failed to analyze query with GitHub Models: {e}. Falling back to OpenRouter.")

        # 5. Fall back to OpenRouter
        if self.openrouter_client is not None:
            try:
                response = self.openrouter_client.invoke([HumanMessage(content=prompt)])
                rewritten = response.content.strip()
                logger.info(f"OpenRouter query analysis for '{query}' returned: '{rewritten}'")
                return rewritten
            except Exception as e:
                if "429" in str(e):
                    logger.warning(f"OpenRouter rate limit hit. Sleeping 5s before retry...")
                    time.sleep(5)
                    try:
                        response = self.openrouter_client.invoke([HumanMessage(content=prompt)])
                        rewritten = response.content.strip()
                        return rewritten
                    except Exception as retry_err:
                        logger.error(f"OpenRouter retry failed: {retry_err}")
                        return query
                else:
                    logger.error(f"Failed to analyze query with OpenRouter: {e}")
                    return query

        return query

        return query

    async def stream_answer(
        self,
        query: str,
        citations: List[Dict[str, Any]],
        chat_history: Optional[List[Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously streams the LLM answer token-by-token while factoring in
        conversational memory and retrieved document citations.

        Always uses the Groq client -- same as stream_answer_sync and
        generate_answer. This used to branch to a Hugging Face Inference API
        fallback whenever `self.model_name` contained a "/", but Groq's own
        model IDs (e.g. "openai/gpt-oss-20b", the currently active model)
        also contain "/", so that check misfired and silently misrouted every
        real request on the FastAPI backend (routes.py, the only caller of
        this method) to Hugging Face instead of Groq. Removed rather than
        fixed: there's no documented use case for the HF fallback today (see
        CLAUDE.md), and keeping three near-identical generation paths
        consistent is simpler than maintaining a second, harder-to-test one.
        """
        if not self.client:
            raise ValueError("LLM Client not initialized. Check Groq API Key.")

        messages = self._build_messages(query, citations, chat_history)

        try:
            async for chunk in self.client.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            yield "\n[An error occurred while generating the response.]"

    def stream_answer_sync(
        self, 
        query: str, 
        citations: List[Dict[str, Any]], 
        chat_history: Optional[List[Any]] = None
    ):
        """
        Synchronously streams the LLM answer token-by-token.
        """
        if not self.client:
            raise ValueError("LLM Client not initialized. Check Groq API Key.")

        messages = self._build_messages(query, citations, chat_history)

        try:
            for chunk in self.client.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            yield "\n[An error occurred while generating the response.]"


    def generate_answer(
        self, 
        query: str, 
        citations: List[Dict[str, Any]], 
        chat_history: Optional[List[Any]] = None
    ) -> str:
        """
        Synchronous fallback method for non-streaming calls.
        """
        if not self.client:
            raise ValueError("LLM Client not initialized. Check Groq API Key.")

        messages = self._build_messages(query, citations, chat_history)

        try:
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "An error occurred while generating the response."

# ---------------------------------------------------------------------------
# Conversational retrieval: give the SEARCH query its missing topic.
# ---------------------------------------------------------------------------

def build_search_query(
    query: str,
    chat_history: Optional[List[Any]] = None,
    rewritten: Optional[str] = None,
    max_turns: int = 2,
    max_chars: int = 400,
) -> str:
    """Returns the string that should be sent to retrieval (NOT to the LLM).

    Follow-up questions carry their topic in a pronoun -- "how do I use
    cross-validation to detect and prevent it?" -- so the bare query has
    nothing for dense or sparse search to match on. The topic word almost
    always sits in the previous assistant turn.

    Order of preference:
      1. The rewriter's output, when it actually rewrote something. A model
         that resolved the referent produces a better standalone query than
         concatenation does.
      2. Otherwise the last `max_turns` messages prepended to the query. This
         is the deterministic fallback for when no rewriter is configured, the
         provider is rate-limited, or the call failed -- all of which currently
         make rewrite_query() hand back the raw pronoun query untouched.
      3. The raw query, when there is no history to draw on.

    Measured on the multi_turn slice of the golden set (n=22), retrieval only,
    no rewriter available:
        baseline        r@3 0.00   r@10 0.00   MRR 0.019
        history concat  r@3 0.05   r@10 0.18   MRR 0.052

    Known gap: this helps follow-ups and hurts topic switches ("never mind,
    what is a p-value?"), because the old topic gets dragged into the query.
    The golden set contains no topic-switch cases, so that cost is unmeasured.
    Gate on a follow-up signal (short query, pronoun present, no standalone
    topic noun) before treating this as free.
    """
    query = (query or "").strip()
    if rewritten and rewritten.strip() and rewritten.strip() != query:
        return rewritten.strip()

    turns = LLMService._format_history(chat_history)[-max_turns:]
    if not turns:
        return query

    context = " ".join(m["content"] for m in turns).strip()
    if len(context) > max_chars:
        context = context[-max_chars:]
    return f"{context} {query}".strip()
