import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
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
                **client_kwargs
            )
        else:
            self.client = None
            logger.warning("GROQ_API_KEY missing in SETTINGS. LLMService disabled.")

        openrouter_key = SETTINGS.OPENROUTER_API_KEY
        if openrouter_key:
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
            logger.warning("OPENROUTER_API_KEY missing in SETTINGS. OpenRouter query rewriter disabled.")

        gemini_key = SETTINGS.GEMINI_API_KEY
        if gemini_key and gemini_key.startswith("AIzaSy"):
            self.gemini_client = ChatGoogleGenerativeAI(
                api_key=gemini_key,
                model="gemini-1.5-flash",
                temperature=rewrite_cfg["temperature"]
            )
        else:
            self.gemini_client = None
            logger.warning("Valid GEMINI_API_KEY (starting with 'AIzaSy') missing in SETTINGS. Gemini query rewriter disabled.")

        # Local rewriter model (only for dev/eval, avoids API limits completely)
        from pathlib import Path
        local_model_path = Path(__file__).resolve().parents[2] / "data" / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        self._local_rewriter = None
        if local_model_path.exists():
            try:
                from llama_cpp import Llama
                logger.info(f"Initializing local query rewriter from {local_model_path.name}...")
                self._local_rewriter = Llama(
                    model_path=str(local_model_path),
                    n_ctx=2048,
                    n_threads=4,
                    chat_format="chatml",
                    verbose=False
                )
            except Exception as e:
                logger.warning(f"Failed to load local query rewriter model: {e}. Falling back to API.")

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
            if role and content:
                normalized.append({"role": role, "content": content})
        return normalized

    def _format_context(self, citations: List[Dict[str, Any]]) -> str:
        """Renders retrieved chunks into the numbered citation blocks the prompt expects."""
        return "".join(
            prompts.render(
                "answer", "citation_block", variant=self.answer_variant,
                index=idx,
                source=cite.get("source_file", "Unknown"),
                page=cite.get("page_number", "N/A"),
                snippet=cite.get("text_snippet", "").strip()
            )
            for idx, cite in enumerate(citations, start=1)
        )

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
        Uses a local model (falling back to OpenRouter or Gemini if configured) to check the query for clarity, 
        resolve history references, and generate a search-friendly query. 
        
        If the query is ambiguous, it returns 'CLARIFICATION_REQUIRED: <question>'.
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

        # 1. Prioritize Local GGUF Rewriter (Offline, avoids API rate limits completely)
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
                logger.warning(f"Local query rewriter failed: {e}. Falling back to API.")

        # 2. Fall back to APIs
        client_to_use = self.openrouter_client or self.gemini_client
        if not client_to_use:
            return query

        try:
            response = client_to_use.invoke([HumanMessage(content=prompt)])
            rewritten = response.content.strip()
            logger.info(f"API query analysis for '{query}' returned: '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"Failed to analyze query with API: {e}")
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