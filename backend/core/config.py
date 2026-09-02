import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Find the project root directory where .env is located
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Models this app actually loads. Kept here (not imported from embeddings.py /
# reranker.py) to avoid a circular import at module-load time -- config.py is
# imported before those modules exist.
_REQUIRED_HF_MODELS = {"BAAI/bge-base-en-v1.5", "cross-encoder/ms-marco-MiniLM-L-6-v2"}


def _models_already_cached() -> bool:
    """True only if every model this app needs is already in the local HF cache."""
    try:
        from huggingface_hub import scan_cache_dir
        cached = {repo.repo_id for repo in scan_cache_dir().repos}
    except Exception:
        # Can't verify what's cached -- do not force offline mode; better to
        # let the load attempt reach the network than fail outright.
        return False
    return _REQUIRED_HF_MODELS.issubset(cached)


# Force Hugging Face Hub offline mode ONLY when the models are already cached,
# so sentence-transformers doesn't waste time re-verifying local cache hashes
# on a warm start. Previously this was set unconditionally, which meant a
# cold container with no cache yet (a fresh HF Space build, a fresh Docker
# volume) failed to load the models at all instead of downloading them once.
# An operator can still force one way or the other with $HF_HUB_OFFLINE --
# os.environ.setdefault leaves an explicit value alone.
if _models_already_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    CEREBRAS_API_KEY: str = os.getenv("CEREBRAS_API_KEY", "")
    JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")

    def validate(self) -> None:
        """Logs a clear, loud warning for missing required config at startup,
        instead of letting a missing key surface later as a confusing runtime
        error (e.g. "LLM Client not initialized") on the first real request.

        Deliberately does not raise/exit: some entrypoints (evaluation
        scripts, local retrieval-only testing) run fine without GROQ_API_KEY,
        so refusing to start would be the over-eager failure mode, not the
        helpful one. Called from both entrypoints' startup path (app.py,
        backend/main.py), not at import time, so importing this module never
        has a side effect beyond reading env vars.
        """
        if not self.GROQ_API_KEY:
            logger.warning(
                "GROQ_API_KEY is not set. Answer generation and query "
                "rewriting will fail on first use. Set it in .env or the "
                "environment before serving real traffic."
            )
        # No other key is required -- every other client in llm_service.py is
        # an optional fallback that is simply skipped when its key is absent.


SETTINGS = Settings()
