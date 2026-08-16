import os
from pathlib import Path
from dotenv import load_dotenv

# Find the project root directory where .env is located
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Force Hugging Face Hub offline mode so sentence-transformers doesn't try
# to reach out to the network to verify local cache hashes on startup.
os.environ["HF_HUB_OFFLINE"] = "1"

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")

SETTINGS = Settings()
