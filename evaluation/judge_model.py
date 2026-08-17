"""
Shared DeepEval-compatible judge model. History of how this landed here,
kept because the reasoning matters if this ever gets revisited:

  - OpenAI (DeepEval's default): rejected, no OPENAI_API_KEY anywhere in
    this project and no reason to add a new paid dependency just to judge.
  - Groq, same model as the generator under test (openai/gpt-oss-20b):
    rejected, a model shouldn't grade its own output, and it would compound
    rate-limit pressure on the model the live app depends on.
  - Groq, a different model (qwen/qwen3.6-27b): worked around the above,
    but still an external API with its own (if more generous) rate limits
    and still something to think about every time this eval is re-run.
  - Landed here: a local GGUF model via llama-cpp-python. No API, no rate
    limit, no key, runs entirely offline. Explicitly NOT added to
    requirements.txt or backend/requirements.txt -- this is eval-only
    tooling and is never meant to ship with the deployed app (see
    evaluation/requirements.txt instead).

Model: Qwen2.5-7B-Instruct, GGUF Q4_K_M quantization. Chosen because
DeepEval's metrics work by having the judge extract claims and produce
structured (often JSON) verdicts across several internal calls per metric --
that needs reliable instruction-following and structured-output compliance,
which Qwen2.5-Instruct is specifically strong at relative to its size. Falls
back to Qwen2.5-3B-Instruct-GGUF if 7B is too slow on CPU -- see
LOCAL_JUDGE_MODEL_PATH below.

SETUP (manual, one-time -- this repo does not download the model for you):
  1. Download a GGUF build of Qwen2.5-7B-Instruct, e.g. from
     https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
     (file: qwen2.5-7b-instruct-q4_k_m.gguf, ~4.7GB)
  2. Place it at data/models/qwen2.5-7b-instruct-q4_k_m.gguf (or set
     LOCAL_JUDGE_MODEL_PATH in .env to wherever you put it).
  3. pip install -r evaluation/requirements.txt (installs llama-cpp-python,
     which is NOT in the main app's requirements files on purpose).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "qwen2.5-7b-instruct-q4_k_m.gguf"
MODEL_PATH = Path(os.getenv("LOCAL_JUDGE_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

N_CTX = 8192       # Qwen2.5's native context; generous enough for a full
                    # retrieved-context + answer + judge-prompt turn
N_THREADS = 2
MAX_TOKENS = 1024  # DeepEval's verdict/claim-extraction outputs are
                    # structured JSON, not long-form -- no need for more


class LocalGGUFJudge(DeepEvalBaseLLM):
    def __init__(self, model_path: Path = MODEL_PATH, n_ctx: int = N_CTX):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self._model = None
        super().__init__(str(self.model_path))

    def load_model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Judge model not found at {self.model_path.resolve()}. "
                    f"Download a GGUF build of Qwen2.5-7B-Instruct (see the "
                    f"setup instructions at the top of judge_model.py) and "
                    f"place it there, or set LOCAL_JUDGE_MODEL_PATH in .env."
                )
            from llama_cpp import Llama
            print(f"Loading local judge model from {self.model_path.name} "
                  f"(n_ctx={self.n_ctx}, n_threads={N_THREADS})...")
            try:
                import psutil
                p = psutil.Process()
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                p.cpu_affinity([0, 1])
                print("Process priority set to BELOW_NORMAL. CPU affinity restricted to Cores 0,1.")
            except Exception as pe:
                print(f"Failed to set process priority/affinity: {pe}")

            self._model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=N_THREADS,
                verbose=False,
            )
        return self._model

    def _chat(self, prompt: str) -> str:
        model = self.load_model()
        response = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        return response["choices"][0]["message"]["content"]

    def generate(self, prompt: str) -> str:
        return self._chat(prompt)

    async def a_generate(self, prompt: str) -> str:
        # llama-cpp-python has no native async API; DeepEval's async path
        # just needs an awaitable, so run the sync call in a worker thread
        # rather than blocking the event loop.
        import asyncio
        return await asyncio.to_thread(self._chat, prompt)

    def get_model_name(self) -> str:
        return self.model_path.stem


def get_judge() -> LocalGGUFJudge:
    return LocalGGUFJudge()
