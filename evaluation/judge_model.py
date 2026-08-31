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

load_dotenv(PROJECT_ROOT / ".env", override=True)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "qwen2.5-7b-instruct-q4_k_m.gguf"
MODEL_PATH = Path(os.getenv("LOCAL_JUDGE_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

N_CTX = 8192       # Qwen2.5's native context; generous enough for a full
                    # retrieved-context + answer + judge-prompt turn
N_THREADS = 4
MAX_TOKENS = 1024  # DeepEval's verdict/claim-extraction outputs are
                    # structured JSON, not long-form -- no need for more


def clean_json_response(text: str) -> str:
    cleaned = text.strip()
    import re
    import json
    # Remove <think>...</think> tags and everything inside them
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    
    # Remove markdown code block wrappers (e.g. ```json ... ```)
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].strip() if len(cleaned.split("```")) > 1 else cleaned
    
    # Use JSONDecoder.raw_decode to extract precisely one valid JSON structure and drop all trailing text!
    try:
        start_idx = -1
        for i, ch in enumerate(cleaned):
            if ch in ('{', '['):
                start_idx = i
                break
        if start_idx != -1:
            decoder = json.JSONDecoder()
            obj, end_idx = decoder.raw_decode(cleaned[start_idx:])
            return json.dumps(obj)
    except Exception:
        pass

    return cleaned.strip()


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
                p.cpu_affinity([0, 1, 2, 3])
                print("Process priority set to BELOW_NORMAL. CPU affinity restricted to Cores 0,1,2,3.")
            except Exception as pe:
                print(f"Failed to set process priority/affinity: {pe}")

            try:
                # 1. Try loading with GPU offloading enabled
                self._model = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_threads=N_THREADS,
                    n_gpu_layers=-1,
                    verbose=False,
                )
                print("Local GGUF judge loaded with GPU offloading.")
            except Exception as gpu_err:
                print(f"Failed to load GGUF model with GPU: {gpu_err}. Falling back to CPU.")
                self._model = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_threads=N_THREADS,
                    n_gpu_layers=0,
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
        return clean_json_response(response["choices"][0]["message"]["content"])

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


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = None):
        # Default to the highly accurate qwen/qwen3.6-27b on Groq
        self.model_name = model_name or os.getenv("GROQ_JUDGE_MODEL", "qwen/qwen3.6-27b")
        self.client = None
        super().__init__(self.model_name)

    def load_model(self):
        if self.client is None:
            from langchain_groq import ChatGroq
            self.client = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model_name=self.model_name,
                temperature=0.0,
                max_retries=6,
            )
        return self.client

    def _chat(self, prompt: str) -> str:
        client = self.load_model()
        from langchain_core.messages import HumanMessage
        response = client.invoke([HumanMessage(content=prompt)])
        return clean_json_response(response.content)

    def generate(self, prompt: str) -> str:
        return self._chat(prompt)

    async def a_generate(self, prompt: str) -> str:
        client = self.load_model()
        from langchain_core.messages import HumanMessage
        response = await client.ainvoke([HumanMessage(content=prompt)])
        return clean_json_response(response.content)

    def get_model_name(self) -> str:
        return self.model_name


class GeminiJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "gemini-flash-latest"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        super().__init__(self.model_name)

    def load_model(self):
        return self

    def _chat(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        import urllib.request
        import json
        
        # Using Google Generative Language API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json"
            }
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                res_data = json.loads(r.read().decode("utf-8"))
                text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                return clean_json_response(text)
        except Exception as e:
            if hasattr(e, "read"):
                err_content = e.read().decode("utf-8")
                raise RuntimeError(f"Gemini API Error: {err_content}")
            raise e

    def generate(self, prompt: str) -> str:
        return self._chat(prompt)

    async def a_generate(self, prompt: str) -> str:
        import asyncio
        return await asyncio.to_thread(self._chat, prompt)

    def get_model_name(self) -> str:
        return self.model_name


def get_judge():
    judge_type = os.getenv("EVAL_JUDGE_TYPE", "local").lower().strip()
    if judge_type == "groq":
        return GroqJudge()
    elif judge_type == "gemini":
        return GeminiJudge()
    return LocalGGUFJudge()
