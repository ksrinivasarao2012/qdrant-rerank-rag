"""
Pre-flight smoke test for the eval stack.

Run this BEFORE a full 79-case eval run. It isolates and tests each external
dependency individually (package installed? key valid? call succeeds? real
output, not a silent None?) so a broken piece fails loudly in ~30 seconds
instead of being discovered 9 minutes into a full run, or worse -- discovered
never, because a silent fallback made the report look fine.

This does NOT touch retrieval, the vector DB, or the reranker -- it only
checks the three things that broke today: the judge model, and the
generator-side LLM providers (Groq / Cerebras / Gemini) used for query
rewrite and Pillar 2 decomposition.

Run with:
    python evaluation/smoke_test.py
"""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

results = []  # (name, status, detail)


def check(name):
    """Decorator-ish helper: runs fn, records PASS/FAIL/SKIP, never raises."""
    def wrapper(fn):
        print(f"\n--- {name} " + "-" * max(1, 60 - len(name)))
        t0 = time.time()
        try:
            detail = fn()
            dt = time.time() - t0
            if detail is False:
                results.append((name, "SKIP", "no key configured"))
                print(f"  SKIP ({dt:.1f}s) -- no key configured, nothing to test")
            else:
                results.append((name, "PASS", str(detail)[:200]))
                print(f"  PASS ({dt:.1f}s): {detail}")
        except Exception as e:
            dt = time.time() - t0
            results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
            print(f"  FAIL ({dt:.1f}s): {type(e).__name__}: {e}")
    return wrapper


# ---------------------------------------------------------------------------
# 1. Package import checks -- these fail SILENTLY elsewhere in the codebase
#    (llm_service.py wraps them in try/except ImportError -> None), so a
#    missing package never raises where you'd see it. Check explicitly here.
# ---------------------------------------------------------------------------
@check("Package: deepeval")
def _():
    import deepeval
    return f"deepeval {getattr(deepeval, '__version__', '?')} importable"


@check("Package: langchain_google_genai (Gemini)")
def _():
    import langchain_google_genai
    return f"importable, version {getattr(langchain_google_genai, '__version__', '?')}"


@check("Package: langchain_openai (used for Cerebras client)")
def _():
    import langchain_openai
    return f"importable, version {getattr(langchain_openai, '__version__', '?')}"


# ---------------------------------------------------------------------------
# 2. Gemini judge -- raw call, bypassing DeepEval entirely, to isolate
#    "is the key/model valid" from "does DeepEval's plumbing work".
# ---------------------------------------------------------------------------
@check("Gemini judge: raw generate() call")
def _():
    from evaluation.judge_model import GeminiJudge
    if not os.getenv("GEMINI_API_KEY"):
        return False
    j = GeminiJudge()
    out = j.generate('Reply with exactly this JSON and nothing else: {"ok": true}')
    if not out or "ok" not in out:
        raise RuntimeError(f"got back unexpected content: {out!r}")
    return out


# ---------------------------------------------------------------------------
# 3. Gemini judge through the REAL DeepEval metric -- this is the exact code
#    path that silently failed in both prior runs. If this passes, the full
#    eval's judge scores are real. If this fails, you now see why.
# ---------------------------------------------------------------------------
@check("Gemini judge: real DeepEval ContextualRecallMetric")
def _():
    import asyncio
    from deepeval.metrics import ContextualRecallMetric
    from deepeval.test_case import LLMTestCase
    from evaluation.judge_model import GeminiJudge

    if not os.getenv("GEMINI_API_KEY"):
        return False

    test_case = LLMTestCase(
        input="What test can I use to check normality?",
        actual_output="The Shapiro-Wilk test is commonly used to check if data follows a normal distribution.",
        expected_output="The Shapiro-Wilk test is a statistical test used to check whether a sample comes from a normally distributed population.",
        retrieval_context=["The Shapiro-Wilk test is commonly used to check if data follows a normal distribution."],
    )
    metric = ContextualRecallMetric(threshold=0.5, model=GeminiJudge(), include_reason=True)
    asyncio.run(metric.a_measure(test_case))
    if metric.score is None:
        raise RuntimeError("metric.score is None after a_measure -- judge silently failed")
    return f"score={metric.score}, reason={metric.reason!r}"


# ---------------------------------------------------------------------------
# 4. Cerebras -- raw client call via LLMService, bypassing rewrite_query's
#    fallback cascade so a failure here can't hide behind Groq happening to
#    succeed first.
# ---------------------------------------------------------------------------
@check("Cerebras client: raw invoke()")
def _():
    from langchain_core.messages import HumanMessage
    from backend.core.llm_service import LLMService

    if not os.getenv("CEREBRAS_API_KEY"):
        return False

    svc = LLMService()
    if svc.cerebras_client is None:
        raise RuntimeError(
            "CEREBRAS_API_KEY is set but svc.cerebras_client is None -- "
            "likely langchain_openai isn't installed (see package check above)"
        )
    resp = svc.cerebras_client.invoke([HumanMessage(content="Reply with exactly the word: OK")])
    return resp.content.strip()


# ---------------------------------------------------------------------------
# 5. Hugging Face Serverless -- already wired into rewrite_query()'s fallback
#    chain via HF_TOKEN (already set in .env), never actually verified.
#    Genuinely free tier, no card, no new signup needed.
# ---------------------------------------------------------------------------
@check("Hugging Face client: raw invoke()")
def _():
    from langchain_core.messages import HumanMessage
    from backend.core.llm_service import LLMService

    if not os.getenv("HF_TOKEN"):
        return False

    svc = LLMService()
    if svc.hf_client is None:
        raise RuntimeError(
            "HF_TOKEN is set but svc.hf_client is None -- "
            "likely langchain_openai isn't installed (see package check above)"
        )
    resp = svc.hf_client.invoke([HumanMessage(content="Reply with exactly the word: OK")], config={"timeout": 10.0})
    return resp.content.strip()


# ---------------------------------------------------------------------------
# 6. GitHub Models -- also already wired in via GITHUB_TOKEN, free tier tied
#    to a GitHub account, no separate signup or card. Only runs if you set
#    GITHUB_TOKEN in .env (a GitHub personal access token).
# ---------------------------------------------------------------------------
@check("GitHub Models client: raw invoke()")
def _():
    from langchain_core.messages import HumanMessage
    from backend.core.llm_service import LLMService

    if not os.getenv("GITHUB_TOKEN"):
        return False

    svc = LLMService()
    if svc.github_client is None:
        raise RuntimeError(
            "GITHUB_TOKEN is set but svc.github_client is None -- "
            "likely langchain_openai isn't installed (see package check above)"
        )
    resp = svc.github_client.invoke([HumanMessage(content="Reply with exactly the word: OK")], config={"timeout": 10.0})
    return resp.content.strip()


# ---------------------------------------------------------------------------
# 7. Groq -- quota probe. Doesn't fail the suite on 429 (that's an expected,
#    known state today) but tells you plainly whether it's usable right now.
# ---------------------------------------------------------------------------
@check("Groq: quota probe (openai/gpt-oss-20b)")
def _():
    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq

    if not os.getenv("GROQ_API_KEY"):
        return False

    client = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="openai/gpt-oss-20b", temperature=0.0, max_tokens=8)
    resp = client.invoke([HumanMessage(content="Reply with exactly the word: OK")])
    return resp.content.strip()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SMOKE TEST SUMMARY")
print("=" * 70)
worst = "PASS"
for name, status, detail in results:
    marker = {"PASS": "OK  ", "FAIL": "FAIL", "SKIP": "skip"}[status]
    print(f"[{marker}] {name}")
    if status == "FAIL":
        worst = "FAIL"

judge_ok = any(n == "Gemini judge: real DeepEval ContextualRecallMetric" and s == "PASS" for n, s, _ in results)
print("\n" + "-" * 70)
if judge_ok:
    print("Judge verdict: REAL DeepEval scores will be produced. Safe to trust")
    print("the contextual_recall_score field in your next full run.")
else:
    print("Judge verdict: DO NOT TRUST contextual_recall_score in a full run yet.")
    print("Every case will silently fall back to the hit-based heuristic (the")
    print("exact failure mode from both prior runs). Fix the FAILs above first.")
print("-" * 70)

sys.exit(1 if worst == "FAIL" else 0)
