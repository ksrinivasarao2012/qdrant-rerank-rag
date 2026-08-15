"""
Component-level generator evaluation (see plan.md Part E, layer 1).

Tests the generator in isolation: real query + hand-fed golden context
(the actual accepted-answer text, bypassing retrieval entirely) -> LLM
answer -> judged for Faithfulness (does it stick to the given context, no
hallucination) and Answer Relevancy (does it address the query). Both are
reference-free LLM-judge metrics via DeepEval.

Context is joined from data/processed/posts.jsonl by answer_id at eval time
-- gold answer text is NOT duplicated into golden_dataset.json.

Skips cases with no gold_answer_ids (adversarial/out_of_scope -- those are
refusal tests, not generation-quality tests) and out_of_scope-style entries
by construction, since they carry no context to hand-feed.

Requires GROQ_API_KEY for the generator under test. The judge now runs
locally (Qwen2.5-7B-Instruct GGUF via llama-cpp-python, see judge_model.py
for setup) -- no API key, no rate limit, but you do need the model file
downloaded first and `pip install -r evaluation/requirements.txt` run.
Does not require Qdrant/the seeded corpus at all, since retrieval is
bypassed by design.

Run with: python evaluation/eval_generator.py
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.llm_service import LLMService
from backend.core.config import SETTINGS
from evaluation.judge_model import get_judge

GOLDEN_JSON_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
POSTS_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "posts.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

FAITHFULNESS_THRESHOLD = 0.5
ANSWER_RELEVANCY_THRESHOLD = 0.5


def load_cases(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        print(f"Error: {path.resolve()} is missing or empty. "
              f"Run backend/scripts/build_golden_dataset.py first.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_answer_lookup(jsonl_path: Path, needed_ids: set) -> dict:
    """Streams posts.jsonl once, keeping only the records this eval run
    actually needs (by answer_id) -- avoids loading all ~93k answers into
    memory just to use a couple hundred of them."""
    lookup = {}
    if not needed_ids:
        return lookup
    remaining = set(needed_ids)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            line = line.strip()
            if not line:
                continue
            post = json.loads(line)
            if post["answer_id"] in remaining:
                lookup[post["answer_id"]] = post
                remaining.discard(post["answer_id"])
    if remaining:
        print(f"Warning: {len(remaining)} gold answer_ids from golden_dataset.json "
              f"were not found in posts.jsonl (corpus may have changed since the "
              f"golden set was built): {sorted(remaining)[:10]}...")
    return lookup


def build_citations(gold_ids, lookup: dict):
    """Same citation shape routes.py builds from live retrieval results
    (question_title as source, score, accepted flag, thread URL, text) --
    kept identical so the generator sees the same prompt structure it would
    in production, just with hand-picked context instead of retrieved context."""
    citations = []
    for aid in gold_ids:
        post = lookup.get(int(aid))
        if not post:
            continue
        citations.append({
            "source_file": post.get("question_title", "Untitled question"),
            "score": post.get("score", 0),
            "is_accepted": post.get("is_accepted", False),
            "url": post.get("url", ""),
            "text_snippet": post.get("answer_text", ""),
        })
    return citations


def query_for_case(case: dict):
    """Generator eval runs one query per case (not every paraphrase variant
    like the retriever eval). Each case costs 1 Groq generation call plus
    however many local judge calls DeepEval's metrics make internally --
    Faithfulness and AnswerRelevancy each do multiple judge sub-steps (claim
    extraction, verdicts, etc.), so real judge-call volume per case is
    closer to 4-8 than 2. The judge now runs locally with no rate limit, but
    each of those calls is a real CPU generation (see judge_model.py), so
    keeping it to one query per case is still what keeps total runtime
    bounded. Paraphrase groups use their base query (first variant)."""
    if case["category"] == "paraphrase_group":
        variants = case.get("variants", [])
        return variants[0] if variants else None
    return case.get("query")


def evaluate():
    if not SETTINGS.GROQ_API_KEY:
        print("Error: GROQ_API_KEY not set -- required for the generator under test "
              "(the judge now runs locally and needs no API key -- see judge_model.py).")
        return 1

    cases = load_cases(GOLDEN_JSON_PATH)
    if cases is None:
        return 1

    evaluable = [c for c in cases if c.get("gold_answer_ids") and query_for_case(c)]
    skipped = len(cases) - len(evaluable)
    print(f"Loaded {len(cases)} cases ({len(evaluable)} evaluable, "
          f"{skipped} skipped -- no gold_answer_ids/query, e.g. adversarial/out_of_scope).")

    all_gold_ids = {int(aid) for c in evaluable for aid in c["gold_answer_ids"]}
    print(f"Joining {len(all_gold_ids)} gold answers from posts.jsonl...")
    lookup = build_answer_lookup(POSTS_JSONL_PATH, all_gold_ids)

    print("Initializing generator (Groq: gpt-oss-20b) and judge (local: Qwen2.5-7B-Instruct GGUF)...")
    llm_service = LLMService()
    judge = get_judge()

    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    faithfulness_metric = FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=judge, include_reason=True)
    relevancy_metric = AnswerRelevancyMetric(threshold=ANSWER_RELEVANCY_THRESHOLD, model=judge, include_reason=True)

    rows = []
    for case in evaluable:
        query = query_for_case(case)
        citations = build_citations(case["gold_answer_ids"], lookup)
        if not citations:
            print(f"  [{case['query_id']}] SKIPPED: gold answer text not found in posts.jsonl")
            continue

        chat_history = case.get("chat_history")
        try:
            answer = llm_service.generate_answer(query, citations, chat_history=chat_history)
        except Exception as e:
            print(f"  [{case['query_id']}] generation failed: {e}")
            continue

        retrieval_context = [c["text_snippet"] for c in citations]
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=retrieval_context,
        )

        row = {
            "query_id": case["query_id"],
            "category": case["category"],
            "query": query,
            "answer": answer,
        }
        try:
            faithfulness_metric.measure(test_case)
            row["faithfulness_score"] = faithfulness_metric.score
            row["faithfulness_reason"] = faithfulness_metric.reason
        except Exception as e:
            row["faithfulness_score"] = None
            row["faithfulness_error"] = str(e)

        try:
            relevancy_metric.measure(test_case)
            row["answer_relevancy_score"] = relevancy_metric.score
            row["answer_relevancy_reason"] = relevancy_metric.reason
        except Exception as e:
            row["answer_relevancy_score"] = None
            row["answer_relevancy_error"] = str(e)

        rows.append(row)
        print(f"  [{case['query_id']}] faithfulness={row.get('faithfulness_score')} "
              f"relevancy={row.get('answer_relevancy_score')}")

    if not rows:
        print("No cases produced results -- nothing to report.")
        return 1

    print_summary(rows)
    save_results(rows)
    return 0


def print_summary(rows):
    def agg(subset, key):
        scores = [r[key] for r in subset if r.get(key) is not None]
        return sum(scores) / len(scores) if scores else None

    print("\n=== OVERALL ===")
    f_avg = agg(rows, "faithfulness_score")
    r_avg = agg(rows, "answer_relevancy_score")
    print(f"  n={len(rows)}  faithfulness={f_avg:.3f}  answer_relevancy={r_avg:.3f}"
          if f_avg is not None and r_avg is not None
          else f"  n={len(rows)}  (some scores missing -- see per-case errors above)")

    print("\n=== BY CATEGORY ===")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        subset = by_cat[cat]
        f = agg(subset, "faithfulness_score")
        rel = agg(subset, "answer_relevancy_score")
        f_str = f"{f:.3f}" if f is not None else "n/a"
        rel_str = f"{rel:.3f}" if rel is not None else "n/a"
        print(f"  {cat:20s} n={len(subset):3d}  faithfulness={f_str}  answer_relevancy={rel_str}")


def save_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"generator_eval_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved per-case results to {out_path.resolve()}")


if __name__ == "__main__":
    sys.exit(evaluate())
