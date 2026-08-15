"""
Application-level evaluation (see plan.md Part E, layer 3 -- "the whole
product... judged as a deliverable, not mechanics").

Deliberately does NOT re-run generation. eval_pipeline.py already produces
real answers from the full live stack; re-generating a third time here
(after eval_generator.py's isolated pass and eval_pipeline.py's full-stack
pass) would triple the Groq + judge cost for the same ~150 cases with
nothing gained. Instead, this reads the most recent
evaluation/results/pipeline_eval_*.json and judges those existing answers
against three more criteria DeepEval's built-in metrics don't cover:

  - Correctness: does the answer actually match the real accepted-answer
    text for the question (joined from posts.jsonl), not just "faithful to
    whatever got retrieved" (which eval_pipeline.py already checks, but a
    faithful answer built from weak/off-topic context can still be wrong).
  - Completeness (multi_hop cases only): does the answer address BOTH
    sub-topics, using the two gold posts' own question titles as the
    "what must be covered" signal -- no hand-authored checklist needed,
    since multi_hop cases already guarantee two distinct, verified topics
    (see build_golden_dataset.py's include_a/include_b fix).
  - Style: is the answer clear and appropriately toned, independent of
    whether it's factually right.

All three are DeepEval GEval (custom LLM-judge criteria), scored by the
same local GGUF judge as the other eval scripts -- see judge_model.py.
Safety is NOT re-implemented here -- it's already covered by the
adversarial/out_of_scope categories' expect_refusal checks, which are a
pass/fail runtime property, not something GEval-graded. Operations
(latency/cost/tokens) also isn't here -- that's runtime instrumentation,
not something judged from saved answers.

Run with: python evaluation/eval_application.py
(after eval_pipeline.py has produced at least one results file)
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.judge_model import get_judge

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
POSTS_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "posts.jsonl"

CORRECTNESS_THRESHOLD = 0.5
COMPLETENESS_THRESHOLD = 0.5
STYLE_THRESHOLD = 0.5


def latest_pipeline_results() -> Path:
    candidates = sorted(RESULTS_DIR.glob("pipeline_eval_*.json"))
    return candidates[-1] if candidates else None


def build_answer_lookup(jsonl_path: Path, needed_ids: set) -> dict:
    """Same approach as eval_generator.py -- stream posts.jsonl once,
    keeping only the records this run actually needs."""
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
    return lookup


def evaluate():
    results_path = latest_pipeline_results()
    if results_path is None:
        print(f"Error: no pipeline_eval_*.json found in {RESULTS_DIR.resolve()}. "
              f"Run evaluation/eval_pipeline.py first.")
        return 1

    print(f"Reading pipeline results from {results_path.name}...")
    with open(results_path, "r", encoding="utf-8") as f:
        pipeline_rows = json.load(f)

    if not pipeline_rows:
        print("Pipeline results file is empty -- nothing to judge.")
        return 1

    all_gold_ids = {int(aid) for r in pipeline_rows for aid in r.get("gold_answer_ids", [])}
    print(f"Joining {len(all_gold_ids)} gold answers from posts.jsonl for correctness/completeness checks...")
    lookup = build_answer_lookup(POSTS_JSONL_PATH, all_gold_ids)

    print("Loading judge (local GGUF)...")
    judge = get_judge()

    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    correctness_metric = GEval(
        name="Correctness",
        criteria=(
            "Determine whether the ACTUAL OUTPUT is factually correct and consistent "
            "with the EXPECTED OUTPUT (a real, community-vetted reference answer to the "
            "same question). Minor differences in wording or level of detail are fine; "
            "penalize contradictions, fabricated claims, or missing the core answer."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=CORRECTNESS_THRESHOLD,
    )
    style_metric = GEval(
        name="Style",
        criteria=(
            "Determine whether the ACTUAL OUTPUT is clear, concise, and appropriately "
            "toned for a technical statistics/ML Q&A assistant -- direct, no unnecessary "
            "hedging, no filler, no excessive apologizing."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=STYLE_THRESHOLD,
    )

    rows = []
    for r in pipeline_rows:
        query = r["query"]
        answer = r["answer"]
        gold_ids = [int(aid) for aid in r.get("gold_answer_ids", [])]
        gold_posts = [lookup[aid] for aid in gold_ids if aid in lookup]

        row = {
            "query_id": r["query_id"],
            "category": r["category"],
            "query": query,
        }

        if gold_posts:
            expected_output = "\n\n---\n\n".join(p["answer_text"] for p in gold_posts)
            test_case = LLMTestCase(input=query, actual_output=answer, expected_output=expected_output)
            try:
                correctness_metric.measure(test_case)
                row["correctness_score"] = correctness_metric.score
                row["correctness_reason"] = correctness_metric.reason
            except Exception as e:
                row["correctness_score"] = None
                row["correctness_error"] = str(e)
        else:
            row["correctness_score"] = None
            row["correctness_error"] = "no gold answer text found in posts.jsonl"

        if r["category"] == "multi_hop" and len(gold_posts) == 2:
            topic_a, topic_b = gold_posts[0]["question_title"], gold_posts[1]["question_title"]
            completeness_metric = GEval(
                name="Completeness",
                criteria=(
                    f"The question requires addressing TWO distinct sub-topics: "
                    f"(1) '{topic_a}' and (2) '{topic_b}'. Determine whether the "
                    f"ACTUAL OUTPUT meaningfully addresses BOTH sub-topics, not just one."
                ),
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                model=judge,
                threshold=COMPLETENESS_THRESHOLD,
            )
            test_case = LLMTestCase(input=query, actual_output=answer)
            try:
                completeness_metric.measure(test_case)
                row["completeness_score"] = completeness_metric.score
                row["completeness_reason"] = completeness_metric.reason
            except Exception as e:
                row["completeness_score"] = None
                row["completeness_error"] = str(e)
        else:
            row["completeness_score"] = None  # not applicable outside multi_hop

        style_test_case = LLMTestCase(input=query, actual_output=answer)
        try:
            style_metric.measure(style_test_case)
            row["style_score"] = style_metric.score
            row["style_reason"] = style_metric.reason
        except Exception as e:
            row["style_score"] = None
            row["style_error"] = str(e)

        rows.append(row)
        print(f"  [{row['query_id']}] correctness={row.get('correctness_score')} "
              f"completeness={row.get('completeness_score')} style={row.get('style_score')}")

    print_summary(rows)
    save_results(rows)
    return 0


def print_summary(rows):
    def agg(subset, key):
        scores = [r[key] for r in subset if r.get(key) is not None]
        return sum(scores) / len(scores) if scores else None

    print("\n=== OVERALL ===")
    for key, label in [
        ("correctness_score", "correctness"),
        ("style_score", "style"),
    ]:
        avg = agg(rows, key)
        print(f"  {label}: {avg:.3f}" if avg is not None else f"  {label}: n/a")

    multi_hop_rows = [r for r in rows if r["category"] == "multi_hop"]
    completeness_avg = agg(multi_hop_rows, "completeness_score")
    print(f"  completeness (multi_hop only, n={len(multi_hop_rows)}): "
          f"{completeness_avg:.3f}" if completeness_avg is not None else "  completeness: n/a")

    print("\n=== BY CATEGORY ===")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        subset = by_cat[cat]
        corr = agg(subset, "correctness_score")
        style = agg(subset, "style_score")
        fmt = lambda v: f"{v:.2f}" if v is not None else "n/a"
        print(f"  {cat:20s} n={len(subset):3d}  correctness={fmt(corr)}  style={fmt(style)}")


def save_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"application_eval_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved per-case results to {out_path.resolve()}")


if __name__ == "__main__":
    sys.exit(evaluate())
