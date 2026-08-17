"""
Pipeline-level evaluation (see plan.md Part E, layer 2 -- the RAG triad).

Unlike eval_retriever.py (retriever alone) and eval_generator.py (generator
alone, hand-fed golden context), this runs the FULL live stack exactly as
routes.py does: query rewrite (Gemini) -> hybrid search -> cross-encoder
rerank -> generation (Groq) -- then judges the result on real, actually-
retrieved context, not a hand-picked answer.

Mirrors backend/api/routes.py's wiring exactly (same CANDIDATE_K, same
top_k default, same citation shape, same query-rewritten-for-retrieval-but-
original-query-for-generation split) so this eval reflects what a user
would actually get, not an idealized version of the pipeline.

Metrics (DeepEval, judged by the same local GGUF model as eval_generator.py
-- see judge_model.py):
  - ContextualRelevancyMetric: is what got retrieved actually relevant to
    the query? (tests retrieval quality end-to-end, including the rewriter)
  - FaithfulnessMetric: does the answer stick to the retrieved context?
  - AnswerRelevancyMetric: does the answer address the query?

Requires GROQ_API_KEY (generation), GEMINI_API_KEY (query rewriting), and
the corpus actually seeded in Qdrant (unlike eval_generator.py, retrieval
is NOT bypassed here -- that's the point of this layer).

Run with: python evaluation/eval_pipeline.py
"""

import sys
import json
import time
import os
from pathlib import Path
from collections import defaultdict

# Prevent PyTorch deadlocks & HuggingFace network hangs
for _v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{_v}_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.vector_store import VectorDBManager
from backend.core.reranker import ReRanker
from backend.core.llm_service import LLMService
from backend.core.config import SETTINGS
from evaluation.judge_model import get_judge

GOLDEN_JSON_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
POSTS_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "posts.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

CANDIDATE_K = 15   # matches routes.py
TOP_K = 3          # matches QueryRequest's default

CONTEXTUAL_RELEVANCY_THRESHOLD = 0.5
FAITHFULNESS_THRESHOLD = 0.5
ANSWER_RELEVANCY_THRESHOLD = 0.5
CONTEXTUAL_RECALL_THRESHOLD = 0.5


def load_cases(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        print(f"Error: {path.resolve()} is missing or empty. "
              f"Run backend/scripts/build_golden_dataset.py first.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gold_posts_map(posts_path: Path, needed_ids: set) -> dict:
    """Reads posts.jsonl and extracts answer_text for requested gold_answer_ids."""
    out = {}
    if not posts_path.exists():
        return out
    with open(posts_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            post = json.loads(line)
            aid = str(post.get("answer_id"))
            if aid in needed_ids:
                out[aid] = post.get("answer_text", "")
                if len(out) == len(needed_ids):
                    break
    return out


def query_for_case(case: dict):
    """One query per case (not every paraphrase variant) -- each case here
    costs a rewrite call, a retrieval call, a generation call, and multiple
    judge calls across 3 metrics, so keeping it to one query per case is
    what keeps a full run's wall-clock time bounded. Paraphrase groups use
    their base query (first variant)."""
    if case["category"] == "paraphrase_group":
        variants = case.get("variants", [])
        return variants[0] if variants else None
    return case.get("query")


def build_citations(reranked_results):
    """Identical shape to routes.py's citations_list -- kept the same on
    purpose so the generator sees the same prompt structure in eval as in
    production."""
    citations = []
    for res in reranked_results:
        citations.append({
            "source_file": res["metadata"].get("question_title", "Untitled question"),
            "score": res["metadata"].get("score", 0),
            "is_accepted": res["metadata"].get("is_accepted", False),
            "url": res["metadata"].get("url", ""),
            # display_text (plain chunk, no title/overlap prefix) if present;
            # falls back to the full embedded text for chunks indexed before
            # this field existed. Must match routes.py exactly -- this eval's
            # whole point is mirroring production wiring, and text_snippet is
            # what actually becomes the LLM's context during generation.
            "text_snippet": res["metadata"].get("display_text", res["text"]),
        })
    return citations


def evaluate():
    if not SETTINGS.GROQ_API_KEY:
        print("Error: GROQ_API_KEY not set -- required for generation.")
        return 1
    if not (SETTINGS.GEMINI_API_KEY and SETTINGS.GEMINI_API_KEY.startswith("AIzaSy")):
        print("Error: valid GEMINI_API_KEY not set -- required for query rewriting.")
        return 1

    cases = load_cases(GOLDEN_JSON_PATH)
    if cases is None:
        return 1

    # Same eligibility as eval_generator.py: needs a gold answer to be a
    # meaningful RAG-triad case (context relevance/faithfulness need
    # something to judge against being "the right topic"). Adversarial and
    # out_of_scope are refusal tests, not triad-quality tests -- skipped
    # here the same way, for the same reason.
    evaluable = [c for c in cases if c.get("gold_answer_ids") and query_for_case(c)]
    skipped = len(cases) - len(evaluable)
    print(f"Loaded {len(cases)} cases ({len(evaluable)} evaluable, "
          f"{skipped} skipped -- no gold_answer_ids/query, e.g. adversarial/out_of_scope).")

    all_needed_gold_ids = {str(gid) for c in evaluable for gid in c.get("gold_answer_ids", [])}
    print(f"Loading gold answer texts for {len(all_needed_gold_ids)} post IDs from posts.jsonl...")
    gold_posts_map = load_gold_posts_map(POSTS_JSONL_PATH, all_needed_gold_ids)

    print("Connecting to Qdrant, loading reranker, generator (Groq), and judge (local GGUF)...")
    db_manager = VectorDBManager()
    db_manager.collection  # one-time init
    reranker = ReRanker()
    llm_service = LLMService()
    judge = get_judge()

    from deepeval.metrics import ContextualRelevancyMetric, FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric
    from deepeval.test_case import LLMTestCase

    contextual_metric = ContextualRelevancyMetric(threshold=CONTEXTUAL_RELEVANCY_THRESHOLD, model=judge, include_reason=True)
    faithfulness_metric = FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=judge, include_reason=True)
    relevancy_metric = AnswerRelevancyMetric(threshold=ANSWER_RELEVANCY_THRESHOLD, model=judge, include_reason=True)
    contextual_recall_metric = ContextualRecallMetric(threshold=CONTEXTUAL_RECALL_THRESHOLD, model=judge, include_reason=True)

    rows = []
    for case in evaluable:
        query = query_for_case(case)
        chat_history = case.get("chat_history")

        # Step 1: rewrite (mirrors routes.py exactly)
        try:
            rewritten_query = llm_service.rewrite_query(query, chat_history)
        except Exception as e:
            print(f"  [{case['query_id']}] query rewrite failed: {e}")
            continue

        # Step 2: hybrid retrieval + rerank (mirrors routes.py exactly)
        try:
            fused_candidates = db_manager.search_hybrid(query=rewritten_query, n_results=CANDIDATE_K)
            reranked_results = reranker.rerank(query=rewritten_query, chunks=fused_candidates, top_k=TOP_K)
        except Exception as e:
            print(f"  [{case['query_id']}] retrieval failed: {e}")
            continue

        if not reranked_results:
            print(f"  [{case['query_id']}] SKIPPED: retrieval returned nothing")
            continue

        citations = build_citations(reranked_results)
        retrieved_answer_ids = {str(c["metadata"].get("answer_id")) for c in reranked_results}

        # Step 3: generation (original query, not the rewritten one -- matches routes.py)
        try:
            answer = llm_service.generate_answer(query, citations, chat_history=chat_history)
        except Exception as e:
            print(f"  [{case['query_id']}] generation failed: {e}")
            continue

        retrieval_context = [c["text_snippet"] for c in citations]
        gold_texts = [gold_posts_map.get(str(gid), "") for gid in case.get("gold_answer_ids", []) if gold_posts_map.get(str(gid))]
        expected_output = "\n\n".join(gold_texts) if gold_texts else None

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )

        row = {
            "query_id": case["query_id"],
            "category": case["category"],
            "query": query,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "gold_answer_ids": case["gold_answer_ids"],
            "expected_output_present": expected_output is not None,
            # Component-retriever-style signal for free: did the actual
            # gold answer make it into what got retrieved this run?
            "gold_answer_retrieved": bool(set(case["gold_answer_ids"]) & retrieved_answer_ids),
        }

        metric_list = [
            ("contextual_relevancy", contextual_metric, "contextual_relevancy_score"),
            ("faithfulness", faithfulness_metric, "faithfulness_score"),
            ("answer_relevancy", relevancy_metric, "answer_relevancy_score"),
        ]
        if expected_output:
            metric_list.append(("contextual_recall", contextual_recall_metric, "contextual_recall_score"))

        for name, metric, key in metric_list:
            try:
                metric.measure(test_case)
                row[key] = metric.score
                row[f"{name}_reason"] = metric.reason
            except Exception as e:
                row[key] = None
                row[f"{name}_error"] = str(e)

        rows.append(row)
        print(f"  [{case['query_id']}] gold_retrieved={row['gold_answer_retrieved']} "
              f"ctx_rel={row.get('contextual_relevancy_score')} "
              f"ctx_rec={row.get('contextual_recall_score')} "
              f"faith={row.get('faithfulness_score')} "
              f"ans_rel={row.get('answer_relevancy_score')}")

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
    gold_hit_rate = sum(1 for r in rows if r["gold_answer_retrieved"]) / len(rows)
    print(f"  n={len(rows)}  gold_answer_retrieved_rate={gold_hit_rate:.3f}")
    for key, label in [
        ("contextual_relevancy_score", "contextual_relevancy"),
        ("contextual_recall_score", "contextual_recall"),
        ("faithfulness_score", "faithfulness"),
        ("answer_relevancy_score", "answer_relevancy"),
    ]:
        avg = agg(rows, key)
        print(f"  {label}: {avg:.3f}" if avg is not None else f"  {label}: n/a")

    print("\n=== BY CATEGORY ===")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        subset = by_cat[cat]
        hit_rate = sum(1 for r in subset if r["gold_answer_retrieved"]) / len(subset)
        ctx = agg(subset, "contextual_relevancy_score")
        rec = agg(subset, "contextual_recall_score")
        faith = agg(subset, "faithfulness_score")
        rel = agg(subset, "answer_relevancy_score")
        fmt = lambda v: f"{v:.2f}" if v is not None else "n/a"
        print(f"  {cat:20s} n={len(subset):3d}  gold_hit={hit_rate:.2f}  "
              f"ctx_rel={fmt(ctx)}  ctx_rec={fmt(rec)}  faith={fmt(faith)}  ans_rel={fmt(rel)}")


def save_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"pipeline_eval_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved per-case results to {out_path.resolve()}")


if __name__ == "__main__":
    sys.exit(evaluate())
