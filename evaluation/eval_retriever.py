"""
Component-level retriever evaluation (see plan.md Part E, layer 1).

Tests the retriever in isolation: query -> [optional rewrite] -> retrieval
-> [optional rerank] -> ranked list of answer_ids. Reference-based,
programmatic -- computes recall@k, precision@k, and MRR against the
gold_answer_ids already sitting in evaluation/golden_dataset.json.

Only cases that carry gold_answer_ids are evaluated (adversarial and
out_of_scope cases have no gold answer by design and are skipped -- they're
refusal tests, not retrieval tests).

Requires the corpus to already be embedded and uploaded to Qdrant
(embed_corpus.py -> upload_embeddings.py). Will not produce meaningful
results against an empty/partial collection.

Ablation flags (no re-embedding needed for any of these -- see judge_model.py
/ README for why embedding-model/chunking ablations are out of scope):
  --method dense|sparse|hybrid   which retrieval method to test (default hybrid)
  --no-rerank                    skip the cross-encoder rerank step
  --rewrite                      rewrite the query with Gemini before retrieval
                                  (default off -- tests the raw query, matching
                                  the "retriever in isolation" design)
  --pool N                       candidate pool size fetched before reranking
                                  (default 5x the largest k, mirrors production)
  --reranker-model NAME           swap the cross-encoder model
  --ks "3,5,10"                  which k values to report recall/precision@k for

Run with: python evaluation/eval_retriever.py [flags]
Each run is tagged with its config in both the printed summary and the
saved results filename, so multiple ablation runs don't overwrite each
other and are easy to line up side by side.
"""

import sys
import json
import time
import os
import argparse
from pathlib import Path
from collections import defaultdict

# Prevent PyTorch / OpenMP CPU execution deadlocks on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.vector_store import VectorDBManager
from backend.core.reranker import ReRanker
from backend.core.llm_service import build_search_query

GOLDEN_JSON_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
POSTS_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "posts.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

DEFAULT_TOP_KS = [3, 5, 10]
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Retriever ablation eval.")
    parser.add_argument("--method", choices=["dense", "sparse", "hybrid"], default="hybrid",
                         help="Retrieval method to test (default: hybrid).")
    parser.add_argument("--no-rerank", action="store_true",
                         help="Skip the cross-encoder rerank step; rank by raw retrieval order.")
    parser.add_argument("--rewrite", action="store_true",
                         help="Rewrite the query with Gemini before retrieval (default: off, raw query).")
    parser.add_argument("--pool", type=int, default=None,
                         help="Candidate pool size fetched before reranking (default: 5x max k).")
    parser.add_argument("--reranker-model", type=str, default=DEFAULT_RERANKER_MODEL,
                         help=f"Cross-encoder model name (default: {DEFAULT_RERANKER_MODEL}).")
    parser.add_argument("--ks", type=str, default="3,5,10",
                         help="Comma-separated k values to report recall/precision@k for (default: 3,5,10).")
    parser.add_argument("--local", action="store_true",
                         help="Force the local on-disk Qdrant index instead of Qdrant Cloud, regardless "
                              "of what's in .env. Use this instead of blanking QDRANT_URL/QDRANT_API_KEY "
                              "in the shell -- that trick doesn't reliably work on Windows (PowerShell's "
                              "$env:VAR=\"\" deletes the variable rather than blanking it, so .env silently "
                              "refills it). Requires evaluation/load_local_qdrant.py to have been run first.")
    parser.add_argument("--category", type=str, default=None,
                         help="Filter evaluation cases to a specific category (e.g. multi_turn).")
    return parser.parse_args()


def config_tag(args, top_ks, pool) -> str:
    """Short identifier for this run's config, used in filenames/summary
    headers so ablation runs are distinguishable at a glance."""
    parts = [args.method]
    parts.append("norerank" if args.no_rerank else "rerank")
    parts.append("rewrite" if args.rewrite else "raw")
    parts.append(f"pool{pool}")
    if not args.no_rerank and args.reranker_model != DEFAULT_RERANKER_MODEL:
        parts.append(args.reranker_model.split("/")[-1])
    return "_".join(parts)


def load_cases(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        print(f"Error: {path.resolve()} is missing or empty. "
              f"Run backend/scripts/build_golden_dataset.py first.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gold_question_ids(cases) -> dict:
    """Maps every gold answer_id in the eval set to its question_id, by
    streaming posts.jsonl once.

    Needed for question-level recall (see evaluate()). A manual audit of 32
    cases found that ~70% of apparent retrieval "misses" were the retriever
    returning a *different answer to the same question*, or a different
    thread on the same topic -- the single hand-picked gold answer_id in
    golden_dataset.json understates real retrieval quality. Answer-level
    recall stays the strict headline metric; this adds the thread-level one
    beside it rather than replacing it.
    """
    needed = {int(a) for c in cases for a in c.get("gold_answer_ids", [])}
    lookup, remaining = {}, set(needed)
    if not remaining or not POSTS_JSONL_PATH.exists():
        if not POSTS_JSONL_PATH.exists():
            print(f"Warning: {POSTS_JSONL_PATH.name} not found -- question-level "
                  f"recall will be reported as 0. Answer-level metrics are unaffected.")
        return lookup
    with open(POSTS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            post = json.loads(line)
            aid = post["answer_id"]
            if aid in remaining:
                lookup[str(aid)] = str(post["question_id"])
                remaining.discard(aid)
    if remaining:
        print(f"Warning: {len(remaining)} gold answer_ids not found in "
              f"{POSTS_JSONL_PATH.name}; their question-level recall will read 0.")
    return lookup


def queries_for_case(case: dict):
    """Returns a list of (query_text, sub_label) pairs to evaluate for this
    case. Paraphrase groups test every variant separately (so we can see if
    retrieval is consistent across rewordings of the same intent); everything
    else contributes a single query."""
    if case["category"] == "paraphrase_group":
        return [(v, f"variant_{i}") for i, v in enumerate(case.get("variants", []))]
    if "query" in case:
        return [(case["query"], None)]
    return []


def retrieve(db_manager, method: str, query: str, pool: int):
    if method == "dense":
        return db_manager.search(query, n_results=pool)
    if method == "sparse":
        return db_manager.search_sparse(query, n_results=pool)
    return db_manager.search_hybrid(query, n_results=pool)


def ranked_answer_ids(db_manager, reranker, method: str, query: str, pool: int, use_rerank: bool):
    """Runs the retrieval path (method -> optional cross-encoder rerank) and
    returns a de-duplicated, rank-ordered list of answer_ids as strings.

    When reranking, reranks the FULL candidate pool, not just the eventual
    top_k -- chunks are per-answer-fragments, so several of the top chunks
    can belong to the same answer. Capping the rerank step at top_k before
    deduplicating would let chunk-level duplicates burn through the top
    slots and unfairly shrink recall@k. Reranking everything first and
    deduplicating after avoids that."""
    hits = retrieve(db_manager, method, query, pool)
    if not hits:
        return []

    if use_rerank:
        ordered = reranker.rerank(query, hits, top_k=len(hits))
    else:
        ordered = hits  # raw retrieval order (already ranked by the method's own scoring)

    ranked_ids, ranked_qids = [], []
    seen = set()
    for chunk in ordered:
        meta = chunk["metadata"]
        aid = str(meta.get("answer_id"))
        if aid and aid not in seen:
            seen.add(aid)
            ranked_ids.append(aid)
            # Parallel list, same rank order: ranked_qids[i] is the thread
            # ranked_ids[i] belongs to. Used for question-level recall.
            ranked_qids.append(str(meta.get("question_id")))
    return ranked_ids, ranked_qids


def reciprocal_rank(ranked_ids, gold_ids: set) -> float:
    for rank, aid in enumerate(ranked_ids, start=1):
        if aid in gold_ids:
            return 1.0 / rank
    return 0.0


def evaluate():
    args = parse_args()
    top_ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    pool = args.pool or max(top_ks) * 5
    tag = config_tag(args, top_ks, pool)

    cases = load_cases(GOLDEN_JSON_PATH)
    if cases is None:
        return 1

    evaluable = [c for c in cases if c.get("gold_answer_ids")]
    if args.category:
        evaluable = [c for c in evaluable if c.get("category") == args.category]
    skipped = len(cases) - len(evaluable)
    print(f"Config: method={args.method} rerank={not args.no_rerank} "
          f"rewrite={args.rewrite} pool={pool} ks={top_ks}"
          + (f" reranker_model={args.reranker_model}" if not args.no_rerank else ""))
    print(f"Loaded {len(cases)} cases ({len(evaluable)} evaluable, "
          f"{skipped} skipped -- no gold_answer_ids, e.g. adversarial/out_of_scope).")

    print("Mapping gold answer_ids to their question_ids (for question-level recall)...")
    gold_qid_lookup = gold_question_ids(evaluable)

    print(f"Connecting to Qdrant ({'local' if args.local else 'cloud/.env'})...")
    db_manager = VectorDBManager(force_local=args.local)
    db_manager.collection  # one-time init

    reranker = None
    if not args.no_rerank:
        print(f"Loading reranker ({args.reranker_model})...")
        reranker = ReRanker(model_name=args.reranker_model)

    llm_service = None
    if args.rewrite:
        print("Loading LLMService for query rewriting (Local/HF/GitHub/OpenRouter/Gemini)...")
        from backend.core.llm_service import LLMService
        llm_service = LLMService()

    # Per-query-instance results, tagged with category for breakdown.
    rows = []

    # Flatten cases into single query instances for accurate progress tracking
    query_instances = []
    for case in evaluable:
        for query_text, sub_label in queries_for_case(case):
            query_instances.append((case, query_text, sub_label))

    try:
        from tqdm import tqdm
        pbar = tqdm(query_instances, desc="Evaluating retriever queries")
    except ImportError:
        print("tqdm not installed. Running without progress bar...")
        pbar = query_instances

    for case, query_text, sub_label in pbar:
        gold_ids = set(case["gold_answer_ids"])
        negative_ids = set(case.get("negative_answer_ids", []))
        gold_qids = {gold_qid_lookup[a] for a in gold_ids if a in gold_qid_lookup}
        search_query = query_text
        rewritten_query = None
        if llm_service is not None:
            try:
                # chat_history matters: multi_turn queries are follow-ups whose
                # topic lives entirely in the pronoun ("prevent it", "test for
                # it"). Without the history the rewriter is asked to resolve a
                # referent it was never shown, so those cases could never pass.
                rewritten_query = llm_service.rewrite_query(query_text, case.get("chat_history"))
                search_query = rewritten_query
            except Exception as e:
                print(f"  [{case['query_id']}] rewrite failed ({e}); using raw query")
        
        # Apply standalone search query construction (concat fallback if rewriter did nothing)
        search_query = build_search_query(query_text, case.get("chat_history"), rewritten_query)

        ranked, ranked_qids = ranked_answer_ids(
            db_manager, reranker, args.method, search_query, pool, not args.no_rerank
        )
        row = {
            "query_id": case["query_id"],
            "sub_label": sub_label,
            "category": case["category"],
            "query": query_text,
            "search_query": search_query,
            "mrr": reciprocal_rank(ranked, gold_ids),
            "hit_at": {},
            "hit_q_at": {},
            "precision_at": {},
            # Saved so follow-up analysis (which posts actually came back, and
            # were they reasonable?) does not require re-running the whole eval.
            "ranked_top10": ranked[:10],
            "ranked_top10_qids": ranked_qids[:10],
        }
        for k in top_ks:
            top_k_ids = set(ranked[:k])
            row["hit_at"][k] = bool(gold_ids & top_k_ids)
            row["hit_q_at"][k] = bool(gold_qids & set(ranked_qids[:k]))
            row["precision_at"][k] = len(gold_ids & top_k_ids) / k
        if negative_ids:
            row["distractor_in_top_k"] = {
                k: bool(negative_ids & set(ranked[:k])) for k in top_ks
            }
        rows.append(row)

    if not rows:
        print("No evaluable query instances produced results -- nothing to report.")
        return 1

    print_summary(rows, top_ks, tag)
    save_results(rows, tag)
    return 0


def print_summary(rows, top_ks, tag):
    def agg(subset):
        n = len(subset)
        out = {"n": n, "mrr": sum(r["mrr"] for r in subset) / n}
        for k in top_ks:
            out[f"recall@{k}"] = sum(1 for r in subset if r["hit_at"][k]) / n
            out[f"qrecall@{k}"] = sum(1 for r in subset if r.get("hit_q_at", {}).get(k)) / n
            out[f"precision@{k}"] = sum(r["precision_at"][k] for r in subset) / n
        return out

    print(f"\n=== OVERALL [{tag}] ===")
    overall = agg(rows)
    print(f"  n={overall['n']}  MRR={overall['mrr']:.3f}")
    for k in top_ks:
        print(f"  recall@{k}={overall[f'recall@{k}']:.3f}  "
              f"qrecall@{k}={overall[f'qrecall@{k}']:.3f}  "
              f"precision@{k}={overall[f'precision@{k}']:.3f}")
    print("  (qrecall = gold answer's THREAD retrieved, vs recall = that exact answer)")

    print("\n=== BY CATEGORY ===")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        stats = agg(by_cat[cat])
        recall_str = "  ".join(
            f"r@{k}={stats[f'recall@{k}']:.2f}/q{stats[f'qrecall@{k}']:.2f}" for k in top_ks
        )
        print(f"  {cat:20s} n={stats['n']:3d}  MRR={stats['mrr']:.3f}  {recall_str}")

    negation_rows = [r for r in rows if "distractor_in_top_k" in r]
    if negation_rows:
        print("\n=== NEGATION DISTRACTOR CHECK (diagnostic, not a headline metric) ===")
        print("  Fraction of negation queries where the excluded-topic post still shows up:")
        for k in top_ks:
            rate = sum(1 for r in negation_rows if r["distractor_in_top_k"][k]) / len(negation_rows)
            print(f"    distractor present in top_{k}: {rate:.2f} ({len(negation_rows)} cases)")


def save_results(rows, tag):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"retriever_eval_{tag}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved per-query results to {out_path.resolve()}")
    print("(Filename is tagged with this run's config -- compare across ablation runs by "
          "diffing the recall@k/MRR summaries printed above, or load multiple result files "
          "and aggregate them yourself for a README table.)")


if __name__ == "__main__":
    sys.exit(evaluate())
