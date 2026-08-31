"""
Standalone Contextual Recall Evaluation script using DeepEval's ContextualRecallMetric.

Evaluates how well the retrieved context (`retrieval_context`) covers the ground-truth
gold answer text (`expected_output`) extracted from posts.jsonl.

Run with:
  python evaluation/eval_contextual_recall.py --n 5
"""

import sys
import json
import time
import os
import argparse
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
from backend.core.llm_service import LLMService, build_search_query, decompose_query
from backend.core.config import SETTINGS
from evaluation.judge_model import get_judge

GOLDEN_JSON_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
POSTS_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "posts.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

CANDIDATE_K = 15
TOP_K = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Contextual Recall using DeepEval.")
    parser.add_argument("--n", type=int, default=0, help="Number of test cases to evaluate (0 = all evaluable cases).")
    parser.add_argument("--category", type=str, default=None, help="Filter test cases by category.")
    parser.add_argument("--local", action="store_true", help="Force local on-disk Qdrant index.")
    return parser.parse_args()


def load_cases(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gold_posts_map(posts_path: Path, needed_ids: set) -> dict:
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


def build_citations(reranked_results):
    citations = []
    for res in reranked_results:
        citations.append({
            "source_file": res["metadata"].get("question_title", "Untitled question"),
            "score": res["metadata"].get("score", 0),
            "is_accepted": res["metadata"].get("is_accepted", False),
            "url": res["metadata"].get("url", ""),
            "text_snippet": res["metadata"].get("display_text", res["text"]),
        })
    return citations


async def eval_case_async(case, db_manager, reranker, llm_service, judge, gold_map):
    from deepeval.metrics import ContextualRecallMetric
    from deepeval.test_case import LLMTestCase
    import asyncio

    query = case["query"]
    try:
        # Run retrieval and reranking in a worker thread so it doesn't block the async loop
        def retrieval_steps():
            rewritten = None
            search_query = query
            try:
                rewritten = llm_service.rewrite_query(query, case.get("chat_history"))
                search_query = rewritten
            except Exception:
                pass
            
            search_query = build_search_query(query, case.get("chat_history"), rewritten)
            decomposed = decompose_query(search_query, llm_service=llm_service)
            if len(decomposed) > 1:
                candidates = db_manager.search_multi_query(decomposed, n_results=CANDIDATE_K)
            else:
                candidates = db_manager.search_hybrid(query=search_query, n_results=CANDIDATE_K)
            reranked = reranker.rerank(query=query, chunks=candidates, top_k=TOP_K)
            return search_query, reranked

        search_query, reranked = await asyncio.to_thread(retrieval_steps)
    except Exception as e:
        print(f"  [{case['query_id']}] Error in retrieval: {e}")
        return None

    if not reranked:
        return None

    citations_3 = build_citations(reranked[:3])
    retrieved_context_3 = [c["text_snippet"] for c in citations_3]

    gold_texts = [gold_map.get(str(gid), "") for gid in case.get("gold_answer_ids", []) if gold_map.get(str(gid))]
    expected_output = "\n\n".join(gold_texts) if gold_texts else None

    if not expected_output:
        return None

    test_case = LLMTestCase(
        input=query,
        actual_output="[Evaluation of Retrieval Context Only]",
        expected_output=expected_output,
        retrieval_context=retrieved_context_3
    )

    # Use a new metric instance per case to avoid concurrent state conflicts
    recall_metric = ContextualRecallMetric(threshold=0.5, model=judge, include_reason=True)
    try:
        await recall_metric.a_measure(test_case)
        score = recall_metric.score
        reason = recall_metric.reason
    except Exception as e:
        score = None
        reason = str(e)

    retrieved_aids_5 = [str(c["metadata"].get("answer_id")) for c in reranked[:5]]
    retrieved_aids_3 = retrieved_aids_5[:3]
    retrieved_aids_1 = retrieved_aids_5[:1]

    gold_set = set(case["gold_answer_ids"])
    exact_gold_hit_1 = bool(gold_set & set(retrieved_aids_1))
    exact_gold_hit_3 = bool(gold_set & set(retrieved_aids_3))
    exact_gold_hit_5 = bool(gold_set & set(retrieved_aids_5))

    rr = 0.0
    for idx, c in enumerate(reranked[:5], start=1):
        if str(c["metadata"].get("answer_id")) in gold_set:
            rr = 1.0 / idx
            break

    retrieved_in_top_1 = sum(1 for c in reranked[:1] if str(c["metadata"].get("answer_id")) in gold_set)
    retrieved_in_top_3 = sum(1 for c in reranked[:3] if str(c["metadata"].get("answer_id")) in gold_set)
    retrieved_in_top_5 = sum(1 for c in reranked[:5] if str(c["metadata"].get("answer_id")) in gold_set)

    precision_1 = retrieved_in_top_1 / 1.0
    precision_3 = retrieved_in_top_3 / 3.0
    precision_5 = retrieved_in_top_5 / 5.0

    print(f"\n[{case['query_id']}] Category: {case['category']}")
    print(f"  Query: {query}")
    print(f"  Search Query: {search_query.encode('ascii', 'replace').decode('ascii')}")
    print(f"  Retrieved IDs: {retrieved_aids_5}")
    print(f"  Gold IDs: {list(gold_set)}")
    print(f"  Hits: Recall@1: {exact_gold_hit_1} | Recall@3: {exact_gold_hit_3} | Recall@5: {exact_gold_hit_5}")
    print(f"  Precision: P@1: {precision_1:.3f} | P@3: {precision_3:.3f} | P@5: {precision_5:.3f}")
    print(f"  MRR: {rr:.3f}")
    print(f"  Contextual Recall Score: {score}")
    print(f"  Reason: {reason}")

    return {
        "query_id": case["query_id"],
        "category": case["category"],
        "query": query,
        "exact_gold_hit_1": exact_gold_hit_1,
        "exact_gold_hit_3": exact_gold_hit_3,
        "exact_gold_hit_5": exact_gold_hit_5,
        "precision_1": precision_1,
        "precision_3": precision_3,
        "precision_5": precision_5,
        "mrr": rr,
        "contextual_recall_score": score,
        "reason": reason
    }


def run_eval():
    args = parse_args()

    cases = load_cases(GOLDEN_JSON_PATH)
    if not cases:
        print("Golden dataset empty or not found.")
        return 1

    evaluable = [c for c in cases if c.get("gold_answer_ids") and c.get("query")]
    if args.category:
        evaluable = [c for c in evaluable if c.get("category") == args.category]

    if args.n and args.n > 0:
        evaluable = evaluable[:args.n]

    print(f"Evaluating {len(evaluable)} cases for Contextual Recall...")

    needed_ids = {str(gid) for c in evaluable for gid in c.get("gold_answer_ids", [])}
    print(f"Loading gold answer text for {len(needed_ids)} post IDs...")
    gold_map = load_gold_posts_map(POSTS_JSONL_PATH, needed_ids)

    db_manager = VectorDBManager(force_local=args.local)
    db_manager.collection
    reranker = ReRanker()
    llm_service = LLMService()
    judge = get_judge()

    import asyncio

    async def main_async():
        # Limit concurrency to 3 to protect Groq API rate limits
        sem = asyncio.Semaphore(3)

        async def sem_eval_case(case):
            async with sem:
                for attempt in range(3):
                    try:
                        res = await eval_case_async(case, db_manager, reranker, llm_service, judge, gold_map)
                        await asyncio.sleep(0.2)
                        return res
                    except Exception as err:
                        if attempt == 2:
                            print(f"[{case.get('query_id')}] Failed after 3 attempts: {err}")
                            return None
                        await asyncio.sleep(2 ** (attempt + 1))

        tasks = [sem_eval_case(c) for c in evaluable]
        
        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating cases")
        return [r for r in results if r is not None]

    rows = asyncio.run(main_async())

    if rows:
        # Category breakdown
        cat_stats = defaultdict(lambda: {
            "count": 0,
            "hits_1": 0,
            "hits_3": 0,
            "hits_5": 0,
            "precision_1_sum": 0.0,
            "precision_3_sum": 0.0,
            "precision_5_sum": 0.0,
            "mrr_sum": 0.0,
            "valid_recall_scores": [],
        })

        for r in rows:
            cat = r["category"]
            cat_stats[cat]["count"] += 1
            if r["exact_gold_hit_1"]:
                cat_stats[cat]["hits_1"] += 1
            if r["exact_gold_hit_3"]:
                cat_stats[cat]["hits_3"] += 1
            if r["exact_gold_hit_5"]:
                cat_stats[cat]["hits_5"] += 1
            cat_stats[cat]["precision_1_sum"] += r["precision_1"]
            cat_stats[cat]["precision_3_sum"] += r["precision_3"]
            cat_stats[cat]["precision_5_sum"] += r["precision_5"]
            cat_stats[cat]["mrr_sum"] += r["mrr"]
            if r["contextual_recall_score"] is not None:
                cat_stats[cat]["valid_recall_scores"].append(r["contextual_recall_score"])

        # Global stats
        total_cases = len(rows)
        global_hit_1 = sum(1 for r in rows if r["exact_gold_hit_1"]) / total_cases
        global_hit_3 = sum(1 for r in rows if r["exact_gold_hit_3"]) / total_cases
        global_hit_5 = sum(1 for r in rows if r["exact_gold_hit_5"]) / total_cases
        global_prec_1 = sum(r["precision_1"] for r in rows) / total_cases
        global_prec_3 = sum(r["precision_3"] for r in rows) / total_cases
        global_prec_5 = sum(r["precision_5"] for r in rows) / total_cases
        global_mrr = sum(r["mrr"] for r in rows) / total_cases
        global_valid_scores = [r["contextual_recall_score"] for r in rows if r["contextual_recall_score"] is not None]
        global_avg_recall = sum(global_valid_scores) / len(global_valid_scores) if global_valid_scores else 0

        print("\n================ GLOBAL SUMMARY ================")
        print(f"Total Evaluated Cases: {total_cases}")
        print(f"Recall:              R@1: {global_hit_1:.3f} | R@3: {global_hit_3:.3f} | R@5: {global_hit_5:.3f}")
        print(f"Precision:           P@1: {global_prec_1:.3f} | P@3: {global_prec_3:.3f} | P@5: {global_prec_5:.3f}")
        print(f"MRR:                 {global_mrr:.3f}")
        print(f"Avg Context Recall:  {global_avg_recall:.3f} (based on {len(global_valid_scores)} valid cases)")
        print("================================================\n")

        print("================ CATEGORY SUMMARY ================")
        print(f"{'Category':<20} | {'Count':<5} | {'R@1':<6} | {'R@3':<6} | {'R@5':<6} | {'P@1':<6} | {'P@3':<6} | {'P@5':<6} | {'MRR':<6} | {'Avg Context Rec':<15}")
        print("-" * 110)
        for cat, stats in sorted(cat_stats.items()):
            count = stats["count"]
            rec_1 = stats["hits_1"] / count
            rec_3 = stats["hits_3"] / count
            rec_5 = stats["hits_5"] / count
            prec_1 = stats["precision_1_sum"] / count
            prec_3 = stats["precision_3_sum"] / count
            prec_5 = stats["precision_5_sum"] / count
            mrr = stats["mrr_sum"] / count
        print("==================================================")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_filename = RESULTS_DIR / f"contextual_recall_eval_{timestamp}.json"
        with open(out_filename, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "total_cases": total_cases,
                "global_avg_contextual_recall": global_avg_recall,
                "global_mrr": global_mrr,
                "global_recall_at_1": global_hit_1,
                "global_recall_at_3": global_hit_3,
                "global_recall_at_5": global_hit_5,
                "category_breakdown": {
                    cat: {
                        "count": st["count"],
                        "recall_at_1": st["hits_1"] / st["count"],
                        "recall_at_3": st["hits_3"] / st["count"],
                        "recall_at_5": st["hits_5"] / st["count"],
                        "mrr": st["mrr_sum"] / st["count"],
                        "avg_contextual_recall": (sum(st["valid_recall_scores"]) / len(st["valid_recall_scores"])) if st["valid_recall_scores"] else 0.0
                    }
                    for cat, st in cat_stats.items()
                },
                "rows": rows
            }, f, indent=2)
        print(f"\nSaved full Contextual Recall report to: {out_filename}")

    return 0


if __name__ == "__main__":
    sys.exit(run_eval())
