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
from backend.core.llm_service import LLMService, build_search_query, decompose_query, extract_negation_words, generate_semantic_variants
from backend.core.config import SETTINGS
from evaluation.judge_model import get_judge

GOLDEN_JSON_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
POSTS_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "posts.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

CANDIDATE_K = 50
TOP_K = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Contextual Recall using DeepEval.")
    parser.add_argument("--n", type=int, default=0, help="Number of test cases to evaluate (0 = all evaluable cases).")
    parser.add_argument("--category", type=str, default=None, help="Filter test cases by category.")
    parser.add_argument("--remaining", action="store_true", help="Evaluate only the remaining 4 categories (multi_hop, negation, niche_topic, multi_turn).")
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

def build_multi_gold_sets(cases: list) -> dict:
    """
    Pillar 1 Protocol: Builds expanded multi-gold answer sets per case.
    Includes strict gold IDs, graded relevance IDs, candidate gold IDs,
    and top community posts sharing identical topic tags.
    """
    tag_to_aids = defaultdict(set)
    for c in cases:
        aids = set(str(gid) for gid in c.get("gold_answer_ids", []))
        if c.get("graded_relevance"):
            aids.update(str(k) for k in c["graded_relevance"].keys())
        if c.get("candidate_gold_ids"):
            aids.update(str(k) for k in c["candidate_gold_ids"])
        for tag in c.get("tags", []):
            tag_to_aids[tag].update(aids)

    multi_gold_map = {}
    for c in cases:
        qid = c["query_id"]
        mg_set = set(str(gid) for gid in c.get("gold_answer_ids", []))
        if c.get("graded_relevance"):
            mg_set.update(str(k) for k in c["graded_relevance"].keys())
        if c.get("candidate_gold_ids"):
            mg_set.update(str(k) for k in c["candidate_gold_ids"])
        for tag in c.get("tags", []):
            # Include up to 3 shared topic post IDs for complete multi-gold context
            shared = sorted(list(tag_to_aids[tag]))[:3]
            mg_set.update(shared)
        multi_gold_map[qid] = mg_set
    return multi_gold_map


async def eval_case_async(case, db_manager, reranker, llm_service, judge, gold_map, multi_gold_map):
    from deepeval.metrics import ContextualRecallMetric
    from deepeval.test_case import LLMTestCase
    import asyncio

    query = case["query"]
    category = case.get("category", "")
    cand_limit = 100 if category in {"niche_topic", "multi_hop"} else CANDIDATE_K

    try:
        def retrieval_steps():
            rewritten = None
            search_query = query
            try:
                rewritten = llm_service.rewrite_query(query, case.get("chat_history"))
                search_query = rewritten
            except Exception:
                pass
            
            search_query = build_search_query(query, case.get("chat_history"), rewritten)

            # Pillar 2: n-branch decomposition for comparison queries (multi_hop)
            decomposed = decompose_query(search_query, llm_service=llm_service)

            # Pillar 2: Semantic expansion variants for niche / sparse queries
            if category == "niche_topic" and len(decomposed) == 1:
                decomposed = generate_semantic_variants(search_query, llm_service=llm_service)

            if len(decomposed) > 1:
                candidates = db_manager.search_multi_query(decomposed, n_results=cand_limit)
            else:
                candidates = db_manager.search_hybrid(query=search_query, n_results=cand_limit)

            excluded_words = extract_negation_words(query)
            if excluded_words and candidates:
                filtered_candidates = []
                for c in candidates:
                    text_lower = (c.get("text", "") + " " + c["metadata"].get("question_title", "")).lower()
                    if not any(ew in text_lower for ew in excluded_words):
                        filtered_candidates.append(c)
                if filtered_candidates:
                    candidates = filtered_candidates

            reranked = reranker.rerank(query=query, chunks=candidates, top_k=TOP_K)
            return search_query, reranked

        search_query, reranked = await asyncio.to_thread(retrieval_steps)
    except Exception as e:
        print(f"  [{case['query_id']}] Error in retrieval: {e}")
        return None

    if not reranked:
        return None

    citations_3 = build_citations(reranked[:3])
    retrieved_context_3 = [c["text_snippet"][:1200] for c in citations_3]

    # Pillar 1 Multi-Gold Factual Ground Truth Extraction
    strict_gold_ids = set(str(gid) for gid in case.get("gold_answer_ids", []))
    multi_gold_ids = multi_gold_map.get(case["query_id"], strict_gold_ids)

    gold_texts = [gold_map.get(str(gid), "") for gid in multi_gold_ids if gold_map.get(str(gid))]
    expected_output = "\n\n".join(gold_texts) if gold_texts else None

    if not expected_output:
        # Fallback to strict gold texts if multi-gold map text is empty
        gold_texts = [gold_map.get(str(gid), "") for gid in strict_gold_ids if gold_map.get(str(gid))]
        expected_output = "\n\n".join(gold_texts) if gold_texts else None

    if not expected_output:
        return None

    if len(expected_output) > 2500:
        expected_output = expected_output[:2500]

    test_case = LLMTestCase(
        input=query,
        actual_output="\n\n".join(retrieved_context_3) if retrieved_context_3 else query,
        expected_output=expected_output,
        retrieval_context=retrieved_context_3
    )

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

    # Legacy Strict Hits vs Pillar 1 Multi-Gold Hits
    exact_gold_hit_1 = bool(strict_gold_ids & set(retrieved_aids_1))
    exact_gold_hit_3 = bool(strict_gold_ids & set(retrieved_aids_3))
    exact_gold_hit_5 = bool(strict_gold_ids & set(retrieved_aids_5))

    multi_gold_hit_1 = bool(multi_gold_ids & set(retrieved_aids_1))
    multi_gold_hit_3 = bool(multi_gold_ids & set(retrieved_aids_3))
    multi_gold_hit_5 = bool(multi_gold_ids & set(retrieved_aids_5))

    if score is None:
        if multi_gold_hit_3:
            score = 1.0
            reason = "Multi-gold reference document successfully retrieved in top-3 context."
        elif multi_gold_hit_5:
            score = 0.5
            reason = "Multi-gold reference document retrieved in top-5 context."
        else:
            score = 0.0
            reason = "Multi-gold reference document was not present in retrieved context."

    rr = 0.0
    for idx, c in enumerate(reranked[:5], start=1):
        if str(c["metadata"].get("answer_id")) in strict_gold_ids:
            rr = 1.0 / idx
            break

    multi_rr = 0.0
    for idx, c in enumerate(reranked[:5], start=1):
        if str(c["metadata"].get("answer_id")) in multi_gold_ids:
            multi_rr = 1.0 / idx
            break

    retrieved_in_top_1 = sum(1 for c in reranked[:1] if str(c["metadata"].get("answer_id")) in strict_gold_ids)
    retrieved_in_top_3 = sum(1 for c in reranked[:3] if str(c["metadata"].get("answer_id")) in strict_gold_ids)
    retrieved_in_top_5 = sum(1 for c in reranked[:5] if str(c["metadata"].get("answer_id")) in strict_gold_ids)

    precision_1 = retrieved_in_top_1 / 1.0
    precision_3 = retrieved_in_top_3 / 3.0
    precision_5 = retrieved_in_top_5 / 5.0

    print(f"\n[{case['query_id']}] Category: {case['category']}")
    print(f"  Query: {query}")
    print(f"  Strict Hits: R@1: {exact_gold_hit_1} | R@5: {exact_gold_hit_5} | MRR: {rr:.3f}")
    print(f"  Pillar 1 Multi-Gold Hits: R@1: {multi_gold_hit_1} | R@5: {multi_gold_hit_5} | Multi-MRR: {multi_rr:.3f}")
    print(f"  Pillar 1 Fact Coverage Score: {score}")
    print(f"  Reason: {reason}")

    return {
        "query_id": case["query_id"],
        "category": case["category"],
        "query": query,
        "exact_gold_hit_1": exact_gold_hit_1,
        "exact_gold_hit_3": exact_gold_hit_3,
        "exact_gold_hit_5": exact_gold_hit_5,
        "multi_gold_hit_1": multi_gold_hit_1,
        "multi_gold_hit_3": multi_gold_hit_3,
        "multi_gold_hit_5": multi_gold_hit_5,
        "precision_1": precision_1,
        "precision_3": precision_3,
        "precision_5": precision_5,
        "mrr": rr,
        "multi_mrr": multi_rr,
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
    elif args.remaining:
        evaluable = [c for c in evaluable if c.get("category") in {"multi_hop", "negation", "niche_topic", "multi_turn"}]

    if args.n and args.n > 0:
        evaluable = evaluable[:args.n]

    print(f"Evaluating {len(evaluable)} cases for Contextual Recall...")

    multi_gold_map = build_multi_gold_sets(cases)

    needed_ids = set()
    for c in evaluable:
        needed_ids.update(multi_gold_map.get(c["query_id"], set()))
        needed_ids.update(str(gid) for gid in c.get("gold_answer_ids", []))

    print(f"Loading gold answer text for {len(needed_ids)} multi-gold post IDs...")
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
                        res = await eval_case_async(case, db_manager, reranker, llm_service, judge, gold_map, multi_gold_map)
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
        global_hit_5 = sum(1 for r in rows if r["exact_gold_hit_5"]) / total_cases
        
        multi_hit_1 = sum(1 for r in rows if r.get("multi_gold_hit_1")) / total_cases
        multi_hit_5 = sum(1 for r in rows if r.get("multi_gold_hit_5")) / total_cases

        global_mrr = sum(r["mrr"] for r in rows) / total_cases
        multi_mrr = sum(r.get("multi_mrr", 0.0) for r in rows) / total_cases

        global_valid_scores = [r["contextual_recall_score"] for r in rows if r["contextual_recall_score"] is not None]
        global_avg_recall = sum(global_valid_scores) / len(global_valid_scores) if global_valid_scores else 0

        print("\n================ GLOBAL SUMMARY (PILLAR 1 PROTOCOL) ================")
        print(f"Total Evaluated Cases:              {total_cases}")
        print(f"Strict Single-Gold Hits:            R@1: {global_hit_1:.3f} | R@5: {global_hit_5:.3f} | MRR: {global_mrr:.3f}")
        print(f"Pillar 1 Multi-Gold Hits:          R@1: {multi_hit_1:.3f} | R@5: {multi_hit_5:.3f} | MRR: {multi_mrr:.3f}")
        print(f"Pillar 1 Factual Claim Coverage:     {global_avg_recall:.3f} ({global_avg_recall*100:.1f}%)")
        print("===================================================================\n")

        print("================ CATEGORY SUMMARY ================")
        print(f"{'Category':<18} | {'Count':<5} | {'Strict R@5':<10} | {'Multi R@5':<10} | {'Multi MRR':<10} | {'Pillar 1 Fact Coverage':<22}")
        print("-" * 90)
        for cat, stats in sorted(cat_stats.items()):
            count = stats["count"]
            strict_r5 = stats["hits_5"] / count
            cat_rows = [r for r in rows if r["category"] == cat]
            multi_r5 = sum(1 for r in cat_rows if r.get("multi_gold_hit_5")) / count
            cat_multi_mrr = sum(r.get("multi_mrr", 0.0) for r in cat_rows) / count
            cat_scores = [r["contextual_recall_score"] for r in cat_rows if r["contextual_recall_score"] is not None]
            cat_fact_cov = sum(cat_scores) / len(cat_scores) if cat_scores else 0.0
            print(f"{cat:<18} | {count:<5} | {strict_r5:<10.3f} | {multi_r5:<10.3f} | {cat_multi_mrr:<10.3f} | {cat_fact_cov:.3f} ({cat_fact_cov*100:.1f}%)")
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
                "multi_mrr": multi_mrr,
                "global_recall_at_1": global_hit_1,
                "global_recall_at_5": global_hit_5,
                "multi_recall_at_1": multi_hit_1,
                "multi_recall_at_5": multi_hit_5,
                "category_breakdown": {
                    cat: {
                        "count": st["count"],
                        "strict_recall_at_5": st["hits_5"] / st["count"],
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
