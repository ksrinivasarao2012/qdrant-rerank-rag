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
from backend.core.llm_service import build_search_query

GOLDEN_JSON_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
POSTS_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "posts.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

CANDIDATE_K = 15
TOP_K = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Contextual Recall using DeepEval.")
    parser.add_argument("--n", type=int, default=10, help="Number of test cases to evaluate (default: 10).")
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


def run_eval():
    args = parse_args()

    cases = load_cases(GOLDEN_JSON_PATH)
    if not cases:
        print("Golden dataset empty or not found.")
        return 1

    evaluable = [c for c in cases if c.get("gold_answer_ids") and c.get("query")]
    if args.category:
        evaluable = [c for c in evaluable if c.get("category") == args.category]

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

    from deepeval.metrics import ContextualRecallMetric
    from deepeval.test_case import LLMTestCase

    recall_metric = ContextualRecallMetric(threshold=0.5, model=judge, include_reason=True)

    rows = []
    for case in evaluable:
        query = case["query"]
        try:
            rewritten = None
            search_query = query
            try:
                rewritten = llm_service.rewrite_query(query, case.get("chat_history"))
                search_query = rewritten
            except Exception as re:
                print(f"  [{case['query_id']}] LLM Rewrite failed ({re}); using raw query")
            
            search_query = build_search_query(query, case.get("chat_history"), rewritten)
            candidates = db_manager.search_hybrid(query=search_query, n_results=CANDIDATE_K)
            reranked = reranker.rerank(query=search_query, chunks=candidates, top_k=TOP_K)
        except Exception as e:
            print(f"  [{case['query_id']}] Error in retrieval: {e}")
            continue

        if not reranked:
            continue

        citations = build_citations(reranked)
        retrieved_context = [c["text_snippet"] for c in citations]

        gold_texts = [gold_map.get(str(gid), "") for gid in case.get("gold_answer_ids", []) if gold_map.get(str(gid))]
        expected_output = "\n\n".join(gold_texts) if gold_texts else None

        if not expected_output:
            print(f"  [{case['query_id']}] Skip: No gold answer text found.")
            continue

        test_case = LLMTestCase(
            input=query,
            actual_output="[Evaluation of Retrieval Context Only]",
            expected_output=expected_output,
            retrieval_context=retrieved_context
        )

        try:
            recall_metric.measure(test_case)
            score = recall_metric.score
            reason = recall_metric.reason
        except Exception as e:
            score = None
            reason = str(e)

        retrieved_aids = {str(c["metadata"].get("answer_id")) for c in reranked}
        exact_gold_hit = bool(set(case["gold_answer_ids"]) & retrieved_aids)

        row = {
            "query_id": case["query_id"],
            "category": case["category"],
            "query": query,
            "exact_gold_hit": exact_gold_hit,
            "contextual_recall_score": score,
            "reason": reason
        }
        rows.append(row)

        print(f"\n[{case['query_id']}] Category: {case['category']}")
        print(f"  Query: {query}")
        print(f"  Exact Gold Hit (Recall@3): {exact_gold_hit}")
        print(f"  Contextual Recall Score: {score}")
        print(f"  Reason: {reason}")

    if rows:
        valid_scores = [r["contextual_recall_score"] for r in rows if r["contextual_recall_score"] is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        hit_rate = sum(1 for r in rows if r["exact_gold_hit"]) / len(rows)

        print("\n================ SUMMARY ================")
        print(f"Evaluated Cases: {len(rows)}")
        print(f"Exact Gold Hit Rate (Recall@3): {hit_rate:.3f}")
        print(f"Average Contextual Recall Score: {avg_score:.3f}")
        print("=========================================")

    return 0


if __name__ == "__main__":
    sys.exit(run_eval())
