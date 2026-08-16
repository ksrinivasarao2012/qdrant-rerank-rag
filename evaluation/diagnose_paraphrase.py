"""
One-off diagnostic: are paraphrase_group "misses" real retrieval failures, or
label artifacts?

paraphrase_group is 80 of 314 query instances and ~47% of all unretrieved gold
answers (recall@100 = 0.38). But its gold answer was chosen by a single
keyword match in build_golden_dataset.py -- one arbitrary post out of however
many good answers Cross Validated has on that topic. The programmatically-built
categories (standard/code_traceback/citation_accuracy), whose query IS the gold
post's own title, score 0.90-1.00. That gap may be measuring label uniqueness
rather than retrieval quality.

This prints, for a handful of paraphrase cases, what actually came back --
so the question can be answered by reading rather than guessing:

  - if the top hits are good answers to the query that simply aren't the
    labeled gold  -> label artifact; the metric understates the retriever
  - if the top hits are off-topic                     -> real retrieval failure

Forces the local on-disk Qdrant index (force_local=True), same as
diagnose_pipeline_stages.py. Requires evaluation/load_local_qdrant.py to have
been run first.

Run with:
    python evaluation/diagnose_paraphrase.py
    python evaluation/diagnose_paraphrase.py --n 10 --top 5 --category negation
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Prevent PyTorch / OpenMP CPU execution deadlocks on Windows (same as
# eval_retriever.py -- must be set before torch is imported).
for _v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{_v}_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.vector_store import VectorDBManager
from backend.core.reranker import ReRanker

GOLDEN_JSON_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
POSTS_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "posts.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description="Inspect what retrieval actually returns.")
    p.add_argument("--category", default="paraphrase_group",
                   help="Golden-dataset category to inspect (default: paraphrase_group).")
    p.add_argument("--n", type=int, default=8, help="How many cases to inspect (default: 8).")
    p.add_argument("--top", type=int, default=5, help="How many hits to show per query (default: 5).")
    p.add_argument("--pool", type=int, default=50, help="Candidate pool before reranking (default: 50).")
    return p.parse_args()


def gold_lookup(needed_ids: set) -> dict:
    """Streams posts.jsonl once, keeping only the gold answers we need."""
    lookup, remaining = {}, set(needed_ids)
    with open(POSTS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            post = json.loads(line)
            if post["answer_id"] in remaining:
                lookup[post["answer_id"]] = post
                remaining.discard(post["answer_id"])
    return lookup


def query_for_case(case: dict) -> str:
    """Paraphrase groups carry `variants` instead of a single `query`."""
    if case.get("variants"):
        return case["variants"][0]
    return case.get("query", "")


def main():
    args = parse_args()

    with open(GOLDEN_JSON_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    selected = [c for c in cases
                if c["category"] == args.category and c.get("gold_answer_ids")][:args.n]
    if not selected:
        print(f"No cases with gold answers found for category '{args.category}'.")
        return 1

    gold_ids = {int(a) for c in selected for a in c["gold_answer_ids"]}
    print(f"Looking up {len(gold_ids)} gold answers in posts.jsonl...")
    gold = gold_lookup(gold_ids)

    print("Connecting to Qdrant (local)...")
    db = VectorDBManager(force_local=True)
    db.collection
    print("Loading reranker...")
    reranker = ReRanker()

    verdict_prompt = []
    for case in selected:
        query = query_for_case(case)
        gold_aids = [int(a) for a in case["gold_answer_ids"]]
        gold_qids = {gold[a]["question_id"] for a in gold_aids if a in gold}

        print("\n" + "=" * 78)
        print(f"{case['query_id']}  [{case['category']}]")
        print(f"QUERY: {query}")
        for a in gold_aids:
            g = gold.get(a)
            title = g["question_title"] if g else "(not found in posts.jsonl)"
            print(f"GOLD:  a/{a}  q/{g['question_id'] if g else '?'}  \"{title}\"")
        print("-" * 78)

        hits = db.search_hybrid(query, n_results=args.pool)
        if not hits:
            print("  (no hits returned)")
            continue
        ranked = reranker.rerank(query, hits, top_k=len(hits))

        shown, seen = 0, set()
        for chunk in ranked:
            meta = chunk["metadata"]
            aid = int(meta.get("answer_id", -1))
            if aid in seen:
                continue          # dedupe: several chunks can share an answer
            seen.add(aid)
            shown += 1
            qid = meta.get("question_id")
            marks = []
            if aid in gold_aids:
                marks.append("<< GOLD")
            elif qid in gold_qids:
                marks.append("<< same thread as gold")
            print(f"  {shown}. a/{aid:<8} q/{qid:<8} score={meta.get('score', 0):>4} "
                  f"{'ACC' if meta.get('is_accepted') else '   '}  "
                  f"{str(meta.get('question_title'))[:60]} {' '.join(marks)}")
            if shown >= args.top:
                break
        verdict_prompt.append(case["query_id"])

    print("\n" + "=" * 78)
    print("READ THE TITLES ABOVE AND ANSWER ONE QUESTION PER CASE:")
    print("  Would a user asking that query be satisfied by any of those posts?")
    print()
    print("  mostly YES -> the misses are label artifacts. The single hand-picked")
    print("               gold understates the retriever; report question-level or")
    print("               human-judged relevance alongside exact-gold recall.")
    print("  mostly NO  -> real retrieval failure. Upgrading the embedding model")
    print("               (bge-small 384d -> bge-base 768d) is the right fix.")
    print(f"\nInspected: {', '.join(verdict_prompt)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
