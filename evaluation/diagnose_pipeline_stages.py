"""
One-off diagnostic: trace a handful of failing `standard` golden-dataset
queries through each stage of the retrieval pipeline, to find exactly where
the gold answer drops out.

Stages checked, in order:
  1. Dense-only search (top 50)
  2. Sparse-only search (top 50)
  3. Hybrid RRF-fused search (top 50) -- what actually feeds the reranker
  4. After cross-encoder reranking (top 10) -- what the user actually sees

This follows diagnose_stale_points.py, which already ruled out stale Qdrant
Cloud data as the cause (local-index eval_retriever.py run produced nearly
identical numbers to the cloud run). This script assumes the pipeline code
is otherwise correct and is looking for where retrieval quality is actually
being lost.

Run locally. Works against whichever Qdrant the current environment points
to -- set QDRANT_URL/QDRANT_API_KEY blank first to use the clean local index
built by load_local_qdrant.py (recommended, since it's equally valid and
faster):

    (PowerShell)
    $env:QDRANT_URL=""; $env:QDRANT_API_KEY=""; python evaluation/diagnose_pipeline_stages.py
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.vector_store import VectorDBManager
from backend.core.reranker import ReRanker

GOLDEN_JSON_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
N_CASES_TO_CHECK = 6
POOL = 50


def find_rank(hits, gold_ids, dedupe=True):
    """Returns the 1-based rank of the first gold-answer hit, or None."""
    seen = set()
    rank = 0
    for chunk in hits:
        aid = str(chunk["metadata"].get("answer_id"))
        if dedupe:
            if aid in seen:
                continue
            seen.add(aid)
        rank += 1
        if aid in gold_ids:
            return rank
    return None


def main():
    with open(GOLDEN_JSON_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    standard_cases = [c for c in cases if c["category"] == "standard"][:N_CASES_TO_CHECK]

    print("Connecting to Qdrant...")
    db = VectorDBManager()
    db.collection

    print("Loading reranker...")
    reranker = ReRanker()

    for case in standard_cases:
        query = case["query"]
        gold_ids = set(case["gold_answer_ids"])
        print(f"\n=== {case['query_id']}: \"{query}\" ===")
        print(f"  gold_answer_ids: {gold_ids}")

        dense_hits = db.search(query, n_results=POOL)
        sparse_hits = db.search_sparse(query, n_results=POOL)
        hybrid_hits = db.search_hybrid(query, n_results=POOL)

        dense_rank = find_rank(dense_hits, gold_ids)
        sparse_rank = find_rank(sparse_hits, gold_ids)
        hybrid_rank = find_rank(hybrid_hits, gold_ids)

        print(f"  dense-only   top{POOL}: gold at rank {dense_rank}")
        print(f"  sparse-only  top{POOL}: gold at rank {sparse_rank}")
        print(f"  hybrid(RRF)  top{POOL}: gold at rank {hybrid_rank}")

        if hybrid_hits:
            reranked = reranker.rerank(query, hybrid_hits, top_k=len(hybrid_hits))
            reranked_rank = find_rank(reranked, gold_ids)
            reranked_rank_at10 = reranked_rank if reranked_rank and reranked_rank <= 10 else None
            print(f"  after rerank top{POOL}: gold at rank {reranked_rank}"
                  f"  (within top10: {'YES' if reranked_rank_at10 else 'NO'})")
        else:
            print("  after rerank: skipped -- hybrid returned 0 hits")


if __name__ == "__main__":
    main()
