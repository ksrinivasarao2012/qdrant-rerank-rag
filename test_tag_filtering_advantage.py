"""
Demonstration & Empirical Test: Advantages of Metadata Tag Filtering in Qdrant RAG.
Runs ambiguous queries with and without tag filters and displays the exact retrieved citations.
"""

import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.vector_store import VectorDBManager
from backend.core.reranker import ReRanker

def run_test():
    print("=" * 80)
    print("   EMPIRICAL TEST: ADVANTAGES OF METADATA TAG FILTERING IN QDRANT RAG")
    print("=" * 80)

    db_manager = VectorDBManager()
    db_manager.collection
    reranker = ReRanker()

    test_scenarios = [
        {
            "title": "Scenario 1: Disambiguating Polysemous Statistical Terms ('Kernel')",
            "query": "How to choose the optimal kernel and its hyperparameter?",
            "filters": [
                ("🔍 All Topics (No Filter)", None),
                ("🏷️ distributions (KDE Bandwidth)", "distributions"),
                ("🏷️ machine-learning (SVM Margin)", "machine-learning")
            ]
        },
        {
            "title": "Scenario 2: Enforcing Specific Paradigms ('Confidence vs Credible Intervals')",
            "query": "How to construct parameter intervals and quantify estimation uncertainty?",
            "filters": [
                ("🔍 All Topics (General Frequentist)", None),
                ("🏷️ bayesian (Posterior Credible Intervals)", "bayesian")
            ]
        },
        {
            "title": "Scenario 3: Domain-Constrained Regression ('Time-Series Dependencies')",
            "query": "How to test for autocorrelation and lag structure in residuals?",
            "filters": [
                ("🔍 All Topics (No Filter)", None),
                ("🏷️ time-series (ARIMA / ACF Analysis)", "time-series")
            ]
        }
    ]

    for sc in test_scenarios:
        print(f"\n\n{'='*80}")
        print(f"📌 {sc['title']}")
        print(f"❓ Query: \"{sc['query']}\"")
        print(f"{'='*80}")

        for label, tag_filter in sc["filters"]:
            print(f"\n--- [Filter: {label}] ---")
            
            # Step 1: Hybrid Search with / without Tag Filter
            candidates = db_manager.search_hybrid(
                query=sc["query"],
                n_results=10,
                source_file=tag_filter
            )
            
            # Step 2: Cross-Encoder Rerank
            reranked = reranker.rerank(query=sc["query"], chunks=candidates, top_k=2)

            if not reranked:
                print("  [No matching chunks found with this filter constraint]")
                continue

            for rank, chunk in enumerate(reranked, 1):
                meta = chunk.get("metadata", {})
                title = meta.get("question_title", "Untitled")
                tags = meta.get("tags", [])
                score = chunk.get("rerank_score", 0.0)
                snippet = chunk.get("text", "")[:180].replace("\n", " ")
                print(f"  Rank {rank} (Score: {score:.3f}): {title}")
                print(f"    Tags: {tags}")
                print(f"    Snippet: \"{snippet}...\"\n")

if __name__ == "__main__":
    run_test()
