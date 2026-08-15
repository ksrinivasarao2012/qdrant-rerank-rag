"""
One-off diagnostic: is the live Qdrant collection returning stale points?

Picks a handful of `standard` category golden-dataset cases that failed
recall@10 in the last eval_retriever.py run, re-runs search_hybrid() against
the LIVE collection, and checks each top hit's answer_id against the local
data/processed/embedded_points.jsonl (the known-good, current corpus).

A hit whose answer_id is NOT in embedded_points.jsonl did not come from the
current corpus -- it's leftover from an older seed run and confirms the
16,140-point discrepancy is actually polluting search results, not just
sitting there unused.

Run locally (needs live Qdrant access):
    python evaluation/diagnose_stale_points.py
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.vector_store import VectorDBManager

GOLDEN_JSON_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
EMBEDDED_POINTS_PATH = PROJECT_ROOT / "data" / "processed" / "embedded_points.jsonl"

N_CASES_TO_CHECK = 5
POOL = 50


def load_current_answer_ids() -> set:
    print("Loading known-good answer_ids from embedded_points.jsonl...")
    ids = set()
    with open(EMBEDDED_POINTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            ids.add(str(json.loads(line)["payload"]["answer_id"]))
    print(f"  {len(ids)} unique answer_ids in current corpus.\n")
    return ids


def main():
    with open(GOLDEN_JSON_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    standard_cases = [c for c in cases if c["category"] == "standard"][:N_CASES_TO_CHECK]
    current_ids = load_current_answer_ids()

    print("Connecting to Qdrant...")
    db = VectorDBManager()
    db.collection

    for case in standard_cases:
        query = case["query"]
        gold_ids = set(case["gold_answer_ids"])
        print(f"\n=== {case['query_id']}: \"{query}\" ===")
        print(f"  gold_answer_ids: {gold_ids}")

        hits = db.search_hybrid(query, n_results=POOL)
        seen = set()
        rows = []
        for chunk in hits:
            aid = str(chunk["metadata"].get("answer_id"))
            if aid in seen:
                continue
            seen.add(aid)
            is_current = aid in current_ids
            is_gold = aid in gold_ids
            rows.append((aid, is_current, is_gold))
            if len(rows) >= 10:
                break

        stale_count = sum(1 for _, is_current, _ in rows if not is_current)
        print(f"  top10 unique answer_ids -- stale (not in current corpus): {stale_count}/10")
        for aid, is_current, is_gold in rows:
            tag = []
            if is_gold:
                tag.append("GOLD")
            if not is_current:
                tag.append("STALE")
            print(f"    {aid:>12}  {' '.join(tag)}")


if __name__ == "__main__":
    main()
