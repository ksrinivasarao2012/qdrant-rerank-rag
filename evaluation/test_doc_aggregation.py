import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOLS_PATH = ROOT / "evaluation" / "results" / "_candidate_pools.json"
GOLDEN_PATH = ROOT / "evaluation" / "golden_dataset.json"

def main():
    if not POOLS_PATH.exists():
        print("Candidate pools file not found.")
        return

    pools = json.load(open(POOLS_PATH, encoding="utf-8"))
    cases = json.load(open(GOLDEN_PATH, encoding="utf-8"))

    case_golds = {}
    for c in cases:
        qid = c["query_id"]
        golds = set(str(g) for g in c.get("gold_answer_ids", []))
        if c.get("graded_relevance"):
            golds.update(str(k) for k in c["graded_relevance"])
        if c.get("candidate_gold_ids"):
            golds.update(str(k) for k in c["candidate_gold_ids"])
        case_golds[qid] = golds

    total = 0
    hit5_chunk = 0
    hit5_doc_max = 0
    hit5_doc_sum = 0

    for qid, pool in pools.items():
        if qid not in case_golds or not case_golds[qid]:
            continue
        total += 1
        golds = case_golds[qid]

        # 1. Standard Chunk-level Top-5
        chunk_top5_aids = set(str(item.get("answer_id") or item.get("metadata", {}).get("answer_id", "")) for item in pool[:5])
        if golds & chunk_top5_aids:
            hit5_chunk += 1

        # 2. Document-level Aggregation (Max Score)
        doc_max_scores = {}
        doc_sum_scores = {}
        for rank, item in enumerate(pool):
            aid = str(item.get("answer_id") or item.get("metadata", {}).get("answer_id", ""))
            score = 1.0 / (rank + 1.0)
            doc_max_scores[aid] = max(doc_max_scores.get(aid, 0.0), score)
            doc_sum_scores[aid] = doc_sum_scores.get(aid, 0.0) + score

        sorted_by_max = sorted(doc_max_scores.keys(), key=lambda k: doc_max_scores[k], reverse=True)[:5]
        if golds & set(sorted_by_max):
            hit5_doc_max += 1

        sorted_by_sum = sorted(doc_sum_scores.keys(), key=lambda k: doc_sum_scores[k], reverse=True)[:5]
        if golds & set(sorted_by_sum):
            hit5_doc_sum += 1

    print("==========================================================================")
    print("DOCUMENT-LEVEL AGGREGATION TEST RESULTS")
    print("==========================================================================")
    print(f"Total Evaluated Queries: {total}")
    print(f"Chunk-level Top-5 Recall:             {hit5_chunk}/{total} = {hit5_chunk/total*100:.1f}%")
    print(f"Answer-level Top-5 (Max Aggregation): {hit5_doc_max}/{total} = {hit5_doc_max/total*100:.1f}%")
    print(f"Answer-level Top-5 (Sum Aggregation): {hit5_doc_sum}/{total} = {hit5_doc_sum/total*100:.1f}%")
    print("==========================================================================")

if __name__ == "__main__":
    main()
