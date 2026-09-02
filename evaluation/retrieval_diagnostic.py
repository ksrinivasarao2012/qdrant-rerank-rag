"""
Retrieval vs. Reranker diagnostic. Makes ZERO LLM API calls.

Answers one question: when the gold document does NOT end up in the final
top-5, was it ever in the candidate pool at all?

  - In the pool but not in the final top-5  -> the RERANKER is dropping it.
  - Never in the pool                        -> EMBEDDINGS / SPARSE / CHUNKING
                                                never surfaced it.

These need completely different fixes, so this decides where the effort goes.

Note: this uses the deterministic no-LLM query path (history concatenation +
heuristic decomposition) rather than the LLM rewriter, so it needs no API
quota and is fully reproducible. That makes absolute numbers slightly
pessimistic vs. the graded eval for multi_turn, but the POOL-vs-TOP5 gap --
the thing being measured -- is unaffected, since both stages see the same query.

Run:  python evaluation/retrieval_diagnostic.py
"""
import sys, os, json
from pathlib import Path
from collections import defaultdict

for _v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{_v}_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
torch.set_num_threads(1)

from backend.core.vector_store import VectorDBManager
from backend.core.reranker import ReRanker
from backend.core.llm_service import build_search_query, decompose_query

GOLDEN = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
CATS = {"multi_hop", "negation", "niche_topic", "multi_turn"}
TOP_K = 5


def gold_ids_for(case):
    s = set(str(g) for g in case.get("gold_answer_ids", []))
    if case.get("graded_relevance"):
        s.update(str(k) for k in case["graded_relevance"])
    if case.get("candidate_gold_ids"):
        s.update(str(k) for k in case["candidate_gold_ids"])
    return s


def main():
    cases = [c for c in json.load(open(GOLDEN, encoding="utf-8"))
             if c.get("gold_answer_ids") and c.get("query") and c.get("category") in CATS]
    print(f"Diagnosing {len(cases)} cases (no LLM calls)...\n")

    db = VectorDBManager()
    db.collection
    reranker = ReRanker()

    rows = []
    for i, case in enumerate(cases, 1):
        cat = case.get("category", "")
        cand_limit = 100 if cat in {"multi_hop", "niche_topic"} else 80 if cat in {"multi_turn", "negation"} else 50

        search_query = build_search_query(case["query"], case.get("chat_history"), None)
        subqs = decompose_query(search_query, llm_service=None)

        if len(subqs) > 1:
            candidates = db.search_multi_query(subqs, n_results=cand_limit)
        else:
            candidates = db.search_hybrid(query=search_query, n_results=cand_limit)

        gold = gold_ids_for(case)
        cand_ids = [str(c["metadata"].get("answer_id")) for c in candidates]

        pool_hit = bool(gold & set(cand_ids))
        pool_rank = next((idx for idx, a in enumerate(cand_ids, 1) if a in gold), None)

        reranked = reranker.rerank(query=case["query"], chunks=candidates, top_k=TOP_K)
        top_ids = [str(c["metadata"].get("answer_id")) for c in reranked]
        top5_hit = bool(gold & set(top_ids))

        rows.append({
            "qid": case["query_id"], "cat": cat, "pool_size": len(candidates),
            "pool_hit": pool_hit, "pool_rank": pool_rank, "top5_hit": top5_hit,
        })
        flag = "OK " if top5_hit else ("LOST-BY-RERANKER" if pool_hit else "NOT-IN-POOL")
        print(f"[{i:>2}/{len(cases)}] {case['query_id']:<10} {cat:<12} pool={len(candidates):>3} "
              f"gold_rank_in_pool={str(pool_rank):>5}  {flag}")

    # ---------------- summary ----------------
    n = len(rows)
    pool = sum(r["pool_hit"] for r in rows)
    top5 = sum(r["top5_hit"] for r in rows)
    lost = sum(1 for r in rows if r["pool_hit"] and not r["top5_hit"])
    never = sum(1 for r in rows if not r["pool_hit"])

    print("\n" + "=" * 74)
    print("RETRIEVAL vs RERANKER DIAGNOSTIC")
    print("=" * 74)
    print(f"Cases                                            : {n}")
    print(f"Gold IN candidate pool  (retrieval ceiling)      : {pool}/{n}  ({pool/n:.1%})")
    print(f"Gold IN final top-5     (what you actually get)  : {top5}/{n}  ({top5/n:.1%})")
    print("-" * 74)
    print(f"LOST BY RERANKER  (in pool, cut before top-5)    : {lost}/{n}  ({lost/n:.1%})")
    print(f"NEVER RETRIEVED   (absent from pool entirely)    : {never}/{n}  ({never/n:.1%})")
    print("=" * 74)

    print("\nPer category:")
    print(f"{'category':<14}{'n':>4}{'pool recall':>14}{'top5 recall':>13}{'lost by rerank':>17}")
    bycat = defaultdict(list)
    for r in rows:
        bycat[r["cat"]].append(r)
    for cat, rs in sorted(bycat.items()):
        m = len(rs)
        p = sum(x["pool_hit"] for x in rs)
        t = sum(x["top5_hit"] for x in rs)
        l = sum(1 for x in rs if x["pool_hit"] and not x["top5_hit"])
        print(f"{cat:<14}{m:>4}{p/m:>13.1%}{t/m:>13.1%}{l:>13} ({l/m:.0%})")

    ranks = [r["pool_rank"] for r in rows if r["pool_hit"]]
    if ranks:
        ranks.sort()
        print(f"\nWhere gold sits in the candidate pool (when present), n={len(ranks)}:")
        print(f"  median rank {ranks[len(ranks)//2]} | in top-10 of pool: "
              f"{sum(1 for x in ranks if x <= 10)} | in top-25: {sum(1 for x in ranks if x <= 25)}")

    out = PROJECT_ROOT / "evaluation" / "results" / "retrieval_diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": {"cases": n, "pool_recall": pool / n, "top5_recall": top5 / n,
                               "lost_by_reranker": lost, "never_retrieved": never}, "rows": rows},
                  f, indent=2)
    print(f"\nSaved: {out}")
    print("\nHOW TO READ THIS:")
    print("  LOST BY RERANKER high  -> fix the reranker (it's cutting documents it should keep)")
    print("  NEVER RETRIEVED high   -> fix embeddings / chunking / query construction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
