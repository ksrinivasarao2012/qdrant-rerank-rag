"""
Reranker bake-off. Makes ZERO LLM API calls.

Retrieves the candidate pool ONCE per case (cached to disk), then re-ranks
those identical pools with several reranker configurations and reports
recall@1/3/5 and MRR for each. Same pools for every config = a fair,
apples-to-apples comparison.

Motivated by evaluation/retrieval_diagnostic.py, which showed the candidate
pool contains the gold document 78.5% of the time while only 36.7% survives
to the final top-5 -- i.e. the reranker discards 53% of what search finds.

Usage:
  python evaluation/reranker_bakeoff.py                 # default configs
  python evaluation/reranker_bakeoff.py --limit 25      # quick subset first
  python evaluation/reranker_bakeoff.py --heavy         # also test bge-reranker-v2-m3 (slow)
  python evaluation/reranker_bakeoff.py --refresh-pools # rebuild the pool cache

NOTE: BGE models download from HuggingFace on first use (base ~1.1GB,
v2-m3 ~2.3GB). This script deliberately does NOT enable HF offline mode.
Runtime is dominated by the big models on CPU -- start with --limit.
"""
import os, sys, json, time, argparse
from pathlib import Path

# Deliberately NOT offline -- BGE models may need downloading.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
for _v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ.setdefault(f"{_v}_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

GOLDEN = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
POOL_CACHE = PROJECT_ROOT / "evaluation" / "results" / "_candidate_pools.json"
OUT = PROJECT_ROOT / "evaluation" / "results" / "reranker_bakeoff.json"
CATS = {"multi_hop", "negation", "niche_topic", "multi_turn"}

# (label, hf_model_name_or_None, char_window)
BASE_CONFIGS = [
    ("no reranker (raw hybrid RRF)",      None,                                    0),
    ("ms-marco-MiniLM-L-6  @600",         "cross-encoder/ms-marco-MiniLM-L-6-v2",  600),
    ("ms-marco-MiniLM-L-6  @2000",        "cross-encoder/ms-marco-MiniLM-L-6-v2",  2000),
    ("ms-marco-MiniLM-L-12 @600",         "cross-encoder/ms-marco-MiniLM-L-12-v2", 600),
    ("bge-reranker-base    @2000",        "BAAI/bge-reranker-base",                2000),
]
HEAVY_CONFIGS = [
    ("bge-reranker-v2-m3   @2000",        "BAAI/bge-reranker-v2-m3",               2000),
]


def gold_ids_for(case):
    s = set(str(g) for g in case.get("gold_answer_ids", []))
    if case.get("graded_relevance"):
        s.update(str(k) for k in case["graded_relevance"])
    if case.get("candidate_gold_ids"):
        s.update(str(k) for k in case["candidate_gold_ids"])
    return s


def build_pools(cases):
    """Retrieve candidate pools once (no LLM calls) and cache them."""
    from backend.core.vector_store import VectorDBManager
    from backend.core.llm_service import build_search_query, decompose_query

    db = VectorDBManager()
    db.collection
    pools = {}
    for i, case in enumerate(cases, 1):
        cat = case.get("category", "")
        k = 100 if cat in {"multi_hop", "niche_topic"} else 80 if cat in {"multi_turn", "negation"} else 50
        sq = build_search_query(case["query"], case.get("chat_history"), None)
        subqs = decompose_query(sq, llm_service=None)
        cands = (db.search_multi_query(subqs, n_results=k) if len(subqs) > 1
                 else db.search_hybrid(query=sq, n_results=k))
        pools[case["query_id"]] = [
            {"answer_id": str(c["metadata"].get("answer_id")), "text": c.get("text", "")}
            for c in cands
        ]
        print(f"  pooled [{i:>2}/{len(cases)}] {case['query_id']:<10} {len(cands)} candidates")
    POOL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(POOL_CACHE, "w", encoding="utf-8") as f:
        json.dump(pools, f)
    return pools


def score_config(label, model_name, window, cases, pools, device, batch_size):
    encoder = None
    if model_name:
        from sentence_transformers import CrossEncoder
        t0 = time.time()
        print(f"\n>>> loading {model_name} (device={device}) ...")
        encoder = CrossEncoder(model_name, max_length=512, device=device)
        print(f"    loaded in {time.time()-t0:.0f}s")

    r1 = r3 = r5 = r10 = r20 = 0
    mrr_sum = 0.0
    per_cat = {}
    t0 = time.time()

    for idx, case in enumerate(cases, 1):
        qid, query = case["query_id"], case["query"]
        gold = gold_ids_for(case)
        pool = pools.get(qid, [])
        if not pool:
            continue

        if encoder is None:
            ordered = pool  # raw hybrid RRF order
        else:
            pairs = [[query, c["text"][:window]] for c in pool]
            scores = encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            ordered = [c for _, c in sorted(zip(scores, pool),
                                            key=lambda p: float(p[0]), reverse=True)]

        top = [c["answer_id"] for c in ordered[:20]]
        hit1 = bool(gold & set(top[:1])); hit3 = bool(gold & set(top[:3]))
        hit5 = bool(gold & set(top[:5])); hit10 = bool(gold & set(top[:10]))
        hit20 = bool(gold & set(top[:20]))
        rr = next((1.0 / r for r, a in enumerate(top[:5], 1) if a in gold), 0.0)
        r1 += hit1; r3 += hit3; r5 += hit5; r10 += hit10; r20 += hit20; mrr_sum += rr

        cat = case.get("category", "")
        d = per_cat.setdefault(cat, {"n": 0, "h5": 0})
        d["n"] += 1; d["h5"] += int(hit5)

        if idx % 20 == 0:
            print(f"    {idx}/{len(cases)} scored ({time.time()-t0:.0f}s elapsed)")

    n = len(cases)
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "label": label, "model": model_name, "window": window,
        "recall@1": r1 / n, "recall@3": r3 / n, "recall@5": r5 / n,
        "recall@10": r10 / n, "recall@20": r20 / n,
        "mrr@5": mrr_sum / n, "seconds": time.time() - t0,
        "per_category_recall5": {c: v["h5"] / v["n"] for c, v in per_cat.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N cases (quick pass)")
    ap.add_argument("--heavy", action="store_true", help="also test bge-reranker-v2-m3 (slow)")
    ap.add_argument("--refresh-pools", action="store_true", help="rebuild candidate pool cache")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fast", action="store_true",
                    help="only the cheap MiniLM configs (bge already ruled out)")
    args = ap.parse_args()

    cases = [c for c in json.load(open(GOLDEN, encoding="utf-8"))
             if c.get("gold_answer_ids") and c.get("query") and c.get("category") in CATS]
    if args.limit:
        cases = cases[:args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Bake-off over {len(cases)} cases | device={device}\n")

    if POOL_CACHE.exists() and not args.refresh_pools:
        pools = json.load(open(POOL_CACHE, encoding="utf-8"))
        missing = [c["query_id"] for c in cases if c["query_id"] not in pools]
        if missing:
            print(f"cache missing {len(missing)} cases -> rebuilding")
            pools = build_pools(cases)
        else:
            print(f"using cached candidate pools ({POOL_CACHE.name})")
    else:
        print("building candidate pools (retrieval only, no LLM)...")
        pools = build_pools(cases)

    configs = BASE_CONFIGS + (HEAVY_CONFIGS if args.heavy else [])
    if args.fast:
        configs = [c for c in configs if "bge" not in c[0]]
    results = []
    for label, model, window in configs:
        try:
            results.append(score_config(label, model, window, cases, pools, device, args.batch_size))
            r = results[-1]
            print(f"    -> {label}: R@5={r['recall@5']:.1%}  MRR={r['mrr@5']:.3f}  ({r['seconds']:.0f}s)")
        except Exception as e:
            print(f"    !! {label} FAILED: {type(e).__name__}: {e}")
            results.append({"label": label, "model": model, "window": window, "error": str(e)})

    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: r["recall@5"], reverse=True)

    print("\n" + "=" * 88)
    print("RERANKER BAKE-OFF RESULTS  (sorted by recall@5)")
    print("=" * 88)
    print(f"{'configuration':<32}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'R@10':>7}{'R@20':>7}{'MRR':>8}{'secs':>7}")
    print("-" * 88)
    for r in ok:
        print(f"{r['label']:<32}{r['recall@1']:>6.0%}{r['recall@3']:>7.0%}{r['recall@5']:>7.0%}"
              f"{r['recall@10']:>7.0%}{r['recall@20']:>7.0%}{r['mrr@5']:>8.3f}{r['seconds']:>7.0f}")
    print("=" * 88)

    if ok:
        best, cur = ok[0], next((r for r in ok if "MiniLM-L-6  @2000" in r["label"]), None)
        print(f"\nBEST: {best['label']}  ->  R@5 {best['recall@5']:.1%}")
        if cur and best["label"] != cur["label"]:
            print(f"CURRENT PRODUCTION CONFIG: {cur['label']} -> R@5 {cur['recall@5']:.1%}")
            print(f"IMPROVEMENT AVAILABLE: {best['recall@5']-cur['recall@5']:+.1%} recall@5")
        print("\nper-category recall@5 for the best config:")
        for c, v in sorted(best.get("per_category_recall5", {}).items()):
            print(f"  {c:<14}{v:>7.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
