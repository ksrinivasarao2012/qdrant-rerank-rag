"""
Step 1 of pooled judging: build the list of (query, document) pairs that need a
human relevance judgment.

WHY THIS EXISTS
---------------
The golden set names ONE answer_id per query. Cross Validated routinely has
several good answers to the same question, and several good threads for the same
topic. So a run is currently scored as a miss whenever it returns a *different
correct answer* than the one that happens to be labelled. Measured on the 79 hard
cases: of the 40 queries whose gold falls outside the top-5, 8 (20%) already have
a different answer to the SAME question sitting at rank 1-5, and eyeballing the
rest shows more that are correct-but-different-thread.

This script collects everything the system actually returned that has never been
judged, so a human can rule on it once. Nothing is written to the golden set --
judgments live in their own file and are applied at scoring time.

Usage:
  python evaluation/build_judgment_pool.py                 # misses only (default)
  python evaluation/build_judgment_pool.py --all           # every case
  python evaluation/build_judgment_pool.py --top-n 10      # judge deeper
  python evaluation/build_judgment_pool.py --run results/<other_run>.json
"""
import json, argparse, sys
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
RES  = ROOT / "evaluation" / "results"
DEFAULT_RUN = RES / "retriever_eval_hybrid_rerank_raw_pool50_20260901_022349.json"


def dedup(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(DEFAULT_RUN), help="retriever eval result JSON")
    ap.add_argument("--golden", default="evaluation/golden_dataset.json",
                    help="golden set to judge against (default v1 -- v2 labels are "
                         "retrieval-derived and inflate scores)")
    ap.add_argument("--top-n", type=int, default=5, help="judge the top N of each ranking")
    ap.add_argument("--all", action="store_true", help="include cases that already hit at 5")
    ap.add_argument("--categories", default="",
                    help="comma-separated categories to include, e.g. "
                         "negation,multi_hop,multi_turn,niche_topic")
    ap.add_argument("--out", default="evaluation/results/judgment_pool.json")
    a = ap.parse_args()

    run    = json.load(open(ROOT / a.run if not Path(a.run).is_absolute() else a.run, encoding="utf-8"))
    golden = {c["query_id"]: c for c in json.load(open(ROOT / a.golden, encoding="utf-8"))}
    pools_p = RES / "_candidate_pools.json"
    pools  = json.load(open(pools_p, encoding="utf-8")) if pools_p.exists() else {}

    # every answer_id we might need text for
    need = set()
    for row in run:
        need.update(row.get("ranked_top10") or [])
    for p in pools.values():
        need.update(c["answer_id"] for c in p[: a.top_n * 4])
    for c in golden.values():
        need.update(c.get("gold_answer_ids") or [])

    posts = {}
    with open(ROOT / "data/processed/posts.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            i = str(p.get("answer_id"))
            if i in need:
                posts[i] = {
                    "question_id": str(p.get("question_id")),
                    "title": p.get("question_title", ""),
                    "text": (p.get("answer_text") or ""),
                    "score": p.get("score"),
                    "accepted": p.get("is_accepted"),
                    "url": p.get("url", ""),
                }

    want_cats = {c.strip() for c in a.categories.split(",") if c.strip()}
    cases, n_skip = [], 0
    for row in run:
        qid = row["query_id"]
        case = golden.get(qid)
        if not case:
            continue
        cat = row.get("category") or case.get("category")
        if want_cats and cat not in want_cats:
            continue
        golds = [str(g) for g in (case.get("gold_answer_ids") or [])]
        if not golds:
            continue                                    # refusal case, nothing to judge

        reranked = [str(x) for x in (row.get("ranked_top10") or [])][: a.top_n]
        rrf      = dedup([c["answer_id"] for c in pools.get(qid, [])])[: a.top_n]

        hit = any(g in reranked[:5] for g in golds)
        if hit and not a.all:
            n_skip += 1
            continue

        cands, seen = [], set()
        for src, ids in (("rerank", reranked), ("rrf", rrf)):
            for rank, aid in enumerate(ids, 1):
                if aid in seen or aid in golds:
                    continue
                seen.add(aid)
                p = posts.get(aid, {})
                cands.append({
                    "answer_id": aid,
                    "source": src,
                    "rank": rank,
                    "question_id": p.get("question_id"),
                    "title": p.get("title", "(text unavailable)"),
                    "snippet": " ".join((p.get("text") or "").split())[:700],
                    "score": p.get("score"),
                    "url": p.get("url", ""),
                })

        cases.append({
            "query_id": qid,
            "sub_label": row.get("sub_label"),
            "category": row.get("category") or case.get("category"),
            "query": row.get("query"),
            "gold_ids": golds,
            "gold": [{"answer_id": g,
                      "title": posts.get(g, {}).get("title", "?"),
                      "question_id": posts.get(g, {}).get("question_id"),
                      "snippet": " ".join((posts.get(g, {}).get("text") or "").split())[:500]}
                     for g in golds],
            "candidates": cands,
        })

    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "run": str(a.run),
        "golden": a.golden,
        "top_n": a.top_n,
        "scope": "all" if a.all else "misses_only",
        "categories": sorted(want_cats) or "all",
        "n_cases": len(cases),
        "n_judgments_needed": sum(len(c["candidates"]) for c in cases),
        "cases": cases,
    }
    dest = ROOT / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print(f"run     : {Path(a.run).name}")
    print(f"golden  : {a.golden}")
    print(f"scope   : {out['scope']}  (skipped {n_skip} cases that already hit at 5)")
    print(f"cases   : {out['n_cases']}")
    print(f"pairs to judge: {out['n_judgments_needed']}")
    print(f"\nwrote {dest}")
    print("next: python evaluation/judge_pool.py")


if __name__ == "__main__":
    main()
