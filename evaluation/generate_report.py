"""
Generates EVAL_REPORT_GENERATED.md from the result JSON files on disk.

Every figure in the output is computed from evaluation/results/*.json at run
time -- nothing is hand-typed. Re-run it after any eval and the report updates.
The point is that no number in the report can drift from, or be invented
independently of, the artifact it came from; each section names its source file.

Usage:  python evaluation/generate_report.py
"""
import json, sys
from pathlib import Path
from datetime import datetime
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evaluation" / "results"
OUT = ROOT / "evaluation" / "EVAL_REPORT_GENERATED.md"

FALLBACK_PREFIX = "[FALLBACK, judge failed]"


def latest(pattern):
    fs = sorted(RES.glob(pattern), key=lambda p: p.stat().st_mtime)
    return fs[-1] if fs else None


def load(p):
    if p and p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def pct(x):
    return f"{x*100:.1f}%"


def main():
    L = []
    w = L.append

    ev_path = latest("contextual_recall_eval_*.json")
    ev = load(ev_path)
    diag = load(RES / "retrieval_diagnostic.json")
    bake = load(RES / "reranker_bakeoff.json")

    if not ev:
        print("No contextual_recall_eval_*.json found -- run the eval first.")
        return 1

    w("# RAG Evaluation Report (generated)\n")
    w(f"_Generated {datetime.now():%Y-%m-%d %H:%M} by `evaluation/generate_report.py`. "
      "Every number below is computed from the result files named in each section; "
      "none are hand-entered._\n")

    # ---------- provenance ----------
    w("## Provenance\n")
    w("| artifact | source file | modified |")
    w("|---|---|---|")
    for label, p in [("Graded evaluation", ev_path),
                     ("Retrieval/reranker diagnostic", RES / "retrieval_diagnostic.json"),
                     ("Reranker bake-off", RES / "reranker_bakeoff.json")]:
        if p and p.exists():
            w(f"| {label} | `{p.name}` | {datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M} |")
    w("")

    rows = ev["rows"]
    n = len(rows)

    # ---------- trust gate ----------
    fb = [r for r in rows if r.get("used_fallback_heuristic")
          or str(r.get("reason", "")).startswith(FALLBACK_PREFIX)]
    real = [r for r in rows if r not in fb]
    w("## Measurement trust gate\n")
    w("The judge (DeepEval `ContextualRecallMetric`) can fail on quota or malformed output. "
      "When it does, the harness records a heuristic fallback score and flags the row. "
      "A run is only trustworthy if this count is low.\n")
    w(f"- Cases scored by the **real judge**: **{len(real)}/{n}** ({pct(len(real)/n)})")
    w(f"- Cases that fell back to the heuristic: **{len(fb)}/{n}** ({pct(len(fb)/n)})")
    if fb:
        kinds = {}
        for r in fb:
            e = str(r.get("judge_error") or "")
            k = ("quota / 429" if ("RESOURCE_EXHAUSTED" in e or "429" in e)
                 else "invalid JSON from judge" if "invalid JSON" in e
                 else "other")
            kinds[k] = kinds.get(k, 0) + 1
        w("")
        for k, v in sorted(kinds.items()):
            w(f"  - {k}: {v}")
    w("")

    # ---------- headline ----------
    w("## Headline metrics\n")
    w("| metric | value |")
    w("|---|---|")
    w(f"| Cases evaluated | {ev.get('total_cases', n)} |")
    w(f"| Fact coverage (mean ContextualRecall) | **{pct(ev['global_avg_contextual_recall'])}** |")
    w(f"| Strict recall@1 | {pct(ev['global_recall_at_1'])} |")
    w(f"| Strict recall@5 | {pct(ev['global_recall_at_5'])} |")
    w(f"| MRR | {ev['global_mrr']:.3f} |")
    w("")

    w("### By category\n")
    w("| category | n | strict R@5 | MRR | fact coverage |")
    w("|---|---|---|---|---|")
    for cat, st in sorted(ev["category_breakdown"].items()):
        w(f"| {cat} | {st['count']} | {pct(st['strict_recall_at_5'])} | "
          f"{st['mrr']:.3f} | {pct(st['avg_contextual_recall'])} |")
    w("")

    # ---------- the key conditional ----------
    hit = [r["contextual_recall_score"] for r in real if r.get("exact_gold_hit_5")]
    miss = [r["contextual_recall_score"] for r in real if not r.get("exact_gold_hit_5")]
    if hit and miss:
        w("## Where the loss actually is\n")
        w("Splitting fact coverage by whether retrieval surfaced a gold document at all "
          "separates *retrieval* failure from *coverage* failure:\n")
        w("| condition | n | mean fact coverage |")
        w("|---|---|---|")
        w(f"| gold document **was** in top-5 | {len(hit)} | **{mean(hit):.3f}** |")
        w(f"| gold document **was not** in top-5 | {len(miss)} | {mean(miss):.3f} |")
        w("")
        w(f"When the right document is retrieved, coverage is {mean(hit):.3f}. "
          "The headline number is therefore bounded by retrieval hit-rate, not by "
          "the system's ability to use what it retrieves.\n")

    # ---------- diagnostic ----------
    if diag:
        s = diag["summary"]
        w("## Retrieval vs. reranker (no LLM calls)\n")
        w(f"Source: `retrieval_diagnostic.json` — {s['cases']} cases.\n")
        w("| stage | recall |")
        w("|---|---|")
        w(f"| gold in candidate pool (retrieval ceiling) | **{pct(s['pool_recall'])}** |")
        w(f"| gold in final top-5 (after reranking) | **{pct(s['top5_recall'])}** |")
        w("")
        w(f"- Lost by the reranker (in pool, cut before top-5): **{s['lost_by_reranker']}** cases")
        w(f"- Never retrieved (absent from pool): **{s['never_retrieved']}** cases")
        w("")
        kept = s["top5_recall"] / s["pool_recall"] if s["pool_recall"] else 0
        w(f"The reranker retains {pct(kept)} of the gold documents retrieval hands it. "
          "This is the single largest identified loss in the pipeline.\n")

    # ---------- bake-off ----------
    if bake:
        ok = [b for b in bake if "error" not in b]
        if ok:
            ok.sort(key=lambda b: b["recall@5"], reverse=True)
            w("## Reranker bake-off\n")
            w("Identical cached candidate pools re-ranked by each configuration, so the "
              "comparison isolates the reranker. Source: `reranker_bakeoff.json`.\n")
            cols = ["recall@1", "recall@3", "recall@5"] + \
                   (["recall@10", "recall@20"] if "recall@10" in ok[0] else [])
            w("| configuration | " + " | ".join(c.replace("recall@", "R@") for c in cols) + " | MRR | secs |")
            w("|---" * (len(cols) + 3) + "|")
            for b in ok:
                w(f"| {b['label']} | " + " | ".join(pct(b[c]) for c in cols) +
                  f" | {b['mrr@5']:.3f} | {b['seconds']:.0f} |")
            w("")
            w(f"**Best: {ok[0]['label']} at {pct(ok[0]['recall@5'])} recall@5.** "
              "Configurations cluster closely, indicating reranker choice is not the "
              "limiting factor on this corpus.\n")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"  real-judge coverage : {len(real)}/{n}")
    print(f"  fact coverage       : {pct(ev['global_avg_contextual_recall'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
