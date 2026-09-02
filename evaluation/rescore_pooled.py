"""
Step 3 of pooled judging: score a run under three definitions of "correct" and
print them side by side.

  strict   the exact labelled answer_id is in the top-k.
           What the project has reported so far. Understates real performance,
           because the label names one answer on a corpus where a question
           routinely has several good ones.

  thread   any answer from the labelled answer's question thread is in the top-k.
           Already computed by the harness as qrecall. Catches sibling answers,
           misses correct answers from a different thread.

  pooled   the labelled answer, OR any document a human judged as answering this
           query (grade 2), is in the top-k. Grade 1 (partial) is reported
           separately as `pooled-lenient`.
           This is the number that reflects what a user would experience.

Judgment coverage is printed alongside. An unjudged document counts as
irrelevant, so `pooled` is a LOWER BOUND until coverage is complete -- report
coverage whenever you report the number.

Usage:
  python evaluation/rescore_pooled.py
  python evaluation/rescore_pooled.py --run results/<other_run>.json
  python evaluation/rescore_pooled.py --md            # markdown table for the report
"""
import json, sys, argparse, collections
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
RES  = ROOT / "evaluation" / "results"
JUDG = RES / "pooled_judgments.json"
DEFAULT_RUN = RES / "retriever_eval_hybrid_rerank_raw_pool50_20260901_022349.json"
KS = (1, 3, 5, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--golden", default="evaluation/golden_dataset.json")
    ap.add_argument("--categories", default="")
    ap.add_argument("--md", action="store_true", help="emit a markdown table")
    a = ap.parse_args()

    rp = Path(a.run) if Path(a.run).is_absolute() else ROOT / a.run
    run = json.load(open(rp, encoding="utf-8"))
    golden = {c["query_id"]: c for c in json.load(open(ROOT / a.golden, encoding="utf-8"))}
    judg = json.load(open(JUDG, encoding="utf-8")) if JUDG.exists() else {}
    judg.pop("_meta", None)
    want = {c.strip() for c in a.categories.split(",") if c.strip()}

    need = set()
    for row in run:
        need.update(str(x) for x in (row.get("ranked_top10") or []))
    for c in golden.values():
        need.update(str(g) for g in (c.get("gold_answer_ids") or []))
    qof = {}
    with open(ROOT / "data/processed/posts.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            i = str(p.get("answer_id"))
            if i in need:
                qof[i] = str(p.get("question_id"))

    metrics = ("strict", "thread", "pooled", "pooled-lenient")
    hits = {m: collections.Counter() for m in metrics}
    rr   = {m: [] for m in metrics}
    percat = collections.defaultdict(lambda: {m: collections.Counter() for m in metrics})
    percat_n = collections.Counter()
    n = 0
    judged_slots = total_slots = 0

    for row in run:
        case = golden.get(row["query_id"])
        if not case:
            continue
        golds = {str(g) for g in (case.get("gold_answer_ids") or [])}
        if not golds:
            continue
        cat = row.get("category") or case.get("category")
        if want and cat not in want:
            continue
        n += 1
        percat_n[cat] += 1

        ranked = [str(x) for x in (row.get("ranked_top10") or [])]
        gq = {qof.get(g) for g in golds if qof.get(g)}
        jr = judg.get(row["query_id"], {})

        for aid in ranked[:5]:
            total_slots += 1
            if aid in golds or jr.get(aid, {}).get("grade") is not None:
                judged_slots += 1

        rel = {
            "strict":         lambda aid: aid in golds,
            "thread":         lambda aid: aid in golds or qof.get(aid) in gq,
            "pooled":         lambda aid: aid in golds or jr.get(aid, {}).get("grade") == 2,
            "pooled-lenient": lambda aid: aid in golds or (jr.get(aid, {}).get("grade") or 0) >= 1,
        }
        for m, fn in rel.items():
            pos = [i + 1 for i, aid in enumerate(ranked) if fn(aid)]
            for k in KS:
                if pos and min(pos) <= k:
                    hits[m][k] += 1
                    percat[cat][m][k] += 1
            rr[m].append(1 / min(pos) if pos else 0.0)

    if not n:
        sys.exit("no evaluable cases matched")

    cov = judged_slots / total_slots if total_slots else 0
    n_j = sum(len(v) for v in judg.values())

    def row_vals(h, r):
        return [h[k] / n for k in KS] + [sum(r) / len(r)]

    if a.md:
        print(f"| definition of correct | " + " | ".join(f"R@{k}" for k in KS) + " | MRR |")
        print("|---|" + "---:|" * (len(KS) + 1))
        for m in metrics:
            v = row_vals(hits[m], rr[m])
            print(f"| {m} | " + " | ".join(f"{x:.1%}" for x in v[:-1]) + f" | {v[-1]:.3f} |")
        print(f"\n_n={n} query instances; {n_j} human judgments; "
              f"top-5 judgment coverage {cov:.0%}. Unjudged documents count as "
              f"irrelevant, so pooled figures are lower bounds._")
        return

    print(f"run     : {rp.name}")
    print(f"golden  : {a.golden}" + (f"   categories={sorted(want)}" if want else ""))
    print(f"n       : {n} query instances")
    print(f"judgments: {n_j} human judgments; top-5 coverage {cov:.0%} "
          f"({judged_slots}/{total_slots} returned slots have a verdict)")
    if cov < 0.99:
        print("           -> unjudged counts as irrelevant; pooled is a LOWER BOUND")
    print()
    print(f"{'definition of correct':22s}" + "".join(f"{'R@'+str(k):>9}" for k in KS) + f"{'MRR':>9}")
    print("-" * (22 + 9 * (len(KS) + 1)))
    for m in metrics:
        v = row_vals(hits[m], rr[m])
        print(f"{m:22s}" + "".join(f"{x:9.1%}" for x in v[:-1]) + f"{v[-1]:9.3f}")
    d = hits["pooled"][5] / n - hits["strict"][5] / n
    print(f"\nlabel-incompleteness gap at k=5: {d:+.1%} "
          f"({hits['pooled'][5] - hits['strict'][5]} of {n} cases were correct "
          f"but scored as failures)")

    print(f"\n{'category':16s}{'n':>4}" + "".join(f"{m[:9]:>11}" for m in metrics) + "   (R@5)")
    for c in sorted(percat):
        cn = percat_n[c]
        print(f"{c:16s}{cn:4d}" + "".join(f"{percat[c][m][5]/cn:11.1%}" for m in metrics))


if __name__ == "__main__":
    main()
