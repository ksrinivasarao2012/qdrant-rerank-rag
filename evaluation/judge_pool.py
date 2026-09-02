"""
Step 2 of pooled judging: rule on each (query, document) pair. NO LLM -- you are
the annotator. An LLM proposing labels against documents the retriever returned
is how golden_dataset_v2 became circular; this stays manual on purpose.

Nothing in the golden set is modified. Judgments go to
results/pooled_judgments.json and are applied at scoring time by
rescore_pooled.py, so the original strict metric is always still reportable.

Grades (the only question that matters: would a user asking THIS query be well
served by THIS document?):
  2  yes -- it answers the question
  1  partial -- related and useful, but does not answer it
  0  no -- irrelevant

Keys:
  2 / 1 / 0   grade and advance
  g           show the full text of this candidate
  ?           re-show the query and the current gold answer
  s           skip (leave unjudged, revisit later)
  b           go back one
  q           save and quit

Saved after every keystroke -- quit any time and re-run to resume.

Usage:
  python evaluation/judge_pool.py
  python evaluation/judge_pool.py --annotator "K S"
  python evaluation/judge_pool.py --limit 50        # do a batch and stop
"""
import json, sys, argparse, textwrap
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
RES  = ROOT / "evaluation" / "results"
POOL = RES / "judgment_pool.json"
JUDG = RES / "pooled_judgments.json"
W = 96


def wrap(s, indent="    "):
    return textwrap.fill(" ".join(str(s).split()), width=W,
                         initial_indent=indent, subsequent_indent=indent)


def save(j):
    tmp = JUDG.with_suffix(".tmp")
    json.dump(j, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    tmp.replace(JUDG)


def header(case, i, n_case, done, total):
    print("\n" + "=" * W)
    print(f"CASE {i}/{n_case}   [{case['category']}]  {case['query_id']}"
          + (f" / {case['sub_label']}" if case.get("sub_label") else ""))
    print(f"judged {done}/{total} pairs")
    print("=" * W)
    print("QUERY:")
    print(wrap(case["query"]))
    print("\nCURRENT GOLD (what the benchmark expects):")
    for g in case["gold"]:
        print(f"    [{g['answer_id']}] {g['title']}")
        print(wrap(g["snippet"][:300], "        "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", default="", help="your name, recorded in provenance")
    ap.add_argument("--limit", type=int, default=0, help="stop after N judgments this session")
    a = ap.parse_args()

    if not POOL.exists():
        sys.exit("no judgment_pool.json -- run build_judgment_pool.py first")
    pool = json.load(open(POOL, encoding="utf-8"))
    cases = pool["cases"]
    judg = json.load(open(JUDG, encoding="utf-8")) if JUDG.exists() else {}
    judg.setdefault("_meta", {"created": datetime.now().isoformat(timespec="seconds")})
    judg["_meta"]["run"] = pool["run"]
    judg["_meta"]["golden"] = pool["golden"]

    total = sum(len(c["candidates"]) for c in cases)
    done = sum(1 for c in cases for cd in c["candidates"]
               if judg.get(c["query_id"], {}).get(cd["answer_id"], {}).get("grade") is not None)

    print(__doc__.split("Usage:")[0])
    print(f"pool: {len(cases)} cases, {total} pairs, {done} already judged\n")

    flat = [(ci, cd) for ci, c in enumerate(cases) for cd in c["candidates"]]
    idx, session = 0, 0
    while idx < len(flat):
        ci, cd = flat[idx]
        case = cases[ci]
        rec = judg.setdefault(case["query_id"], {})
        if rec.get(cd["answer_id"], {}).get("grade") is not None:
            idx += 1
            continue
        if a.limit and session >= a.limit:
            print(f"\nreached --limit {a.limit}. saved. re-run to continue.")
            break

        header(case, ci + 1, len(cases), done, total)
        print("\n" + "-" * W)
        print(f"CANDIDATE  [{cd['answer_id']}]  (returned by {cd['source']} at rank {cd['rank']}, "
              f"post score {cd['score']})")
        same = cd.get("question_id") and any(g.get("question_id") == cd["question_id"]
                                             for g in case["gold"])
        print(f"TITLE: {cd['title']}" + ("    <-- SAME QUESTION as the gold" if same else ""))
        print()
        print(wrap(cd["snippet"]))
        print("-" * W)

        while True:
            k = input("  [2] answers it  [1] partial  [0] no  |  g full text  ? requery  s skip  b back  q quit > ").strip().lower()
            if k in ("2", "1", "0"):
                rec[cd["answer_id"]] = {
                    "grade": int(k),
                    "title": cd["title"],
                    "source": cd["source"], "rank": cd["rank"],
                    "question_id": cd.get("question_id"),
                    "annotator": a.annotator or "unspecified",
                    "at": datetime.now().isoformat(timespec="seconds"),
                    "method": "human_manual",
                }
                save(judg); done += 1; session += 1; idx += 1
                break
            if k == "g":
                print("\n" + wrap(cd["snippet"]))
                print(f"    {cd['url']}")
                continue
            if k == "?":
                header(case, ci + 1, len(cases), done, total); continue
            if k == "s":
                idx += 1; break
            if k == "b":
                idx = max(0, idx - 1)
                pci, pcd = flat[idx]
                judg.get(cases[pci]["query_id"], {}).pop(pcd["answer_id"], None)
                save(judg); done = max(0, done - 1)
                break
            if k == "q":
                save(judg)
                print(f"\nsaved {done}/{total} judgments to {JUDG}")
                print("next: python evaluation/rescore_pooled.py")
                return
            print("    unrecognised key")

    save(judg)
    print(f"\ndone. {done}/{total} judged. saved to {JUDG}")
    print("next: python evaluation/rescore_pooled.py")


if __name__ == "__main__":
    main()
