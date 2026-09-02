"""
Interactive golden-dataset reviewer. NO LLM -- you are the annotator.

Shows each flagged case with its current gold label(s) and the documents the
retriever actually surfaces, then records YOUR decision. Nothing is changed in
golden_dataset.json; decisions are written to results/review_decisions.json and
applied later by apply_review.py, so the process is auditable and reversible.

Decisions are saved after every case -- quit any time with 'q' and re-run to
resume where you stopped.

Keys:
  ENTER / k   keep the current gold as-is
  <number>    REPLACE gold with that candidate  (e.g.  3 )
  +<number>   ADD that candidate to the gold set (e.g. +3 )
  u           mark UNANSWERABLE (no good answer exists in the corpus)
  d           DROP this case from the benchmark
  s           skip (undecided, revisit later)
  m           show more candidates / longer text
  q           save and quit

Usage:
  python evaluation/dataset_review.py            # flagged cases only
  python evaluation/dataset_review.py --all      # every case
"""
import json, sys, argparse, textwrap
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
ROOT = Path(__file__).resolve().parents[1]
RES  = ROOT/"evaluation"/"results"
DEC  = RES/"review_decisions.json"
W    = 100


def wrap(s, indent="      "):
    return textwrap.fill(" ".join(str(s).split()), width=W,
                         initial_indent=indent, subsequent_indent=indent)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="review every case, not just flagged")
    ap.add_argument("--reviewer", default="", help="your name, recorded in provenance")
    a=ap.parse_args()

    triage={r['query_id']:r for r in json.load(open(RES/'dataset_triage.json',encoding='utf-8'))}
    cases={c['query_id']:c for c in json.load(open(ROOT/'evaluation/golden_dataset.json',encoding='utf-8'))}
    pools=json.load(open(RES/'_candidate_pools.json',encoding='utf-8'))

    need=set()
    for t in triage.values(): need|=set(t['gold_ids'])
    posts={}
    with open(ROOT/'data/processed/posts.jsonl',encoding='utf-8') as f:
        for line in f:
            p=json.loads(line); i=str(p.get('answer_id'))
            if i in need:
                posts[i]={'title':p.get('question_title',''),'text':p.get('answer_text',''),
                          'score':p.get('score'),'acc':p.get('is_accepted')}

    decisions = json.load(open(DEC,encoding='utf-8')) if DEC.exists() else {}
    reviewer = a.reviewer or input("Your name (recorded as reviewed_by): ").strip() or "unspecified"

    todo=[q for q,t in sorted(triage.items(), key=lambda kv:-kv[1]['n_serious'])
          if (a.all or t['n_serious']>=1) and q not in decisions]

    print(f"\n{len(todo)} case(s) to review. {len(decisions)} already decided.")
    print("ENTER=keep  N=replace  +N=add  u=unanswerable  d=drop  s=skip  m=more  q=quit\n")

    for n,qid in enumerate(todo,1):
        t=triage[qid]; c=cases[qid]; more=False
        while True:
            print("="*W)
            print(f"[{n}/{len(todo)}]  {qid}   category={t['category']}   flags={','.join(t['flags'])}")
            print("="*W)
            print(f"\nQUERY:\n{wrap(c['query'])}")
            if c.get('chat_history'):
                print("\n  (multi-turn; preceding conversation:)")
                for m_ in c['chat_history'][-2:]:
                    r_=m_.get('role','?') if isinstance(m_,dict) else '?'
                    ct=m_.get('content','') if isinstance(m_,dict) else str(m_)
                    print(wrap(f"{r_}: {ct}", "        "))

            print("\nCURRENT GOLD:")
            for g in t['gold_ids']:
                p=posts.get(g)
                if not p: print(f"   {g}: <not in corpus>"); continue
                print(f"   [{g}] score={p['score']} accepted={p['acc']} | {p['title'][:70]!r}")
                print(wrap(p['text'][:(700 if more else 260)]+"..."))

            print("\nWHAT RETRIEVAL RETURNS:")
            seen=[]; cands=[]
            for ch in pools.get(qid,[]):
                if ch['answer_id'] in seen: continue
                seen.append(ch['answer_id']); cands.append(ch)
                if len(cands)>=(10 if more else 5): break
            for i,ch in enumerate(cands,1):
                tag=" <-- ALREADY GOLD" if ch['answer_id'] in t['gold_ids'] else ""
                print(f"   {i}. [{ch['answer_id']}]{tag}")
                print(wrap(ch['text'][:(500 if more else 230)]+"..."))

            ans=input("\n  decision > ").strip().lower()
            if ans=="m": more=True; print(); continue
            break

        rec={'query_id':qid,'reviewed_by':reviewer,'reviewed_at':datetime.now().isoformat(timespec='seconds'),
             'original_gold':t['gold_ids'],'flags':t['flags']}
        if ans in ("","k"):
            rec.update(action='KEEP', gold_answer_ids=t['gold_ids'], answerable=True)
        elif ans=="u":
            rec.update(action='UNANSWERABLE', gold_answer_ids=[], answerable=False)
        elif ans=="d":
            rec.update(action='DROP', gold_answer_ids=[], answerable=None)
        elif ans=="s":
            print("  skipped\n"); continue
        elif ans=="q":
            break
        elif ans.startswith("+") and ans[1:].isdigit() and 1<=int(ans[1:])<=len(cands):
            new=cands[int(ans[1:])-1]['answer_id']
            rec.update(action='ADD', gold_answer_ids=sorted(set(t['gold_ids'])|{new}), answerable=True)
        elif ans.isdigit() and 1<=int(ans)<=len(cands):
            new=cands[int(ans)-1]['answer_id']
            rec.update(action='REPLACE', gold_answer_ids=[new], answerable=True)
        else:
            print("  unrecognised -> skipped\n"); continue

        note=input("  note (optional, why): ").strip()
        if note: rec['review_note']=note
        decisions[qid]=rec
        json.dump(decisions, open(DEC,'w',encoding='utf-8'), indent=2)
        print(f"  recorded: {rec['action']}   ({len(decisions)} saved)\n")

    from collections import Counter
    print("\n"+"="*W)
    print(f"decisions so far: {len(decisions)}  -> {dict(Counter(d['action'] for d in decisions.values()))}")
    print(f"saved to {DEC}")
    print("NEXT: python evaluation/apply_review.py")
    return 0


if __name__=="__main__": sys.exit(main())
