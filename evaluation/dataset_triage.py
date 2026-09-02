"""
Golden dataset triage. NO LLM, NO API. Ranks cases by deterministic red flags
so review effort goes to the labels most likely to be wrong.

Flags (each is evidence the label may be mis-assigned, not proof):
  WEAK_POST     gold answer has StackExchange score <= 2
  NO_OVERLAP    gold's question title shares no content word with the query
  UNREACHABLE   gold never appeared in an 80-100 candidate retrieval pool
  STUB          gold answer under 400 chars (comment-like, not an answer)
  SHARED_GOLD   the same gold post is assigned to another query too
  UNVERIFIED    human_verified is not True

Usage:  python evaluation/dataset_triage.py
"""
import json, re, sys
from pathlib import Path
from collections import Counter, defaultdict

if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
ROOT = Path(__file__).resolve().parents[1]
CATS = {"multi_hop","negation","niche_topic","multi_turn"}
OUT  = ROOT/"evaluation"/"results"/"dataset_triage.json"

STOP=set("the a an of to in is are and or for with on at by as be this that it how what why when "
         "which from can do does i you we they not without other than excluding using use their "
         "there its it's about into over under between among more most less least very".split())
def tok(s): return {w for w in re.findall(r"[a-z0-9\-]+", (s or "").lower()) if w not in STOP and len(w)>2}

def main():
    cases=[c for c in json.load(open(ROOT/'evaluation/golden_dataset.json',encoding='utf-8'))
           if c.get('category') in CATS and c.get('gold_answer_ids') and c.get('query')]

    need=set()
    for c in cases: need |= {str(g) for g in c['gold_answer_ids']}
    posts={}
    with open(ROOT/'data/processed/posts.jsonl',encoding='utf-8') as f:
        for line in f:
            p=json.loads(line); a=str(p.get('answer_id'))
            if a in need:
                posts[a]={'title':p.get('question_title',''),'text':p.get('answer_text',''),
                          'score':p.get('score'),'acc':p.get('is_accepted')}

    dpath=ROOT/'evaluation/results/retrieval_diagnostic.json'
    unreachable=set()
    if dpath.exists():
        unreachable={r['qid'] for r in json.load(open(dpath,encoding='utf-8'))['rows'] if not r['pool_hit']}

    owner=defaultdict(list)
    for c in cases:
        for g in c['gold_answer_ids']: owner[str(g)].append(c['query_id'])
    shared={g for g,v in owner.items() if len(v)>1}

    rows=[]
    for c in cases:
        flags=[]
        gids=[str(g) for g in c['gold_answer_ids']]
        gp=[posts[g] for g in gids if g in posts]
        if not gp: continue

        if all((p['score'] or 0)<=2 for p in gp):            flags.append("WEAK_POST")
        q=tok(c['query'])
        if not any(q & tok(p['title']) for p in gp):          flags.append("NO_OVERLAP")
        if c['query_id'] in unreachable:                      flags.append("UNREACHABLE")
        if all(len(p['text'])<400 for p in gp):               flags.append("STUB")
        if any(g in shared for g in gids):                    flags.append("SHARED_GOLD")
        if not c.get('human_verified'):                       flags.append("UNVERIFIED")

        rows.append({'query_id':c['query_id'],'category':c['category'],'query':c['query'],
                     'gold_ids':gids,'gold_titles':[p['title'] for p in gp],
                     'gold_scores':[p['score'] for p in gp],
                     'flags':flags,'n_serious':len([f for f in flags if f!='UNVERIFIED'])})

    rows.sort(key=lambda r:(-r['n_serious'], r['query_id']))

    print("="*94); print("GOLDEN DATASET TRIAGE"); print("="*94)
    print(f"cases: {len(rows)}\n")
    fc=Counter(f for r in rows for f in r['flags'])
    for f,n in fc.most_common(): print(f"  {f:<14} {n:>3} cases")

    tiers=Counter(r['n_serious'] for r in rows)
    print(f"\nserious-flag distribution: {dict(sorted(tiers.items(), reverse=True))}")
    prio=[r for r in rows if r['n_serious']>=1]
    print(f"\n=> REVIEW PRIORITY: {len(prio)} cases have >=1 serious flag "
          f"({len(rows)-len(prio)} look clean)\n")

    print("="*94); print("PRIORITISED WORKLIST"); print("="*94)
    for r in rows:
        if r['n_serious']==0: continue
        print(f"\n{r['query_id']:<10} [{r['category']:<11}] {','.join(f for f in r['flags'] if f!='UNVERIFIED')}")
        print(f"   Q    : {r['query'][:82]}")
        for t,s in zip(r['gold_titles'], r['gold_scores']):
            print(f"   gold : {str(t)[:70]!r} (score {s})")

    OUT.parent.mkdir(parents=True,exist_ok=True)
    json.dump(rows, open(OUT,'w',encoding='utf-8'), indent=2)
    print(f"\nSaved: {OUT}")
    print(f"\nNEXT: review the {len(prio)} flagged cases with evaluation/dataset_review.py")
    return 0

if __name__=="__main__": sys.exit(main())
