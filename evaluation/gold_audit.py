"""
Golden dataset audit. NO API calls -- uses the project's own local embedding
model (BAAI/bge-base-en-v1.5), already cached.

The question: are the labelled gold answers actually the best answers to their
queries, or did label assignment pick a topically-adjacent post?

Method: embed each query, its gold answer(s), and the documents retrieval
actually returned. If the RETRIEVED document is consistently closer to the
query than the LABELLED gold, the benchmark is penalising correct behaviour.

Only 9/79 cases carry human_verified=True, so 89% of these labels have never
been checked by a person. This checks them mechanically.

Usage:  python evaluation/gold_audit.py
"""
import os, sys, json
from pathlib import Path
for _v in ("OMP","MKL","OPENBLAS","NUMEXPR"): os.environ[f"{_v}_NUM_THREADS"]="4"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch; torch.set_num_threads(4)
import numpy as np
from backend.core.embeddings import get_embeddings

CATS = {"multi_hop","negation","niche_topic","multi_turn"}
POOLS = ROOT/"evaluation"/"results"/"_candidate_pools.json"
OUT   = ROOT/"evaluation"/"results"/"gold_audit.json"


def cos(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))


def main():
    cases = [c for c in json.load(open(ROOT/'evaluation/golden_dataset.json', encoding='utf-8'))
             if c.get('category') in CATS and c.get('gold_answer_ids') and c.get('query')]
    pools = json.load(open(POOLS, encoding='utf-8')) if POOLS.exists() else {}
    if not pools:
        print("No _candidate_pools.json -- run reranker_bakeoff.py first."); return 1

    need=set()
    for c in cases: need |= set(str(g) for g in c['gold_answer_ids'])
    posts={}
    with open(ROOT/'data/processed/posts.jsonl', encoding='utf-8') as f:
        for line in f:
            p=json.loads(line); a=str(p.get('answer_id'))
            if a in need:
                posts[a]={'text':(p.get('question_title','')+" "+p.get('answer_text',''))[:2000],
                          'title':p.get('question_title',''),'score':p.get('score'),
                          'acc':p.get('is_accepted')}

    emb = get_embeddings()
    print(f"auditing {len(cases)} cases with the project's own embedding model...\n")

    rows=[]
    for i,c in enumerate(cases,1):
        qid=c['query_id']
        qv = emb.embed_query(c['query'])

        golds=[]
        for g in c['gold_answer_ids']:
            p=posts.get(str(g))
            if not p: continue
            golds.append((str(g), cos(qv, emb.embed_query(p['text'])), p))
        if not golds: continue
        best_gold = max(golds, key=lambda t:t[1])

        top=[]
        for ch in pools.get(qid,[])[:3]:
            top.append((ch['answer_id'], cos(qv, emb.embed_query(ch['text'][:2000]))))
        best_ret = max(top, key=lambda t:t[1]) if top else (None,-1)

        gold_ids={str(g) for g in c['gold_answer_ids']}
        rows.append({
            'qid':qid,'cat':c['category'],'query':c['query'],
            'gold_id':best_gold[0],'gold_sim':best_gold[1],
            'gold_title':best_gold[2]['title'],'gold_score':best_gold[2]['score'],
            'ret_id':best_ret[0],'ret_sim':best_ret[1],
            'ret_is_gold':str(best_ret[0]) in gold_ids,
            'human_verified':bool(c.get('human_verified')),
            'delta':best_ret[1]-best_gold[1],
        })
        if i%20==0: print(f"  {i}/{len(cases)}")

    beats=[r for r in rows if not r['ret_is_gold'] and r['delta']>0]
    close=[r for r in rows if not r['ret_is_gold'] and 0>=r['delta']>-0.03]
    lowgold=[r for r in rows if r['gold_sim']<0.60]

    print("\n"+"="*86)
    print("GOLD LABEL AUDIT")
    print("="*86)
    print(f"cases audited                                    : {len(rows)}")
    print(f"human-verified labels                            : {sum(r['human_verified'] for r in rows)}")
    print(f"\nRETRIEVED doc is SEMANTICALLY CLOSER than the gold: {len(beats)}/{len(rows)} "
          f"({len(beats)/len(rows):.0%})")
    print(f"retrieved within 0.03 of gold (effectively tied)  : {len(close)}/{len(rows)}")
    print(f"gold similarity below 0.60 (weak label)           : {len(lowgold)}/{len(rows)}")
    print("="*86)

    print("\nWorst cases -- retriever found a much closer document than the label:")
    for r in sorted(beats, key=lambda r:-r['delta'])[:12]:
        print(f"\n  {r['qid']:<10} [{r['cat']}]  retrieved is +{r['delta']:.3f} closer"
              f"{'   (human_verified!)' if r['human_verified'] else ''}")
        print(f"     query      : {r['query'][:74]}")
        print(f"     GOLD  {r['gold_sim']:.3f} : {r['gold_title'][:64]!r} (score {r['gold_score']})")
        print(f"     RETR  {r['ret_sim']:.3f} : answer {r['ret_id']}")

    json.dump(rows, open(OUT,'w',encoding='utf-8'), indent=2)
    print(f"\nSaved: {OUT}")
    print("\nINTERPRETATION")
    frac=len(beats)/len(rows)
    if frac>=0.30:
        print(f"  {frac:.0%} of cases: the system retrieved a document the embedding model")
        print("  considers a BETTER match than the label. The benchmark is scoring correct")
        print("  retrieval as failure -- reported scores understate real quality.")
    else:
        print(f"  Only {frac:.0%} of cases show the retriever beating the label.")
        print("  Labels are mostly defensible; the low score is genuine retrieval difficulty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
