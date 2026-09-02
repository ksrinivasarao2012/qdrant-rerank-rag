"""
BGE query-instruction A/B test. NO LLM, NO re-indexing.

backend/core/embeddings.py builds a plain HuggingFaceEmbeddings, so embed_query()
applies no instruction prefix. BGE is trained for asymmetric retrieval: the QUERY
side is meant to carry "Represent this sentence for searching relevant passages: "
while passages carry nothing. Because only the query side changes, this can be
tested (and shipped) without touching the index.

Measured on the paraphrase_group slice, where R@5 collapses 40% -> 10% as phrasing
drifts, plus the four hard categories.

Usage:  python evaluation/query_instruction_test.py
"""
import os, sys, json
from pathlib import Path
for _v in ("OMP","MKL","OPENBLAS","NUMEXPR"): os.environ[f"{_v}_NUM_THREADS"]="4"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch; torch.set_num_threads(4)
from collections import defaultdict
from backend.core.vector_store import VectorDBManager

BGE_Q = "Represent this sentence for searching relevant passages: "
CATS  = {"paraphrase_group","multi_hop","negation","niche_topic","multi_turn"}


def gold_of(c):
    s={str(g) for g in c.get('gold_answer_ids',[])}
    if c.get('graded_relevance'): s|={str(k) for k in c['graded_relevance']}
    return s


def main():
    cases=[c for c in json.load(open(ROOT/'evaluation/golden_dataset.json',encoding='utf-8'))
           if c.get('category') in CATS and c.get('gold_answer_ids')]

    # expand paraphrase groups into their variants, like the retriever eval does
    items=[]
    for c in cases:
        gold=gold_of(c)
        if c.get('variants'):
            for i,v in enumerate(c['variants']):
                q=v.get('query') if isinstance(v,dict) else v
                if q: items.append((c['query_id'],f"variant_{i}",c['category'],q,gold))
        elif c.get('query'):
            items.append((c['query_id'],"-",c['category'],c['query'],gold))
    print(f"{len(items)} query instances across {len(cases)} cases\n")

    db=VectorDBManager(); db.collection

    res={}
    for label,prefix in (("WITHOUT instruction (current)",""),("WITH bge instruction",BGE_Q)):
        hits5=hits10=0; percat=defaultdict(lambda:[0,0]); pervar=defaultdict(lambda:[0,0])
        for i,(qid,sl,cat,q,gold) in enumerate(items,1):
            out=db.search_hybrid(query=prefix+q, n_results=10)
            ids=[str(r["metadata"].get("answer_id")) for r in out]
            h5=bool(gold & set(ids[:5])); h10=bool(gold & set(ids))
            hits5+=h5; hits10+=h10
            percat[cat][0]+=h5; percat[cat][1]+=1
            if sl!="-": pervar[sl][0]+=h5; pervar[sl][1]+=1
            if i%40==0: print(f"   {label}: {i}/{len(items)}")
        n=len(items)
        res[label]=dict(r5=hits5/n, r10=hits10/n,
                        cat={k:(v[0]/v[1],v[1]) for k,v in percat.items()},
                        var={k:(v[0]/v[1],v[1]) for k,v in pervar.items()})
        print(f"  -> {label}: R@5 {hits5/n:.1%}  R@10 {hits10/n:.1%}\n")

    a,b=list(res)
    print("="*78); print("BGE QUERY-INSTRUCTION A/B"); print("="*78)
    print(f"{'':<34}{'R@5':>10}{'R@10':>10}")
    for k in res: print(f"{k:<34}{res[k]['r5']:>9.1%}{res[k]['r10']:>10.1%}")
    print(f"{'DELTA':<34}{res[b]['r5']-res[a]['r5']:>+9.1%}{res[b]['r10']-res[a]['r10']:>+10.1%}")

    print("\nby category (R@5):")
    print(f"  {'category':<20}{'n':>5}{'without':>10}{'with':>10}{'delta':>9}")
    for c in sorted(res[a]['cat']):
        n=res[a]['cat'][c][1]
        x,y=res[a]['cat'][c][0],res[b]['cat'][c][0]
        print(f"  {c:<20}{n:>5}{x:>10.1%}{y:>10.1%}{y-x:>+9.1%}")

    if res[a]['var']:
        print("\nparaphrase robustness by variant (R@5):")
        print(f"  {'variant':<12}{'n':>5}{'without':>10}{'with':>10}{'delta':>9}")
        for v in sorted(res[a]['var']):
            n=res[a]['var'][v][1]
            x,y=res[a]['var'][v][0],res[b]['var'][v][0]
            print(f"  {v:<12}{n:>5}{x:>10.1%}{y:>10.1%}{y-x:>+9.1%}")

    out=ROOT/'evaluation'/'results'/'query_instruction_test.json'
    json.dump(res, open(out,'w',encoding='utf-8'), indent=2, default=str)
    print(f"\nSaved: {out}")
    d=res[b]['r5']-res[a]['r5']
    print("\nVERDICT")
    if d>=0.03: print(f"  +{d:.1%} R@5 for a query-side prefix and NO re-indexing. Ship it.")
    elif d>0:   print(f"  +{d:.1%} - real but small. Worth shipping, not the main bottleneck.")
    else:       print(f"  {d:+.1%} - no gain. The instruction is not the issue; look elsewhere.")
    return 0

if __name__=="__main__": sys.exit(main())
