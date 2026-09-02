"""
Query expansion A/B. NO LLM, NO API, NO re-indexing.

Problem this targets: retrieval finds the answer under one phrasing and misses it
under another ("Lasso vs Ridge" works, "L1 vs L2 regularization" does not). The
paraphrase slice degrades 30% -> 0% across variant index; the BGE query-instruction
prefix was tested and did not help (-1.3%).

Strategies compared, all deterministic:
  baseline   current single-query hybrid search
  synonym    domain synonym/acronym substitution -> extra phrasings
  prf        pseudo-relevance feedback (Rocchio/RM3 style): search once, mine the
             distinctive terms from the top documents, append them, search again
  combined   RRF fusion over [original, synonym variants, prf variant]

Fusion always includes the original query's ranking, so combined should not lose
to baseline except through rank displacement.

Usage:  python evaluation/query_expansion_test.py
"""
import os, sys, json, re, math
from pathlib import Path
from collections import defaultdict, Counter
for _v in ("OMP","MKL","OPENBLAS","NUMEXPR"): os.environ[f"{_v}_NUM_THREADS"]="4"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch; torch.set_num_threads(4)
from backend.core.vector_store import VectorDBManager

CATS={"paraphrase_group","multi_hop","negation","niche_topic","multi_turn"}
RRF_K=60

# Bidirectional domain equivalences. Hand-built for this corpus (statistics / ML).
SYN=[
 ("l1 regularization","lasso"),("l2 regularization","ridge"),("l1","lasso"),("l2","ridge"),
 ("mle","maximum likelihood estimation"),("ols","ordinary least squares"),
 ("pca","principal component analysis"),("fa","factor analysis"),
 ("rf","random forest"),("gbm","gradient boosting"),("nn","neural network"),
 ("cv","cross validation"),("ci","confidence interval"),
 ("anova","analysis of variance"),("glm","generalized linear model"),
 ("glmm","generalized linear mixed model"),("iv","instrumental variable"),
 ("kde","kernel density estimation"),("em","expectation maximization"),
 ("mcmc","markov chain monte carlo"),("ess","effective sample size"),
 ("roc","receiver operating characteristic"),("auc","area under the curve"),
 ("rnn","recurrent neural network"),("lstm","long short-term memory"),
 ("tsne","t-sne"),("dimensionality reduction","dimension reduction"),
 ("overfit","overfitting"),("heteroscedasticity","heteroskedasticity"),
 ("stepwise","stepwise regression"),("arima","autoregressive integrated moving average"),
]
STOP=set("the a an of to in is are and or for with on at by as be this that it how what why when "
         "which from can do does i you we they not without other than excluding using use "
         "there its about into over under between among more most less least very should".split())
_tok=re.compile(r"[a-z0-9\-]+")
def toks(s): return [w for w in _tok.findall(s.lower()) if w not in STOP and len(w)>2]


def synonym_variants(q):
    ql=q.lower(); out=[]
    for a,b in SYN:
        if re.search(rf"\b{re.escape(a)}\b", ql): out.append(re.sub(rf"\b{re.escape(a)}\b", b, ql))
        if re.search(rf"\b{re.escape(b)}\b", ql): out.append(re.sub(rf"\b{re.escape(b)}\b", a, ql))
    seen=set(); uniq=[]
    for v in out:
        if v!=ql and v not in seen: seen.add(v); uniq.append(v)
    return uniq[:2]


def prf_query(db, q, n_feedback=5, n_terms=6):
    """Search once, mine distinctive terms from the top docs, append them."""
    try: top=db.search_hybrid(query=q, n_results=n_feedback)
    except Exception: return None
    if not top: return None
    qt=set(toks(q)); df=Counter(); tf=Counter()
    for d in top:
        t=set(toks(d.get("text","")[:1200]))
        df.update(t)
        tf.update(toks(d.get("text","")[:1200]))
    # prefer terms common across the feedback set but not already in the query
    cand=[(w, df[w]*math.log(1+tf[w])) for w in df if w not in qt and df[w]>=2]
    cand.sort(key=lambda x:-x[1])
    add=[w for w,_ in cand[:n_terms]]
    return (q+" "+" ".join(add)) if add else None


def rrf(lists, k=RRF_K):
    sc=defaultdict(float); seen={}
    for L in lists:
        for rank,d in enumerate(L,1):
            i=str(d["metadata"].get("answer_id"))
            sc[i]+=1.0/(k+rank); seen.setdefault(i,d)
    return [seen[i] for i in sorted(sc,key=lambda x:-sc[x])]


def gold_of(c):
    s={str(g) for g in c.get('gold_answer_ids',[])}
    if c.get('graded_relevance'): s|={str(k) for k in c['graded_relevance']}
    return s


def main():
    cases=[c for c in json.load(open(ROOT/'evaluation/golden_dataset.json',encoding='utf-8'))
           if c.get('category') in CATS and c.get('gold_answer_ids')]
    items=[]
    for c in cases:
        g=gold_of(c)
        if c.get('variants'):
            for i,v in enumerate(c['variants']):
                q=v.get('query') if isinstance(v,dict) else v
                if q: items.append((c['query_id'],f"variant_{i}",c['category'],q,g))
        elif c.get('query'):
            items.append((c['query_id'],"-",c['category'],c['query'],g))
    print(f"{len(items)} query instances, {len(cases)} cases\n")

    db=VectorDBManager(); db.collection
    strategies=["baseline","synonym","prf","combined"]
    stats={s:{"h5":0,"h10":0,"cat":defaultdict(lambda:[0,0]),"var":defaultdict(lambda:[0,0])}
           for s in strategies}
    expanded=0

    for i,(qid,sl,cat,q,gold) in enumerate(items,1):
        base=db.search_hybrid(query=q, n_results=10)
        syns=synonym_variants(q)
        syn_lists=[db.search_hybrid(query=v, n_results=10) for v in syns]
        pq=prf_query(db,q)
        prf_list=db.search_hybrid(query=pq, n_results=10) if pq else []
        if syns or pq: expanded+=1

        got={
          "baseline": base,
          "synonym":  rrf([base]+syn_lists) if syn_lists else base,
          "prf":      rrf([base,prf_list]) if prf_list else base,
          "combined": rrf([base]+syn_lists+([prf_list] if prf_list else [])),
        }
        for s in strategies:
            ids=[str(d["metadata"].get("answer_id")) for d in got[s]][:10]
            h5=bool(gold&set(ids[:5])); h10=bool(gold&set(ids))
            stats[s]["h5"]+=h5; stats[s]["h10"]+=h10
            stats[s]["cat"][cat][0]+=h5; stats[s]["cat"][cat][1]+=1
            if sl!="-": stats[s]["var"][sl][0]+=h5; stats[s]["var"][sl][1]+=1
        if i%25==0: print(f"  {i}/{len(items)}")

    n=len(items)
    print(f"\n(expansion produced extra phrasings for {expanded}/{n} queries)\n")
    print("="*72); print("QUERY EXPANSION RESULTS"); print("="*72)
    print(f"{'strategy':<14}{'R@5':>9}{'R@10':>9}{'vs baseline':>14}")
    b5=stats['baseline']['h5']/n
    for s in strategies:
        r5=stats[s]['h5']/n; r10=stats[s]['h10']/n
        print(f"{s:<14}{r5:>8.1%}{r10:>9.1%}{(r5-b5):>+13.1%}")

    print("\nby category (R@5):")
    cats=sorted(stats['baseline']['cat'])
    print("  "+f"{'category':<20}"+"".join(f"{s:>11}" for s in strategies))
    for c in cats:
        row=f"  {c:<20}"
        for s in strategies:
            a,b=stats[s]['cat'][c]; row+=f"{a/b:>11.1%}"
        print(row)

    print("\nparaphrase robustness by variant (R@5):")
    vs=sorted(stats['baseline']['var'])
    print("  "+f"{'variant':<12}"+"".join(f"{s:>11}" for s in strategies))
    for v in vs:
        row=f"  {v:<12}"
        for s in strategies:
            a,b=stats[s]['var'][v]; row+=f"{a/b:>11.1%}"
        print(row)

    out=ROOT/'evaluation'/'results'/'query_expansion_test.json'
    json.dump({s:{"r5":stats[s]['h5']/n,"r10":stats[s]['h10']/n} for s in strategies},
              open(out,'w',encoding='utf-8'), indent=2)
    print(f"\nSaved: {out}")
    best=max(strategies,key=lambda s:stats[s]['h5'])
    d=stats[best]['h5']/n-b5
    print("\nVERDICT")
    print(f"  best: {best}  ({stats[best]['h5']/n:.1%} R@5, {d:+.1%} vs baseline)")
    if d>=0.05: print("  Substantial. Wire this into the live retrieval path.")
    elif d>0:   print("  Real but modest. Worth shipping; keep looking for bigger wins.")
    else:       print("  No gain. Vocabulary expansion is not the bottleneck either.")
    return 0

if __name__=="__main__": sys.exit(main())
