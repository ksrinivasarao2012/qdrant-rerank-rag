"""
Index completeness check. No LLM calls, ~30 seconds.

neg_05 asks "How to select model features without stepwise regression?" and the
gold document sits under the question "What are modern, easily used alternatives
to stepwise regression?" -- a near-exact lexical AND semantic match that hybrid
search should rank #1. It was absent from an 80-candidate pool entirely.

Either (a) those documents are not in the Qdrant index at all, or (b) they are
indexed but retrieval cannot find them. Those are completely different bugs.
This decides which.
"""
import os, sys, json
from pathlib import Path
for _v in ("OMP","MKL","OPENBLAS","NUMEXPR"): os.environ[f"{_v}_NUM_THREADS"]="1"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch; torch.set_num_threads(1)
from qdrant_client.http import models
from backend.core.vector_store import VectorDBManager

cases = {c['query_id']: c for c in json.load(open(ROOT/'evaluation/golden_dataset.json', encoding='utf-8'))}
diag  = json.load(open(ROOT/'evaluation/results/retrieval_diagnostic.json', encoding='utf-8'))['rows']
never = [r['qid'] for r in diag if not r['pool_hit']]

want = {}
for q in never:
    for g in cases[q].get('gold_answer_ids', []):
        want.setdefault(str(g), q)

db = VectorDBManager(); db.collection
print(f"collection: {db.collection_name}")
try:
    print(f"points in collection: {db.client.count(db.collection_name, exact=True).count:,}\n")
except Exception as e:
    print(f"(count failed: {e})\n")

print(f"Checking {len(want)} gold answer_ids from {len(never)} never-retrieved cases:\n")
present, absent = [], []
for aid, qid in sorted(want.items()):
    try:
        hits, _ = db.client.scroll(
            collection_name=db.collection_name,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="answer_id", match=models.MatchValue(value=int(aid)))]),
            limit=3, with_payload=True, with_vectors=False)
        if not hits:
            hits, _ = db.client.scroll(
                collection_name=db.collection_name,
                scroll_filter=models.Filter(must=[models.FieldCondition(
                    key="answer_id", match=models.MatchValue(value=str(aid)))]),
                limit=3, with_payload=True, with_vectors=False)
    except Exception as e:
        print(f"  {aid}: scroll error {e}"); continue

    if hits:
        present.append(aid)
        t = (hits[0].payload or {}).get("question_title", "?")
        print(f"  INDEXED  {aid:<9} ({qid:<9}) {len(hits)} chunk(s) | {t[:62]!r}")
    else:
        absent.append(aid)
        print(f"  MISSING  {aid:<9} ({qid:<9}) <-- not in the index at all")

print(f"\n  indexed: {len(present)}   MISSING: {len(absent)}")

# Direct retrieval sanity test on the clearest example
print("\n" + "="*80)
print("DIRECT TEST -- can hybrid search find a near-exact title match?")
for q in ["What are modern, easily used alternatives to stepwise regression?",
          "alternatives to stepwise regression",
          "How to select model features without stepwise regression?"]:
    res = db.search_hybrid(query=q, n_results=10)
    ids = [str(r["metadata"].get("answer_id")) for r in res]
    mark = "FOUND at rank %d" % (ids.index("13698")+1) if "13698" in ids else "NOT in top-10"
    print(f"\n  query: {q!r}\n    -> {mark}")
    for i, r in enumerate(res[:3], 1):
        print(f"       {i}. {r['metadata'].get('answer_id')} | {str(r['metadata'].get('question_title'))[:66]!r}")
