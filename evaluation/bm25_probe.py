"""
BM25 hypothesis probe. No re-indexing, no API calls, no GPU.

The production sparse channel is CRC32 feature hashing (2^18 buckets,
term-frequency only, IDF applied server-side) -- a BM25 approximation without
length normalisation or TF saturation, and with hash collisions. This builds a
REAL BM25 over the same corpus in memory and asks one question:

    does proper BM25 retrieve the gold documents that the current hybrid
    pipeline misses entirely?

If yes, the sparse channel is a genuine architectural weakness and re-indexing
with BM25 is justified. If no, the gap is elsewhere and re-indexing would be
wasted effort. Diagnostic only -- it changes nothing in the pipeline.

Usage:  python evaluation/bm25_probe.py
"""
import json, math, re, sys, time
from pathlib import Path
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
POSTS = ROOT / "data" / "processed" / "posts.jsonl"
GOLDEN = ROOT / "evaluation" / "golden_dataset.json"
DIAG = ROOT / "evaluation" / "results" / "retrieval_diagnostic.json"
CATS = {"multi_hop", "negation", "niche_topic", "multi_turn"}
MAX_CHARS = 4000
K1, B = 1.5, 0.75

_tok = re.compile(r"[a-z0-9]+")
STOP = set("the a an of to in is are and or for with on at by as be this that it "
           "how what why when which from can do does i you we they not".split())


def toks(s):
    return [t for t in _tok.findall(s.lower()) if t not in STOP and len(t) > 1]


def gold_of(c):
    s = set(str(g) for g in c.get("gold_answer_ids", []))
    if c.get("graded_relevance"): s.update(str(k) for k in c["graded_relevance"])
    if c.get("candidate_gold_ids"): s.update(str(k) for k in c["candidate_gold_ids"])
    return s


def main():
    cases = [c for c in json.load(open(GOLDEN, encoding="utf-8"))
             if c.get("gold_answer_ids") and c.get("query") and c.get("category") in CATS]

    print("building BM25 index over posts.jsonl (one pass, in memory)...")
    t0 = time.time()
    doc_ids, doc_len, tf_list = [], [], []
    df = Counter()
    titles = {}
    with open(POSTS, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            aid = str(p.get("answer_id"))
            text = (p.get("question_title", "") + " " + p.get("answer_text", ""))[:MAX_CHARS]
            tk = toks(text)
            if not tk:
                continue
            tf = Counter(tk)
            doc_ids.append(aid); doc_len.append(len(tk)); tf_list.append(tf)
            titles[aid] = p.get("question_title", "")
            df.update(tf.keys())

    N = len(doc_ids)
    avgdl = sum(doc_len) / N
    print(f"  {N:,} documents, avg length {avgdl:.0f} tokens, "
          f"{len(df):,} vocabulary terms  ({time.time()-t0:.0f}s)")

    postings = defaultdict(list)
    for i, tf in enumerate(tf_list):
        for term, c in tf.items():
            postings[term].append((i, c))
    idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    print(f"  inverted index built ({time.time()-t0:.0f}s)\n")

    def search(q, k=10):
        scores = defaultdict(float)
        for term in set(toks(q)):
            if term not in postings:
                continue
            w = idf[term]
            for i, c in postings[term]:
                dl = doc_len[i]
                scores[i] += w * (c * (K1 + 1)) / (c + K1 * (1 - B + B * dl / avgdl))
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [doc_ids[i] for i, _ in top]

    never = set()
    if DIAG.exists():
        never = {r["qid"] for r in json.load(open(DIAG, encoding="utf-8"))["rows"]
                 if not r["pool_hit"]}

    r5 = r10 = 0
    rescued = []
    for c in cases:
        gold = gold_of(c)
        got = search(c["query"], k=10)
        h5, h10 = bool(gold & set(got[:5])), bool(gold & set(got))
        r5 += h5; r10 += h10
        if c["query_id"] in never and h10:
            rank = next(i for i, a in enumerate(got, 1) if a in gold)
            rescued.append((c["query_id"], rank, c["query"]))

    n = len(cases)
    print("=" * 78)
    print("BM25-ONLY RESULTS (no dense, no reranker, no query rewriting)")
    print("=" * 78)
    print(f"  recall@5 : {r5}/{n} = {r5/n:.1%}")
    print(f"  recall@10: {r10}/{n} = {r10/n:.1%}")
    print()
    print("  for reference, current production pipeline (dense+sparse+reranker):")
    print("    recall@5 = 36.7%   recall@10 = 53.2%")
    print("=" * 78)

    print(f"\nCases the CURRENT pipeline never retrieved at all ({len(never)} of them)")
    print(f"that plain BM25 finds in its top-10: {len(rescued)}\n")
    for qid, rank, q in rescued:
        print(f"  {qid:<10} BM25 rank {rank:<3} | {q[:66]}")

    print("\nINTERPRETATION")
    if len(rescued) >= 4:
        print("  BM25 recovers documents the production sparse channel misses entirely.")
        print("  -> the CRC32 hashed sparse vectors are a real architectural weakness;")
        print("     re-indexing with proper BM25 is justified.")
    else:
        print("  BM25 rescues few or none of the missed documents.")
        print("  -> the sparse channel is NOT the main gap; do not spend time re-indexing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
