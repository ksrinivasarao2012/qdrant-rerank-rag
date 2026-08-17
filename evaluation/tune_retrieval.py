"""
Fast, category-scoped retrieval experiments.

eval_retriever.py runs all 314 query instances and takes ~43 minutes, which is
far too slow to iterate on one weak category. This runs a single category
(20-30 cases, ~2-6 minutes) across several strategies and prints them side by
side, so a change can be accepted or rejected before paying for a full run.

Nothing here replaces eval_retriever.py -- once a strategy wins on its category,
re-run the full eval to confirm it did not regress the other 290 instances.

Strategies
----------
  baseline        hybrid search -> cross-encoder rerank      (current production)
  norerank        hybrid search, raw fusion order
  rerank_display  rerank against `display_text` (plain chunk) instead of `text`
                  (title + overlap prefix). Tests whether the repeated title
                  prefix is noise for the cross-encoder.
  rerank_blend    RRF-fuse the rerank ordering with the retrieval ordering,
                  instead of trusting the cross-encoder outright. Helps when the
                  reranker is out-of-domain (MiniLM is trained on MS MARCO web
                  search, not statistics prose).
  history_concat  multi_turn only: prepend the last conversation turns to the
                  query. A no-model alternative to LLM query rewriting, for
                  follow-ups whose topic lives in a pronoun ("prevent it").
  decompose       multi_hop only: split a comparison query ("compare X and Y")
                  into two sub-queries, retrieve each, RRF-merge, then rerank
                  with the ORIGINAL query.

Examples
--------
    python evaluation/tune_retrieval.py --category niche_topic \
        --strategies baseline,norerank,rerank_display,rerank_blend
    python evaluation/tune_retrieval.py --category multi_turn \
        --strategies baseline,history_concat
    python evaluation/tune_retrieval.py --category multi_hop \
        --strategies baseline,decompose
    python evaluation/tune_retrieval.py --category niche_topic \
        --strategies baseline --reranker-model BAAI/bge-reranker-base
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

for _v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{_v}_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.vector_store import VectorDBManager
from backend.core.reranker import ReRanker

GOLDEN_JSON_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
POSTS_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "posts.jsonl"

RRF_K = 60  # same constant Qdrant's RRF uses


def parse_args():
    p = argparse.ArgumentParser(description="Category-scoped retrieval experiments.")
    p.add_argument("--category", default="niche_topic")
    p.add_argument("--strategies", default="baseline,norerank,rerank_display,rerank_blend")
    p.add_argument("--pool", type=int, default=100)
    p.add_argument("--pools", default="",
                   help="Comma list of pool sizes to sweep, e.g. 15,30,50,100. "
                        "Overrides --pool. Answers the production question: does "
                        "fetching more candidates actually improve r@3, the only "
                        "depth the generator ever sees?")
    p.add_argument("--ks", default="3,10,50")
    p.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--limit", type=int, default=0, help="Cap cases (0 = all).")
    return p.parse_args()


# --------------------------------------------------------------------------- data

def load_cases(category, limit=0):
    with open(GOLDEN_JSON_PATH, "r", encoding="utf-8") as f:
        cases = [c for c in json.load(f)
                 if c["category"] == category and c.get("gold_answer_ids")]
    return cases[:limit] if limit else cases


def gold_question_ids(cases):
    needed = {int(a) for c in cases for a in c["gold_answer_ids"]}
    lookup, remaining = {}, set(needed)
    if not POSTS_JSONL_PATH.exists():
        return lookup
    with open(POSTS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            post = json.loads(line)
            if post["answer_id"] in remaining:
                lookup[str(post["answer_id"])] = str(post["question_id"])
                remaining.discard(post["answer_id"])
    return lookup


def queries_for_case(case):
    """(query, sub_label) pairs -- paraphrase groups expand to their variants."""
    if case.get("variants"):
        return [(v, f"variant_{i}") for i, v in enumerate(case["variants"])]
    return [(case.get("query", ""), None)] if case.get("query") else []


# ------------------------------------------------------------------- query shaping

def history_concat(case, query):
    """Prepend the last two conversation turns so a pronoun-only follow-up
    carries its own topic. The topic word usually sits in the assistant's
    previous reply ('Overfitting happens when...'), which is exactly what the
    bare query is missing."""
    history = case.get("chat_history") or []
    if not history:
        return query
    tail = " ".join((m.get("content") or "") for m in history[-2:])
    return f"{tail} {query}".strip()


_SPLIT_PATTERNS = [
    r"(?:difference|differences)\s+between\s+(.+?)\s+and\s+(.+?)(?:[?.,]|$)",
    r"\bcompare\s+(.+?)\s+(?:and|with|to|versus|vs\.?)\s+(.+?)(?:[?.,]|$)",
    r"^(.+?)\s+(?:versus|vs\.?)\s+(.+?)(?:[?.,]|$)",
]


def decompose(query):
    """Split a comparison query into its two sides. Pattern-based on purpose:
    a comparison question is syntactically marked ('compare X and Y', 'X vs Y'),
    so this needs no model. Returns [] when nothing matches."""
    for pat in _SPLIT_PATTERNS:
        m = re.search(pat, query, flags=re.IGNORECASE)
        if m:
            a, b = m.group(1).strip(), m.group(2).strip()
            if len(a) > 2 and len(b) > 2:
                return [a, b]
    return []


def extract_negation_terms(query: str) -> list[str]:
    # Normalise whitespace and punctuation
    q = re.sub(r'[?.,;!]', '', query).strip()
    
    patterns = [
        # "without using the X", "without X"
        r"\bwithout\s+(?:using\s+)?(?:the\s+)?([a-zA-Z0-9_\-\s]+)",
        # "excluding X"
        r"\bexcluding\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+)",
        # "other than X"
        r"\bother\s+than\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+)",
        # "doesn't use X", "does not use X", "do not require X", "doesn't require X"
        r"\b(?:doesn't|does\s+not|do\s+not|don't)\s+(?:use|require)\s+(?:specifying\s+)?(?:the\s+)?([a-zA-Z0-9_\-\s]+)"
    ]
    
    terms = []
    for pat in patterns:
        matches = re.finditer(pat, q, re.IGNORECASE)
        for m in matches:
            term = m.group(1).strip()
            
            # If the extracted phrase is long, let's clean it up.
            # Stop matching at conjunctions like "or", "to", "and", "but"
            conj_split = re.split(r'\b(?:or|and|to|but|for|with|in)\b', term, flags=re.IGNORECASE)
            if conj_split:
                term = conj_split[0].strip()
                
            term_words = term.split()
            # Remove leading articles and action verbs
            if term_words and term_words[0].lower() in ["a", "an", "the"]:
                term_words = term_words[1:]
            if term_words and term_words[0].lower() in ["assuming", "checking", "specifying", "using", "having"]:
                term_words = term_words[1:]
                
            cleaned_term = " ".join(term_words).strip()
            if cleaned_term and len(cleaned_term) > 2 and len(cleaned_term.split()) <= 4:
                terms.append(cleaned_term)
                
    return list(set(terms))


# ------------------------------------------------------------------------ ranking

def rrf_merge(lists):
    """Fuse several ranked chunk lists by reciprocal rank, keyed on chunk id."""
    scores, by_id = {}, {}
    for lst in lists:
        for rank, chunk in enumerate(lst, start=1):
            cid = chunk["id"]
            by_id[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return [by_id[c] for c in sorted(scores, key=scores.get, reverse=True)]


def dedupe(chunks):
    """Rank-ordered, de-duplicated (answer_id, question_id) pairs. Mirrors
    eval_retriever.py so numbers stay comparable."""
    aids, qids, seen = [], [], set()
    for chunk in chunks:
        meta = chunk["metadata"]
        aid = str(meta.get("answer_id"))
        if aid and aid not in seen:
            seen.add(aid)
            aids.append(aid)
            qids.append(str(meta.get("question_id")))
    return aids, qids


def rank(strategy, db, reranker, case, query, pool):
    """Runs one strategy end to end, returns (answer_ids, question_ids)."""
    if strategy == "history_concat":
        hits = db.search_hybrid(history_concat(case, query), n_results=pool)
        return dedupe(reranker.rerank(query, hits, top_k=len(hits)) if hits else [])

    if strategy == "decompose":
        parts = decompose(query)
        if parts:
            lists = [db.search_hybrid(p, n_results=pool) for p in parts]
            hits = rrf_merge([l for l in lists if l])[:pool]
        else:
            hits = db.search_hybrid(query, n_results=pool)
        # Rerank with the ORIGINAL query: sub-queries were only for recall.
        return dedupe(reranker.rerank(query, hits, top_k=len(hits)) if hits else [])

    if strategy == "negation_filter":
        from qdrant_client.http import models
        negated = extract_negation_terms(query)
        qdrant_filter = None
        if negated:
            qdrant_filter = models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="text",
                        match=models.MatchText(text=term)
                    ) for term in negated
                ]
            )
        hits = db.search_hybrid(query, n_results=pool, qdrant_filter=qdrant_filter)
        return dedupe(reranker.rerank(query, hits, top_k=len(hits)) if hits else [])

    hits = db.search_hybrid(query, n_results=pool)
    if not hits:
        return [], []

    if strategy == "norerank":
        return dedupe(hits)

    if strategy == "rerank_display":
        # Score against the plain chunk, not title + overlap prefix.
        pairs = [[query, c["metadata"].get("display_text", c["text"])] for c in hits]
        scores = reranker.encoder.predict(pairs)
        order = sorted(zip(hits, scores), key=lambda t: t[1], reverse=True)
        return dedupe([c for c, _ in order])

    reranked = reranker.rerank(query, hits, top_k=len(hits))

    if strategy == "rerank_blend":
        return dedupe(rrf_merge([reranked, hits]))

    return dedupe(reranked)   # baseline


# ------------------------------------------------------------------------ scoring

def score(rows, ks):
    out = {"n": len(rows)}
    out["mrr"] = sum(r["mrr"] for r in rows) / len(rows)
    for k in ks:
        out[f"r@{k}"] = sum(1 for r in rows if r["hit"][k]) / len(rows)
        out[f"q@{k}"] = sum(1 for r in rows if r["qhit"][k]) / len(rows)
        multi = [r for r in rows if r["n_gold"] > 1]
        out[f"both@{k}"] = (sum(1 for r in multi if r["both"][k]) / len(multi)) if multi else None
    return out


def main():
    args = parse_args()
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    cases = load_cases(args.category, args.limit)
    if not cases:
        print(f"No evaluable cases for category '{args.category}'.")
        return 1
    instances = [(c, q, s) for c in cases for q, s in queries_for_case(c)]
    print(f"Category '{args.category}': {len(cases)} cases -> {len(instances)} query instances")
    print(f"Strategies: {strategies}   pool={args.pool}  ks={ks}")

    qid_lookup = gold_question_ids(cases)

    print("Connecting to Qdrant (local)...")
    db = VectorDBManager(force_local=True)
    db.collection
    print(f"Loading reranker ({args.reranker_model})...")
    reranker = ReRanker(model_name=args.reranker_model)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw): return x

    pools = [int(p) for p in args.pools.split(",") if p.strip()] or [args.pool]
    combos = [(s, p) for p in pools for s in strategies]

    results, labels = {}, []
    for strategy, pool in combos:
        label = f"{strategy}@{pool}" if len(pools) > 1 else strategy
        labels.append(label)
        rows = []
        for case, query, _ in tqdm(instances, desc=f"{label:20s}"):
            gold = set(map(str, case["gold_answer_ids"]))
            gold_q = {qid_lookup[a] for a in gold if a in qid_lookup}
            aids, qids = rank(strategy, db, reranker, case, query, pool)
            mrr = next((1.0 / i for i, a in enumerate(aids, 1) if a in gold), 0.0)
            rows.append({
                "mrr": mrr,
                "n_gold": len(gold),
                "hit":  {k: bool(gold & set(aids[:k])) for k in ks},
                "qhit": {k: bool(gold_q & set(qids[:k])) for k in ks},
                "both": {k: gold.issubset(set(aids[:k])) for k in ks},
            })
        results[label] = score(rows, ks)

    strategies = labels
    base = results[labels[0]]
    print(f"\n=== {args.category} (n={base['n']}) ===")
    print("r@3 is the production-relevant column: top_k=3 is what reaches the LLM.")
    header = f"{'strategy':20s} {'MRR':>6}"
    for k in ks:
        header += f" {'r@'+str(k):>7} {'q@'+str(k):>7}"
    print(header)
    for s in strategies:
        r = results[s]
        line = f"{s:20s} {r['mrr']:6.3f}"
        for k in ks:
            d = r[f"r@{k}"] - base[f"r@{k}"]
            mark = "" if s == strategies[0] else ("+" if d > 0 else ("-" if d < 0 else "="))
            line += f" {r[f'r@{k}']:6.2f}{mark} {r[f'q@{k}']:7.2f}"
        print(line)

    if base.get(f"both@{ks[0]}") is not None:
        print("\n  BOTH golds retrieved (multi-gold cases only -- the honest bar for a")
        print("  comparison question, since recall@k credits finding just one side):")
        for s in strategies:
            vals = "  ".join(f"both@{k}={results[s][f'both@{k}']:.2f}" for k in ks)
            print(f"    {s:16s} {vals}")

    print("\nWinner here is a hypothesis, not a result -- confirm with the full eval:")
    print(f"  python evaluation/eval_retriever.py --local --pool {args.pool} --ks 10,50,100")
    return 0


if __name__ == "__main__":
    sys.exit(main())
