# DIAGNOSIS — why retrieval recall is not improving

**Date:** 2026-09-01
**Scope:** 79 hard cases (`negation`, `multi_hop`, `multi_turn`, `niche_topic`)
**Method:** offline analysis of `evaluation/results/_candidate_pools.json` and
`retriever_eval_hybrid_rerank_raw_pool50_20260901_022349.json`. No LLM, no API,
no re-indexing. Every number below is reproducible from files already on disk.

---

## Summary

Three findings, in order of how much they matter.

1. **`golden_dataset_v2.json` is contaminated and must not be used to measure this
   retriever.** Its new labels were sourced from documents the retriever returned,
   so it scores the system against its own output.
2. **Two-thirds of the loss is ranking, not retrieval.** The gold document is in the
   candidate pool 78.5% of the time and in the top-5 36.7% of the time.
3. **A large part of the remaining gap is not a real gap.** The benchmark counts one
   `answer_id` as correct on a corpus where the same question has several good
   answers, so correct results are scored as failures.

---

## 1. `golden_dataset_v2.json` is contaminated

Identical pools, identical pipeline, only the labels differ:

```
FIRST-STAGE (RRF, no reranker), 79 hard cases
  v1 labels:  R@1 16.5%  R@3 24.1%  R@5 27.8%  R@10 36.7%  MRR 0.230
  v2 labels:  R@1 21.5%  R@3 34.2%  R@5 39.2%  R@10 49.4%  MRR 0.306
```

+11.4 points at R@5 from relabelling alone. Split by whether the case was touched
in the 2026-09-01 review:

| | n | v1 R@5 | v2 R@5 | delta |
|---|---|---|---|---|
| unchanged cases | 66 | 28.8% | 28.8% | **+0.0%** |
| relabelled (ADD / REPLACE) | 13 | 23.1% | **92.3%** | **+69.2%** |

12 of 13 relabelled cases now hit at rank <= 5; before, 3 did. The unchanged cases
moving by exactly 0.0% is the control: this is not noise.

**Cause**, stated in `GOLDEN_DATASET_CHANGELOG.md` itself: *"All 39 were then
reviewed against the query and **the documents retrieval actually returns**."*
Candidates were drawn from the system's own ranking, so the added golds are
documents this retriever already ranks highly. The labels may well be *better* —
`neg_03`'s replacement clearly is — but they are no longer independent of the
system under test.

**Consequence:** `HANDOFF.md` §9 step 4 ("evaluate v2 and compare against v1")
would have shown +11 points and read as label-quality improvement. It is
measurement contamination. Same failure class as the fake judge: a number that
looks like progress and is not.

**Action:** report on v1. Mark v2 pending. To rehabilitate it, re-derive candidates
for those 13 cases from a source independent of the pipeline's ranking (e.g. BM25
over question titles) and re-approve.

---

## 2. The bottleneck is ranking, and it is quantified

v1 labels, 79 hard cases:

```
gold present in the 80-100 candidate pool : 78.5%   <- ceiling
gold in top-5 after RRF only              : 27.8%
gold in top-5 after cross-encoder         : 36.7%

  loss to RANKING     (in pool, not top-5) : 41.8 points
  loss to FIRST STAGE (never in pool)      : 21.5 points
```

The reranker is worth **+8.9 points R@5** over raw RRF — it is not useless — but it
plateaus, which is why every reranker config in the bake-off clustered at 35-37%.
The limit is not the model.

First-stage rank of the gold when it is in the pool (n=62), median **14**:

```
  rank  1-5  : 22
  rank  6-10 :  7
  rank 11-20 :  7
  rank 21-50 : 18     <- 26 of 62 golds sit past rank 20
  rank 51-100:  8
```

Neither RRF nor a cross-encoder rescues a document from rank 40.

### The two problems split cleanly by category

| category | pool ceiling | diagnosis |
|---|---|---|
| niche_topic | **94%** | pure ranking |
| multi_hop | **87%** | pure ranking |
| multi_turn | **67%** | first stage — query formulation |
| negation | **65%** | first stage — query formulation |

For `niche_topic` / `multi_hop` the document is almost always in the pool; no
embedding change or re-index will help, only better scoring.

For `multi_turn` / `negation` it is absent a third of the time, and the queries
show why — `mturn_09` is *"Do we need to scale features before running it?"*,
a sentence whose entire subject is a pronoun. No retrieval model fixes that; it
needs history-aware query rewriting *before* the search. Note that rewriting is
currently **silently disabled whenever Groq hits its daily cap** (`HANDOFF.md` §8).

---

## 3. Much of the remaining gap is a labelling artifact

What actually outranks the gold, from the cached pools:

```
niche_13  "Explain Tweedie distributions and their link functions in GLM"
  GOLD (rank 42): "Can a model for non-negative data with clumping at zeros..."
  RANK 1:         "When should one use a Tweedie GLM over a Zero-Inflated GLM?"
  RANK 3:         "Given a GLM using Tweedie, how do I find the coefficients?"

niche_05  "Explain Jackknife estimation of parameter bias and how it compares to Bootstrap"
  GOLD (rank 36): "Bootstrap vs. jackknife"
  RANK 2:         "Why is the jackknife less computationally intensive than the bootstrap?"

niche_06  "Core assumptions for causal identification using Instrumental Variables"
  GOLD (rank 76): [152985] "Is the key assumption for instrumental variables not testable?"
  RANK 3:         [421880] "Is the key assumption for instrumental variables not testable?"
                           ^ same question, different answer. Scored as a miss.
```

Measured: of the **40** cases whose gold falls outside the top-5, **8 (20%)**
already have a different answer to the *same question* at rank 1-5. That is only
what is detectable mechanically via `question_id`; the `niche_13` and `niche_05`
cases above are different threads with equally good answers and cannot be counted
automatically.

**So neither bound is real.** The 78.5% ceiling counts one `answer_id` as the only
correct document; the 36.7% score punishes the system for returning a different
correct one. Optimising against this metric tunes toward returning the specific
post that happened to be labelled, which is not a product goal.

### A genuine defect visible in the same data

Semantically unrelated documents appear at the very top:

- `niche_06` rank 1: *"The strongest password"* — for a query on instrumental variables
- `niche_02` rank 2: *"How to reduce the dimensionality of a similarity matrix"* — for Benjamini-Hochberg
- `niche_24` rank 2: *"Fit between two curves"* — for path analysis vs SEM

The cause is in `backend/core/sparse_store.py`. **It is not BM25.** It is raw
TF-IDF: `sum(raw_tf x idf)`, with Qdrant supplying IDF server-side. Two of BM25's
three components are missing, and a third defect (hashing) is present.

**Missing TF saturation — affects every query.** Measured over the cached pools,
maximum repetition of a content term within one chunk: p50 = 6, p95 = 22,
**max = 167**. A chunk repeating a term 167 times contributes 167x. BM25 with
k1 = 1.2 caps that at ~2.2x. A code block or table that repeats one variable name
outscores a well-written prose answer on any query containing that token. This is
the most likely mechanism behind the unrelated top hits above.

**Missing length normalisation — affects every query.** Chunk length in tokens:
p10 = 37, p50 = 196, p90 = 279, p99 = 350, max = 555. **p90/p10 = 7.5x.** With raw
TF and no `b` parameter, a long chunk carries a ~7.5x head start on identical
relevance. The chunker emits variable-length paragraphs, so this bias is
structural.

**Hash collisions — minor, ~4% of rare terms.** CRC32 into 2^18 dims. An earlier
draft of this document called collisions the headline defect; **that was wrong**
and is corrected here. 27.2% of the vocabulary shares a slot, but IDF is only
damaged where a token's slot is dominated by a much commoner term:

| band | tokens | IDF destroyed |
|---|---:|---:|
| ultra-rare (df 1-10) | 72,597 | 2,982 (4.1%) |
| rare (df 11-100) | 7,693 | 56 (0.7%) |
| mid (df 101-1000) | 2,667 | 2 (0.1%) |
| common (df >1000) | 550 | 0 (0.0%) |

Rare terms overwhelmingly collide with *other rare* terms, so slot df stays close
to term df and IDF survives. The specific technical terms these queries depend on
are mostly intact: `tweedie` df 17 / slot df 19; `hausman` df 36 / slot df 37;
`mahalanobis`, `hosmer`, `ljung` do not collide at all.

So: fix the sparse channel, but for `k1` and `b`, not for the hash.

**Fix.** Replace the hand-rolled generator with real BM25 — FastEmbed's
`Qdrant/bm25` sparse model is a drop-in behind the existing `to_qdrant_sparse`
interface, or build a real vocabulary in-house. **The dense vectors do not
change**, so this is a sparse-only rebuild over `embedded_points.jsonl`, not a
re-embed. Do it into a new collection behind an alias so rollback is a flip.
Delete the "hashing avoids a vocabulary file" justification from the docstring —
a vocabulary for 93k documents is a few MB, which is what that trade cost.

**Expected magnitude.** This is the sparse half of an RRF fusion; dense carries
most of the load and RRF is rank-based, so the gain should be real but moderate,
concentrated in `negation` and `niche_topic` where rare exact terms matter. A
large jump should be treated as suspicious and checked for contamination.
`evaluation/bm25_probe.py` exists to measure this offline and has never been run.

### Tested and rejected while investigating

- *Reranker input is starved by title/overlap boilerplate.* **Wrong.** The
  `question_title` block is 10.3% of the 600-char window; the reranker sees 58% of
  the average chunk.
- *Hash collisions in the sparse channel are destroying IDF for rare technical
  terms.* **Over-claimed.** Only 4.1% of ultra-rare terms lose their IDF; the
  terms these queries actually depend on are intact. The real sparse defects are
  the missing TF saturation and length normalisation. See above.
- *Duplicate chunks of the same answer are eating top-5 slots.* **Real but
  immaterial**, confirming the earlier finding. 0.48 wasted slots per query and
  32/79 cases affected, but deduping by `answer_id` gives R@5 27.8% -> 27.8%,
  R@10 36.7% -> 38.0%. The gold was not the next item in line.

---

## 4. What to do, in order

1. **Pool-judge the misses.** `build_judgment_pool.py` -> `judge_pool.py` ->
   `rescore_pooled.py`. Human judgments only — an LLM judging documents the
   retriever returned is exactly how v2 became circular. Expected to move reported
   recall by 10-20 points with no pipeline change, and produces a metric that
   responds to real improvements instead of label luck.
2. **Run `bm25_probe.py`.** The sparse channel is raw TF-IDF, not BM25: no TF
   saturation (max in-chunk repetition 167x vs BM25's ~2.2x cap) and no length
   normalisation (7.5x p90/p10 bias). Both affect every query. Measure offline
   first, then rebuild sparse-only behind an alias.
3. **Deterministic pronoun resolution for `multi_turn`**, using `chat_history`. No
   LLM. Targets the 67% pool ceiling, which nothing downstream can touch.
4. **Establish a noise floor** before trusting any A/B. Still not done; it is why
   the rejected-hypotheses table in `HANDOFF.md` §4 states per-config verdicts
   more strongly than n=79 supports.
5. **Then** `query_expansion_test.py`. It targets the 21.5% first-stage loss, which
   is the smaller half of the problem.

## 5. Reporting

Report all three definitions of correct side by side, with judgment coverage:

| definition | what counts as correct |
|---|---|
| strict | the exact labelled `answer_id` |
| thread | any answer from the labelled answer's question thread |
| pooled | the label, or any document a human judged as answering the query |

An unjudged document counts as irrelevant, so `pooled` is a lower bound until
coverage is complete. State the coverage whenever the number is quoted.
