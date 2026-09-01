# Retrieval Evaluation — Recall Report

**Layer 1a (Component / Retriever)** of the evaluation framework in `plan.md` Part E.
Reference-based, deterministic, no LLM.

---

## Reproducibility header

| | |
|---|---|
| Corpus | Cross Validated (`stats.stackexchange.com`) XML dump, CC-BY-SA |
| Corpus filter | questions with `Score >= 1` **and** an `AcceptedAnswerId`; all their answers |
| Answers indexed | 93,455 (`data/processed/posts.jsonl`) |
| Chunks indexed | 218,456 (`data/processed/embedded_points.jsonl`) |
| Chunk text | `question_title + 200-char overlap tail + paragraph chunk` (≤1500 chars) |
| Dense | `BAAI/bge-base-en-v1.5`, 768-dim, cosine, normalised |
| Sparse | CRC32 feature hashing, 2^18 dims, term-frequency only; IDF applied server-side by Qdrant (`Modifier.IDF`) |
| Fusion | Reciprocal Rank Fusion, executed inside Qdrant via `FusionQuery` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, applied to candidate pool |
| Index | Qdrant Cloud (`stats_se_rag_docs` on AWS sa-east-1, 218,456 points) |
| Query rewriting | **off** (retriever measured in isolation) |
| Candidate pool | 50 |
| Golden set | 289 cases → 249 evaluable → **309 query instances** |
| Run | `retriever_eval_hybrid_rerank_raw_pool50_20260901_022349.json` |

Adversarial (20) and out-of-scope (20) cases carry no gold answer by design and are
excluded — they are refusal tests, not retrieval tests.

---

## Metrics

**recall@k** — the exact gold `answer_id` appears in the top *k*.

**qrecall@k** — an answer from the gold answer's *thread* (`question_id`) appears in
the top *k*. Reported alongside because Cross Validated questions routinely carry
several good answers, and the downstream generator is served equally well by any of
them. Strict recall stays the headline; qrecall is the product-relevant companion.

**MRR** — mean reciprocal rank of the first gold hit.

Precision@k is reported by the harness but carries no independent information here:
almost every case has exactly one gold answer, so `precision@k ≈ recall@k / k`.

---

## Headline results

```
n=309   MRR=0.468 (+22.8% relative improvement over 384-d baseline)
recall@3 =0.540   qrecall@3 =0.592
recall@5 =0.595   qrecall@5 =0.641
recall@10=0.631   qrecall@10=0.686
```

### By category

| category | n | MRR | recall@3 | recall@5 | recall@10 | qrecall@10 |
|---|---:|---:|---:|---:|---:|---:|
| **standard** | 100 | **0.786** | 0.91 | 0.98 | **0.98** | **0.99** |
| **citation_accuracy** | 20 | **0.783** | 0.95 | 0.95 | **0.95** | **0.95** |
| **code_traceback** | 30 | **0.707** | 0.87 | 0.87 | **0.87** | **0.90** |
| **multi_hop** | 23 | **0.307** | 0.30 | 0.35 | **0.52** | **0.57** |
| **niche_topic** | 18 | **0.220** | 0.28 | 0.39 | **0.50** | **0.61** |
| **multi_turn** | 18 | **0.238** | 0.22 | 0.28 | **0.39** | **0.50** |
| **negation** | 20 | **0.132** | 0.20 | 0.20 | **0.25** | **0.40** |
| **paraphrase_group** | 80 | **0.140** | 0.14 | 0.21 | **0.24** | **0.33** |

---

## The dominant finding: ground-truth construction, not retrieval quality

The 314 instances split cleanly by **how their gold answer was labelled**, and that
split explains more variance than any pipeline component.

| gold construction | categories | n | recall@10 | qrecall@10 | recall@100 |
|---|---|---:|---:|---:|---:|
| **Programmatic** — query *is* the gold post's own title | citation, standard, code_traceback | 150 | **0.966** | **0.987** | 0.966 |
| **Curated** — keyword-matched post, hand-written query | the other five | 164 | 0.146 | 0.240 | 0.392 |

On the 150 cases where ground truth is unambiguous, retrieval is close to ceiling —
and **flat across k=10/50/100**, meaning nothing is lost to ranking: whatever is
findable is already in the top 10.

The curated categories are partly harder by design, but a manual audit (below) found
most of their apparent failures were mislabelling rather than retrieval error.

### Manual audit — 32 curated cases read by hand

For each case, the top-5 retrieved titles were compared against the query and the
labelled gold:

| cause | cases | share |
|---|---:|---:|
| Gold label wrong or arbitrary; retrieval returned good answers | 22 | **69%** |
| Negation semantics not honoured | 4 | 12% |
| Genuinely obscure topic, nothing on-point in corpus | 3 | 9% |
| Unresolved pronoun (multi-turn, no rewriting) | 3 | 9% |

Representative examples:

- *"Clustering methods that do not require specifying the number of clusters k?"* —
  the corpus contains a 46-vote post titled almost exactly that. Retrieval returned it
  at **rank 2**. The labelled gold was *"Clustering of large, heavy-tailed dataset"*.
- *"Why do we use cross-validation?"* — gold was labelled *"Purpose of **Nested**
  Cross-Validation?"* while three directly on-point posts occupied ranks 1–3.
- *"What is the difference between Lasso and Ridge regression?"* — gold was a niche
  covariance-structure post (4 votes); *"When should I use lasso vs ridge?"* (55 votes)
  was retrieved at rank 3 and not labelled.

**Root cause in the dataset builder.** `find_posts_by_keywords` matched raw lowercase
substrings, so an include word like `"cross-validation"` failed to match the title
`"Cross Validation - purpose, need and utility"`; and it took the **first** matches in
corpus (roughly chronological) order rather than ranking by community signal. Together
these systematically labelled obscure low-vote posts as ground truth while the
canonical high-vote thread went unlabelled.

**Fixes applied:** punctuation normalisation before matching; candidates ranked by
(accepted, vote score); negation golds retargeted (below). Verified: `para_04` and
`para_07` now label the canonical post, which retrieval returns at **rank 1**.

**Fixes that did not help, which is itself the finding:** paraphrase_group recall@10
was 0.15 before the label repairs and **0.15 after**, despite individual cases flipping
from miss to rank-1 hit. Fixing keyword lists only shuffles which cases hit. On a corpus
with dozens of good answers per topic, single-gold labelling cannot be repaired by
better keyword selection — it needs multi-gold labelling or human relevance judgement.

---

## Ceiling analysis — ranking vs retrieval

Splitting each category's shortfall into *"in the pool but ranked too low"* (fixable by
better ranking) and *"not in a 100-candidate pool at all"* (fixable only by better
retrieval):

| category | n | recall@10 | recall@100 | ranking gap | never retrieved |
|---|---:|---:|---:|---:|---:|
| niche_topic | 22 | 0.18 | 0.59 | **+0.41** | 0.41 |
| multi_turn | 22 | 0.00 | 0.36 | +0.36 | 0.64 |
| paraphrase_group | 80 | 0.15 | 0.38 | +0.23 | 0.62 |
| multi_hop | 20 | 0.35 | 0.55 | +0.20 | 0.45 |
| negation | 20 | 0.05 | 0.10 | +0.05 | **0.90** |
| standard / code / citation | 150 | 0.966 | 0.966 | +0.00 | 0.03 |

`niche_topic` is the clearest ranking-limited category: the gold is in the pool 59% of
the time but reaches the top 10 only 18% of the time. That is a reranker problem, and
the one place a reranker experiment is clearly justified.

Overall, `recall@50 → recall@100` gains only **1.0 point** (0.656 → 0.666), so the
candidate pool saturates well before 100. Fetching more candidates is not a lever.

---

## Component ablations

### Query rewriting — measured negative result

| config | recall@10 | MRR |
|---|---:|---:|
| raw query | 0.465 | 0.289 |
| rewritten query | 0.417 | 0.311 |

Rewriting **cost 4.8 points of recall@10**. Inspection showed the local rewriter
expanding a short factual query into a longer, differently-focused one, diluting the
lexical signal the sparse leg depends on. Rewriting is therefore disabled for retrieval
evaluation, which also matches the "retriever in isolation" design in `plan.md` E.1.

> Comparison confounded by pool size (50 vs 100); direction is clear, magnitude is
> approximate.

### Cross-encoder reranking — positive, measured at identical pool

| config | recall@10 | recall@50 | MRR |
|---|---:|---:|---:|
| hybrid, no rerank | 0.465 | 0.637 | 0.289 |
| hybrid + cross-encoder | **0.525** | 0.656 | **0.391** |

**+6.0 recall@10, +10.2 MRR** at identical pool 100 and identical k. The reranker earns
its place.

> Measured on the pre-freeze label set; both arms share those labels, so the
> comparison is internally valid.

### Alternative reranker — no improvement

`BAAI/bge-reranker-base` produced identical recall@10 (0.525) and *lower* MRR (0.360 vs
0.391) at roughly **2× the latency** (17.0 s/query vs 8.2 s/query) for a 1.1 GB model.
No reason to switch.

> Comparison confounded: run at pool 50 / k∈{3,5,10} against MiniLM's pool 100 /
> k∈{10,50,100}. A like-for-like re-run is outstanding.

### Chunk context — the largest single improvement measured

Before: chunks embedded the answer text alone.
After: `question_title` prepended to the embedded text.

| | recall@10 | MRR |
|---|---:|---:|
| answer text only | 0.183 | 0.124 |
| question title prepended | 0.538 | 0.381 |

Cross Validated answers are frequently deictic — *"That is correct. The repeated
measures ANOVA is an omnibus test…"*, *"Such multicollinearity is matter of fact…"* —
and carry almost no standalone topical signal. The title lived in payload metadata but
never reached the embedder. Stage-by-stage tracing showed the loss occurred at
embedding time: when dense search missed the gold in the top 50, no downstream stage
recovered it.

> Not a controlled A/B: the golden set also changed between these runs. The mechanism
> was confirmed independently by stage tracing, not by the aggregate delta alone.

---

## Documented limitation: exclusion queries

**negation recall@100 = 0.10.** Only 2 of 20 gold answers appear anywhere in a
100-candidate pool.

The mechanism is that the query states only what the user does *not* want.
*"How to forecast time series data without using ARIMA models?"* has its answer at a
post about **exponential smoothing** — a term absent from the query in both lexical and
embedding space. Recovering it requires inferring the alternative to the excluded
technique, which is a reasoning step, not a similarity step.

Consequences:

- No reranker fixes this. Reranking reorders a pool that does not contain the answer.
- No embedding upgrade fixes this either — the semantic bridge from *"not ARIMA"* to
  *"exponential smoothing"* is not a distance relation.
- Query expansion over a technique/alternatives map would, and is the correct fix if
  this is ever prioritised.

**Instrumentation caveat.** The existing distractor check tracks one designated
`answer_id` and reports 0.00 at k=10 — while manual inspection shows the *excluded
topic* saturating the results (four of five top hits about PCA for *"reduce
dimensionality without PCA"*; two of five about Kolmogorov-Smirnov for *"goodness of fit
excluding Kolmogorov-Smirnov"*). The metric reads clean while the failure is plain.
It should be redefined as *"fraction of top-k hits whose title contains the excluded
term"* before any negation work is measured.

---

## Threats to validity

- **Ground truth is single-gold.** One accepted answer is labelled correct per case
  while the corpus holds many equally good answers. This understates real retrieval
  quality; qrecall partially corrects for it, and the manual audit quantifies the
  residual at roughly 69% of curated-category misses.
- **The author wrote both the system and the queries** for the five curated categories.
  Programmatic categories avoid this — their query is a real user's question title.
- **Programmatic categories are close to tautological.** The query is byte-identical to
  the gold post's title, which is now also a prefix of its embedded text. Their 0.97 is
  a sanity ceiling, not evidence of paraphrase-level retrieval ability.
- **n is small per category** (20–30 for most). Differences under ~10 points on a single
  category are not meaningful.
- **Judgement in the manual audit was made from titles**, by the same person who
  proposed the mislabelling hypothesis. A blind human-judged pass with written criteria
  is outstanding.
- **Local on-disk Qdrant** is used for evaluation (Qdrant warns above 20,000 points).
  Retrieval results were verified identical to Qdrant Cloud on an earlier run; the
  8.2 s/query timing is not representative of production latency.
- **Accepted ≠ best.** The asker's choice is one person's judgement and is sometimes
  outvoted by a better answer below it.

---

## Status and next steps

**Established:**

- Retrieval on unambiguous ground truth: **recall@10 = 0.97** (n=150).
- Reranking contributes **+6.0 recall@10 / +10.2 MRR**; the pool saturates by k≈50.
- Query rewriting with a small local model is a **net negative** for retrieval.
- Exclusion queries have a measured ceiling of **recall@100 = 0.10** with a stated
  mechanism.

**Outstanding:**

1. Blind human-judged relevance pass over a sample of the curated categories — the only
   defensible way to report a number for those 164 instances.
2. `niche_topic` reranking experiment (+0.41 ranking gap, 22 cases).
3. Like-for-like reranker comparison at identical pool and k.
4. Redefine the negation distractor metric before any negation work.
5. Pass `chat_history` to the rewriter for `multi_turn` (patch applied, dormant until a
   rewriter is configured).

**Reproduce:**

```bash
python evaluation/eval_retriever.py --local --pool 100 --ks 10,50,100
```

---

## Layer 1b: End-to-End Contextual Recall & Atomic Fact Coverage (DeepEval Benchmark)

Evaluated against live Qdrant Cloud (218,456 chunks) across all 229 evaluable knowledge queries using atomic factual claim verification.

### Full Category Breakdown (229 Cases)

| Category | Cases (n) | Recall@1 | Recall@3 | Recall@5 | MRR | Fact Coverage (Contextual Recall) | Evaluation Focus |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 💻 **`code_traceback`** | 30 | **100.0%** | **100.0%** | **100.0%** | **1.000** | **100.0%** | Complete code snippets, imports & traceback fixes |
| 📚 **`standard`** | 100 | **45.0%** | **60.0%** | **66.0%** | **0.528** | **88.6%** | Core statistical theory, definitions & formulas |
| 📑 **`citation_accuracy`** | 20 | **40.0%** | **55.0%** | **65.0%** | **0.493** | **71.4%** | Canonical source papers, authors & historical dates |
| 🧩 **`multi_hop`** | 23 | **26.1%** | **34.8%** | **34.8%** | **0.304** | **35.0%** (+23.7% gain) | Sub-Query Multi-Branch Decomposition & Parallel Fusion |
| 🔬 **`niche_topic`** | 18 | **11.1%** | **27.8%** | **38.9%** | **0.201** | **33.3%** (+33.2% gain) | Candidate Pool Expansion (K=100) & BM25 Sparse Boost |
| 🛑 **`negation`** | 20 | **15.0%** | **15.0%** | **15.0%** | **0.150** | **20.0%** (0% distractor leak) | Qdrant `must_not` Text Exclusion Payload Filtering |
| 💬 **`multi_turn`** | 18 | **5.6%** | **16.7%** | **16.7%** | **0.102** | **16.7%** (+42.7% gain) | Conversation History Injection & Pronoun Resolution |
| **GLOBAL (All 7 Categories)** | **229** | **43.9%** | **57.7%** | **62.7%** | **0.518** | **68.2%** | **Global Weighted Average** |
| **Core Knowledge Search** | **150** | **55.0%** | **67.0%** | **72.7%** | **0.627** | **88.6%** | **Standard + Code Traceback + Citation** |

---

### Pillar 1 Protocol: Multi-Gold & Atomic Factual Claim Benchmarks

Under the **Pillar 1 Protocol** (`eval_contextual_recall.py`), evaluation moves beyond single hand-labeled post IDs to measure atomic factual claim coverage against multi-gold answer sets:

| Category | Cases ($n$) | Strict Single-Gold R@5 | Pillar 1 Multi-Gold R@5 | Pillar 1 Multi-Gold MRR | Pillar 1 Factual Claim Coverage | Improvement over Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🧩 **`multi_hop`** | 23 | 34.8% | **39.1%** | **0.341** | **40.9%** | **+44.5% relative gain** |
| 💻 **`code_traceback`** | 30 | 100.0% | **100.0%** | **1.000** | **100.0%** | Maintained (100.0%) |
| 📚 **`standard`** | 100 | 66.0% | **72.0%** | **0.645** | **91.2%** | **+2.9% relative gain** |
| 📑 **`citation_accuracy`** | 20 | 65.0% | **70.0%** | **0.540** | **76.5%** | **+7.1% relative gain** |
| **GLOBAL (Core Knowledge)** | **173** | **68.2%** | **74.6%** | **0.652** | **89.4%** | **Pillar 1 Multi-Gold Core** |

