# HANDOFF — RAG Portfolio project, evaluation debugging session

**Written:** 2026-09-01, end of a full-day session.
**Purpose:** hand this to a fresh assistant so it does not re-derive, re-guess, or
re-break what has already been established. Read all of it before proposing anything.

---

## 0. Rules for whoever picks this up

These are not stylistic preferences. Every one of them was learned by being burned today.

1. **Never state a finding you have not verified against the actual data.** This project
   has been damaged repeatedly by confident, fluent, wrong explanations — including one
   at the start of this session that invented a post ID and a failure mechanism that did
   not exist. If you catch yourself constructing a plausible story, stop and go read the file.
2. **Check before you claim a capability is broken or missing.** Read the code, run the query,
   open the JSON.
3. **Test hypotheses cheaply before shipping them.** Three "obviously correct" optimizations
   were killed by five-minute experiments today (see §4). Assume yours might be too.
4. **Say when you are wrong, immediately and plainly.** No hedging, no burying it.
5. **Log every change** to `PROJECT_LOG.md` in its `WHAT / WHY / OUTCOME` table format.
   `CLAUDE.md` makes this a top-priority rule. It was ignored for most of this session; do not repeat that.
6. **Do not let the user rush you into damaging the dataset.** Late-night bulk re-labelling
   produced 3/3 wrong decisions before it was caught. Slow down or stop.

---

## 1. What the project is

A RAG system over the Cross Validated (stats.stackexchange.com) corpus, built as a
job-search portfolio piece.

| Component | Value |
|---|---|
| Corpus | 93,455 answers → 218,456 chunks (`data/processed/posts.jsonl`) |
| Chunk text | `question_title + 200-char overlap tail + paragraph chunk` (≤1500 chars) |
| Dense | `BAAI/bge-base-en-v1.5`, 768-dim, cosine, normalised |
| Sparse | CRC32 feature hashing, 2^18 dims, TF only; IDF applied by Qdrant (`Modifier.IDF`) — **not real BM25** |
| Fusion | Reciprocal Rank Fusion inside Qdrant (`FusionQuery`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, input truncated to 600 chars |
| Index | Qdrant Cloud, `stats_se_rag_docs`, AWS sa-east-1 |
| Generator | Groq (`openai/gpt-oss-20b`) |
| Eval judge | Gemini via DeepEval `ContextualRecallMetric` |
| Golden set | 289 cases / 309 query instances, 10 categories |

---

## 2. The real numbers (all reproducible from `evaluation/results/*.json`)

### Full golden set — 309 query instances
| Metric | Answer-level | Question-level |
|---|---|---|
| Recall@5 | **59.5%** | 64.1% |
| Recall@10 | 63.1% | 68.6% |
| MRR | **0.468** | |

### By category (R@5)
| Category | n | R@5 | MRR |
|---|---|---|---|
| standard | 100 | **98.0%** | 0.786 |
| citation_accuracy | 20 | 95.0% | 0.783 |
| code_traceback | 30 | 86.7% | 0.707 |
| niche_topic | 18 | 38.9% | 0.220 |
| multi_hop | 23 | 34.8% | 0.307 |
| multi_turn | 18 | 27.8% | 0.238 |
| paraphrase_group | 80 | **21.2%** | 0.140 |
| negation | 20 | 20.0% | 0.132 |

**Excluding the paraphrase stress-test: 72.9% R@5.**

### Hard-category contextual recall (79 cases, real LLM judge, 0 fallbacks)
`contextual_recall_eval_20260901_181021.json` — **30.2% fact coverage**,
strict R@1 16.5%, R@5 38.0%, MRR 0.245.

### The decisive diagnostic
- Gold document **in the 80–100 candidate pool: 78.5%**
- Gold document **in final top-5: 36.7%**
- Fact coverage **when gold retrieved: 0.635**; **when not: 0.098**

Coverage is fine when retrieval works. **The headline is bounded by retrieval hit-rate.**

---

## 3. Bugs found and fixed (all verified, all in git)

1. **The judge was fake.** DeepEval failed on every case; the harness silently substituted a
   hit-based heuristic (1.0 if gold in top-3 / 0.5 top-5 / 0.0) while printing
   "Pillar 1 Fact Coverage". Audited: **79/79 rows fallback, 0 real scores.** Causes: judge
   model was `openai/gpt-oss-20b` — the *same family as the generator under test*, which
   `judge_model.py`'s own header says was rejected as self-grading — and it is capped at 8K TPM.
   *Consequence: every contextual-recall number before this was retrieval hit-rate relabelled,
   including the 82.3% in `PROJECT_LOG.md` dated 2026-09-01 03:42. That figure is not real.*
2. **Negation filter deleted its own answers.** Any candidate containing an excluded term was
   dropped. **9/20 negation cases had gold answers containing that term** (an answer recommending
   alternatives names the method it replaces). Now demotes on question title only.
   Existed **only in the eval harness**, never in `app.py`. negation 20.0% → 35.0%.
3. **Multi-gold tag contamination.** `build_multi_gold_sets` pulled "up to 3 shared topic post IDs"
   from any case sharing a tag into `expected_output`. 64/79 cases exceeded the 2500-char cap;
   **55% of gold text discarded**, survival decided by `set` iteration order. `hop_08`: 8 golds /
   33,312 chars → 2,500, scored 0.00 despite correct gold at rank 1. Score tracked gold-doc *count*
   (1→0.379, 2→0.243, 3-4→0.207), not quality. Now scores per gold document, deterministically.
4. **Silent failure masking.** The fallback overwrote the real exception text, so a 100%-broken run
   looked identical to a working one. Rows now carry `used_fallback_heuristic` and `judge_error`.
5. **`os` shadowing** in `llm_service.py.__init__` (a local `import os` made an earlier `os.getenv`
   raise `UnboundLocalError`). Removed.
6. **Dead `HuggingFace` serverless endpoint** — `api-inference.huggingface.co` was deprecated
   ~Nov 2025. Updated to `router.huggingface.co/v1`; still fails (`model_not_supported`) because
   HF Inference Providers requires enabling a provider in the account. **Unresolved, low priority.**
7. **Gemini fallback silently disabled** in `llm_service.py` by a `gemini_key.startswith("AIzaSy")`
   check that this project's key does not match. Removed.

---

## 4. Hypotheses TESTED AND REJECTED — do not retry these

| Hypothesis | Result | Evidence |
|---|---|---|
| Swap in a stronger reranker (`BAAI/bge-reranker-base`) | **WORSE** — R@5 20.0% vs 28.0%, 12x slower | `reranker_bakeoff.py` |
| Reranker input window 600 → 2000 chars | **No R@5 gain**, worse MRR, 2x slower. Reverted. | `reranker_bakeoff.py` |
| BGE asymmetric query instruction prefix | **−1.3% R@5, −3.8% R@10** | `query_instruction_test.py` |
| Chunk deduplication before top-k | **+0.0% @5, +1.3% @10** — real but immaterial | pool analysis |
| "The golden dataset is broadly mislabelled" | **REFUTED.** 26 keep / 9 add / 4 replace across 39 flagged → ~84% sound, ~5% genuine mislabels | `dataset_triage.py` + manual review |

Every reranker config clusters at ~35–37% R@5. **Reranker choice is not the bottleneck.**

---

## 5. Golden dataset state

- **Only 9/79 hard-category labels were human-verified.**
- `golden_audit_report.json` marked **40/40 cases `is_aligned: true, NO_ACTION`** with
  justifications that contradict the post text. Example: for `neg_03` it claims the answer
  "discusses methods to compare group means"; the post is two sentences on Kruskal-Wallis
  homoscedasticity, **score 0**. That audit was an LLM confabulation and should not be trusted.
- All 39 triage-flagged cases were reviewed this session: **26 KEEP, 9 ADD, 4 REPLACE**.
  Genuine mislabels: `neg_03`, `hop_12`, `niche_15`, `neg_16`.
- Output: `golden_dataset_v2.json` + `GOLDEN_DATASET_CHANGELOG.md`, provenance
  `method: llm_proposed_human_approved` (weaker than two independent annotators — recorded as such).
- **`golden_dataset_v2.json` has NOT yet been evaluated.** The eval still reads `golden_dataset.json`.
- **`neg_05` is the case to remember:** gold `[13698]` *"What are modern, easily used alternatives to
  stepwise regression?"* (score 61) is an exact match for the query and retrieval failed to surface it
  within **80 candidates**. Label correct, retrieval wrong.

---

## 6. THE OPEN PROBLEM — vocabulary mismatch

This is the one real, unfixed finding. Everything else is downstream of it.

Retrieval is **sensitive to surface phrasing**. Same question, same gold answer:

- *"What is the difference between Lasso and Ridge regression?"* → **hit**
- *"How does L1 regularization differ from L2 regularization?"* → **miss**

Quantified on `paraphrase_group` (20 groups × 4 phrasings):

| Variant | R@5 |
|---|---|
| variant_0 | 40.0% |
| variant_1 | 25.0% |
| variant_2 | 10.0% |
| variant_3 | 10.0% |

- 1/20 groups: all four phrasings retrieved
- 8/20: all four failed (genuine corpus gap)
- **11/20: inconsistent — the answer is findable, but only under some wordings**

This also explains **negation** ("without X" is an unusual phrasing) and `neg_05`.

### Next step, already built but NOT yet run
`evaluation/query_expansion_test.py` — deterministic, no API, no re-indexing. Compares:
`baseline` / `synonym` (domain synonym+acronym substitution) / `prf` (pseudo-relevance feedback:
search once, mine distinctive terms from top docs, re-search) / `combined` (RRF fusion of all).
Run it. If `combined` beats baseline meaningfully, wire it into `vector_store.py`.

### Other candidates, unexplored
- **Real BM25** instead of CRC32 hashed TF-only sparse vectors (no length normalisation, no
  saturation, hash collisions). `evaluation/bm25_probe.py` was written but **never run** — it tests
  whether real BM25 finds the 17 documents the current pipeline misses entirely. Requires re-indexing if adopted.
- **HyDE** — embed a hypothetical answer instead of the question. Needs an LLM call per query.
- **Query expansion is already half-built** in `llm_service.py` (`generate_semantic_variants`,
  `decompose_query`) but only fires for `niche_topic`, and depends on a rate-limited API. Making it
  deterministic and universal is the highest-leverage change available.

---

## 7. Tools built this session (all no-LLM unless noted)

| File | Purpose |
|---|---|
| `evaluation/smoke_test.py` | Pre-flight: are deps installed, judge alive, providers reachable. **Run before any eval.** |
| `evaluation/retrieval_diagnostic.py` | Splits "never retrieved" from "lost by reranker" |
| `evaluation/reranker_bakeoff.py` | A/B reranker configs on identical cached pools |
| `evaluation/query_instruction_test.py` | BGE prefix A/B (rejected) |
| `evaluation/query_expansion_test.py` | **Not yet run** — the next experiment |
| `evaluation/bm25_probe.py` | **Not yet run** — real BM25 vs hashed sparse |
| `evaluation/dataset_triage.py` | Ranks golden cases by red flags |
| `evaluation/dataset_review.py` | Interactive human review tool (resumable) |
| `evaluation/gold_audit.py` | Embedding-based label sanity check |
| `evaluation/generate_report.py` | Builds `EVAL_REPORT_GENERATED.md` **from result JSONs** so no number can be hand-typed |

---

## 8. Operational gotchas

- **Gemini free tier: 15 requests/MINUTE and 500 requests/DAY *per model*.** Per-gold-document
  scoring pushed a full run to ~360 requests and exhausted the daily cap mid-run. `MAX_GOLD_DOCS_SCORED`
  is now **1** (~160 requests/run). Switching to a *different* Gemini model gives a fresh daily quota
  (`GEMINI_JUDGE_MODEL` in `.env`).
- **Groq free tier: 8K TPM, 200K tokens/DAY per model.** Exhausted repeatedly. When it fails,
  `rewrite_query` and `decompose_query` silently fall back to heuristics — which *disables the
  Pillar 2 machinery* and degrades multi_hop / multi_turn without any error surfacing in the results.
- **Jina reranker API returns 403 (`insufficient balance`)** on every call; everything falls back to
  the local cross-encoder. Expected, not a bug.
- **`dataset_review.py` holds all decisions in memory and rewrites the whole file on save** — editing
  `review_decisions.json` externally while it runs will be clobbered. Quit it first.
- **The Windows `.venv` is not usable from a Linux shell**; run evals in PowerShell on the host.

---

## 9. What to do next, in order

1. Run `python evaluation/query_expansion_test.py`. This targets the actual root cause.
2. If it wins, wire the winning strategy into `backend/core/vector_store.py` so it benefits the live
   app, not just the eval. Re-run `eval_retriever.py` on all 309 instances and log the delta.
3. Run `python evaluation/bm25_probe.py` to decide whether the sparse channel justifies re-indexing.
4. Evaluate `golden_dataset_v2.json` and compare against v1 to quantify the label corrections.
5. Only then write the final report — `generate_report.py` builds it from the result files.

**Do not** start re-indexing, re-chunking, or bulk re-labelling without measuring first.

---

## 10. How to frame this project honestly

The headline is **not** "30.2%". That is the adversarial slice with a strict metric.

> Hybrid dense+sparse RAG over 93k Cross Validated answers (218k chunks, Qdrant, RRF fusion,
> cross-encoder reranking). **98% Recall@5 on standard queries, 59.5% overall across 309 query
> instances** spanning 8 categories including deliberately adversarial ones — negation, multi-hop,
> multi-turn, and paraphrase robustness. Isolated the dominant failure mode to vocabulary mismatch
> via controlled experiments, after discovering and fixing an evaluation harness that had been
> silently substituting heuristic scores for LLM judgments on 100% of cases.

The strength of this project is the **evaluation rigour and the debugging trail**, not the score.
Three optimization hypotheses were tested and rejected with data; a fabricated automated audit was
caught; a benchmark that was scoring correct retrieval as failure was fixed. Most portfolio RAG
projects have no evaluation at all.
