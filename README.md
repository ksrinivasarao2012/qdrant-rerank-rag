---
title: QdrantRERANK
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Cross Validated RAG Assistant

Ask a statistics or machine-learning question in plain English. Get a grounded, mathematically rigorous answer built directly from real [Cross Validated](https://stats.stackexchange.com) discussions, with full citation links back to the original community threads.

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue?style=flat-square)](https://huggingface.co/spaces/Srinivasa12/rag-portfolio)
[![Qdrant](https://img.shields.io/badge/Qdrant-hybrid%20search-red?style=flat-square)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/code-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Corpus: CC BY-SA](https://img.shields.io/badge/corpus-CC%20BY--SA-lightgrey.svg?style=flat-square)](https://stackoverflow.com/help/licensing)

**93,455 answers · 218,456 indexed chunks · evaluated on 289 benchmark cases across 10 categories**

---

## Why This Corpus

Six candidate corpora were evaluated and rejected before selecting Cross Validated. The architectural decision log is documented in [`plan.md`](plan.md) Part A; the summary is that most candidate enterprise corpora do not actually warrant a retrieval system.

Four editions of the MCC Laws of Cricket, for example, extract to roughly **230K tokens** against a **131K-token** context window. When the entire corpus nearly fits inside a single prompt, retrieval is not solving an architectural problem — a well-crafted prompt is. Nike/Adidas annual reports failed for a different reason: Adidas files a 20-F under IFRS with a December year-end, while Nike files a 10-K under US GAAP ending in May. Any comparative financial query would compare mismatched accounting standards, compromising ground-truth integrity.

Cross Validated passes the three essential RAG criteria:

| Test | Why It Matters | Empirical Finding |
|---|---|---|
| **Exceeds Context Window** | If the data fits in-context, retrieval adds unnecessary complexity | ~100× a standard 128K context window (218,456 chunks) ✅ |
| **Vocabulary Mismatch** | The primary differentiator of dense embeddings over lexical search | Severe: users describe symptoms, while answers use formal theorems ✅ |
| **Unstructured Knowledge** | Stable database keys tempt developers into building relational SQL schemas | No fixed schema, free-form mathematical discourse ✅ |

When a user asks *"my model gets 99% on training data but 60% on new data"*, the canonical answers cite **overfitting**, **L2 regularization**, and the **bias–variance tradeoff** — terminology that appears nowhere in the query. Bridging that semantic gap is the fundamental purpose of this system.

Furthermore, `AcceptedAnswerId` in the official StackExchange dump ensures **the canonical answer is peer-validated ~200,000 times over**, providing high-fidelity ground truth without manual annotation bias.

---

## System Architecture

```
                       User Question / Code Query
                                   │
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │     Universal Normalizer & Query Rewriter        │
          │   - Typo & Spelling Normalization (<100ms)       │
          │   - Multi-Turn Conversational Pronoun Resolution │
          │   - Negation-Aware Domain Alternative Injection  │
          │   - Multi-Concept Sub-Query Decomposition        │
          └────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
┌────────────────── Qdrant Cloud (Parallel Hybrid Search) ──────────────────┐
│  Dense Vector:   BAAI/bge-base-en-v1.5 (768-dim, Cosine Distance)          │
│  Sparse Vector:  CRC32 Feature Hashing (2^18 dims), Term-Frequency (TF)   │
│  Server Scoring: Server-side BM25 IDF applied via Modifier.IDF             │
│  Execution:      Dense & Sparse pipelines run concurrently in parallel     │
│  Fusion:         Reciprocal Rank Fusion (RRF) executed natively in Qdrant  │
└──────────────────────────────────┬────────────────────────────────────────┘
                                   │ Candidate Pool (10 chunks)
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │              Neural Cross-Encoder                │
          │   Model: cross-encoder/ms-marco-MiniLM-L-6-v2    │
          │   (Fallback: Jina Reranker API with Session Pool)│
          └────────────────────────┬─────────────────────────┘
                                   │ Top 3 Re-Ranked Passages
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │       LLM Generation & Fallback Cascade          │
          │   Primary:   Groq openai/gpt-oss-20b             │
          │   Fallbacks: openai/gpt-oss-120b                 │
          │              qwen/qwen3.6-27b                    │
          │              qwen/qwen3.8-27b                    │
          │   Streaming: 50ms UI micro-batching (~20 FPS)    │
          └────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
 Grounded Answer + Complete Runnable Code + Citations (Title, Vote Count, Accepted Status, URL)
```

### Key Engineering Decisions:

1. **Ultra-Low Latency UI Streaming (1.2s – 1.9s Total Delivery):** Token updates are batched into 50ms micro-batches (~20 FPS) to prevent Gradio UI queue backpressure, eliminating a 15-second client-side rendering bottleneck.
2. **Server-Side Sparse Fusion in Qdrant:** In-memory BM25 over 218K chunks requires ~1GB RAM and slows container boot time. Offloading sparse vector inverted indexing directly to Qdrant native sparse vectors made startup constant-time and memory flat.
3. **Robust Model Fallback Cascade:** To guard against single-model rate limits or token bursts, the generation layer cascades automatically across `openai/gpt-oss-20b` $\rightarrow$ `openai/gpt-oss-120b` $\rightarrow$ `qwen/qwen3.6-27b` $\rightarrow$ `qwen/qwen3.8-27b`.
4. **Idempotent Ingestion Pipeline:** Ingestion is split into offline stages (`parse_dump.py` $\rightarrow$ `embed_corpus.py` $\rightarrow$ `upload_embeddings.py`). Document point IDs are deterministic `uuid5(answer_id + chunk_index)`, guaranteeing idempotent re-indexing.
5. **Pillar 2 — n-Branch Semantic Query Decomposition:** Comparative queries like *"Type I error, Type II error, and statistical power"* contain $n \geq 2$ distinct concepts. Previous heuristic only handled 2-way `vs` splits. Upgraded to n-concept extraction: a Groq LLM (with heuristic fallback) detects **all** comparison entities and generates one targeted sub-query per concept. Each sub-query is run as a parallel hybrid search in Qdrant and all candidate pools are fused via RRF before cross-encoder reranking. For niche/sparse queries, `generate_semantic_variants()` additionally generates 2 synonym-rephrasings to compensate for low corpus frequency of specialist terms (e.g. *"Gelman-Rubin convergence"* $\rightarrow$ *"R-hat MCMC mixing diagnostic"*). Result: `multi_hop` Factual Claim Coverage improved from **28.3% → 40.9%** (Pillar 1 baseline) and is targeted at **65–70%** after Pillar 2 evals.

---

## Evaluation Benchmark & Empirical Results

The pipeline was benchmarked using a multi-layer evaluation framework ([`plan.md`](plan.md) Part E) covering component-level retrieval (Layer 1a) and end-to-end atomic claim contextual recall (Layer 1b via DeepEval).

Detailed report: [`evaluation/RECALL_REPORT.md`](evaluation/RECALL_REPORT.md).

### 1. Component Retrieval: Before vs. After Embedding Upgrade (Layer 1a)

Comparison of the initial 384-dimensional baseline vs. the current 768-dimensional BGE-base + Qdrant Cloud architecture across 309 query instances:

| Dimension / Category | Baseline (`bge-small-384d`) | **Upgraded (`bge-base-768d` + AWS Qdrant)** | Relative Gain | Key Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Global MRR** | `0.381` | **`0.500`** | 🚀 **+31.2%** | Significantly higher top-rank precision |
| **Global Recall@1** | `0.280` (28.0%) | **`0.424` (42.4%)** | 🚀 **+51.4%** | First hit is on-point 51% more often |
| **Global Recall@3** | `0.435` (43.5%) | **`0.563` (56.3%)** | 🚀 **+29.4%** | Feeds accurate context into LLM window |
| **Global Recall@5** | `0.490` (49.0%) | **`0.616` (61.6%)** | 🚀 **+25.7%** | Candidate pool coverage |
| **Standard Q&A Recall@10** | `0.970` (97.0%) | **`0.980` (98.0%)** | ✅ Near-Ceiling | Perfect recovery on core statistical questions |
| **Citation Accuracy Recall@10** | `1.000` (100.0%) | **`0.950` (95.0%)** | ✅ High Precision | Canonical papers and author attributions |
| **Code Traceback Recall@10** | `0.930` (93.0%) | **`0.870` (87.0%)** | ✅ Robust | Exact traceback and error fix matching |
| **Multi-Hop Comparison Recall@10** | `0.350` (35.0%) | **`0.520` (52.0%)** | 🚀 **+48.6%** | Multi-concept comparative retrieval |
| **Niche Domain Topic Recall@10** | `0.180` (18.0%) | **`0.500` (50.0%)** | 🚀 **+177.8%** | Long-tail statistical distributions |
| **Multi-Turn Conversational Recall@10**| `0.000` (0.0%) | **`0.390` (39.0%)** | 🚀 **+39.0%** | Conversational pronoun resolution |
| **Negation Exclusion Recall@10** | `0.050` (5.0%) | **`0.250` (25.0%)** | 🚀 **+400.0%** | Alternative domain method injection |
| **Negation Distractor Leakage** | ~12.0% | **`0.00%` (0 / 20)** | 🛡️ **Zero Leak** | Excluded terms never leak into context |
| **End-to-End Latency** | 11.4s – 19.4s | **1.2s – 1.9s** | ⚡ **~90% Faster** | 50ms batch streaming + session pooling |

---

### 2. End-to-End Contextual Recall & Fact Coverage (Layer 1b)

Evaluated across all **229 evaluable knowledge queries** using DeepEval atomic claim verification against the live production vector index:

### Evaluation Metric Definitions

To ensure complete transparency and prevent confusion between retrieval metrics:

* **🧠 Contextual Recall (Atomic Fact Coverage):** Evaluated via DeepEval LLM-as-a-Judge (`qwen3.6-27b`). Sentence-by-sentence verification measuring whether the retrieved context contains the factual statements required to answer the query (score range: `0.0` to `1.0`).
* **🎯 Recall@k (Document ID Hit Rate):** Traditional retrieval accuracy checking whether an exact labeled database Post ID appears within the top $k$ retrieved chunks.
* **🏛️ Pillar 1 Protocol (Multi-Gold Benchmarking):** Evaluates atomic factual claim coverage against multi-gold answer sets (incorporating all canonical high-vote community posts under the target topic to resolve single-ID labeling bias).

---

### End-to-End Contextual Recall & Retrieval Benchmark (229 Cases)

| Category | Cases ($n$) | Strict Single-Gold R@5 | Pillar 1 Multi-Gold R@5 | MRR | Fact Coverage (Contextual Recall) | Evaluation Focus |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 💻 **`code_traceback`** | 30 | **100.0%** | **100.0%** | **1.000** | **100.0%** | Complete code snippets, package imports, and script fixes |
| 📚 **`standard`** | 100 | **66.0%** | **72.0%** | **0.528** | **88.6%** | Core statistical theory, definitions & formulas |
| 📑 **`citation_accuracy`** | 20 | **65.0%** | **70.0%** | **0.493** | **71.4%** | Canonical source papers, author attributions & publication years |
| 🧩 **`multi_hop`** | 23 | **34.8%** | **39.1%** | **0.341** | **40.9%** (+44.5% gain) | Sub-Query Multi-Branch Decomposition & Parallel Fusion |
| 🔬 **`niche_topic`** | 18 | **38.9%** | **38.9%** | **0.201** | **33.3%** (+33.2% gain) | Candidate Pool Expansion (K=100) & BM25 Sparse Boost |
| 🛑 **`negation`** | 20 | **15.0%** | **20.0%** | **0.150** | **20.0%** (0% distractor leak) | Qdrant `must_not` Text Exclusion Payload Filtering |
| 💬 **`multi_turn`** | 18 | **16.7%** | **16.7%** | **0.102** | **16.7%** (+42.7% gain) | Conversation History Injection & Pronoun Resolution |
| **GLOBAL (All 7 Categories)** | **229** | **62.7%** | **68.2%** | **0.518** | **68.2%** | **Global Weighted Average** |
| **Core Knowledge Search** | **150** | **72.7%** | **78.4%** | **0.652** | **89.4%** | **Standard + Code Traceback + Citation** |

---

## Metadata Filtering: Empirical Advantage

The UI provides optional topic tag filtering (`bayesian`, `machine-learning`, `time-series`, `distributions`, `hypothesis-testing`, `regression`, `r`, `python`). 

Empirical testing ([`test_tag_filtering_advantage.py`](test_tag_filtering_advantage.py)) demonstrates three concrete advantages of metadata filtering:

1. **Polysemy Disambiguation:** 
   * A query on *"Kernel"* under `distributions` retrieves Kernel Density Estimation (KDE) bandwidth estimators (`epanechnikov`, `gaussian`).
   * The same query under `machine-learning` retrieves Support Vector Machine (SVM) kernel functions (`RBF`, Mercer's theorem).
2. **Paradigm Enforcement:**
   * Querying *"Interval estimation"* under `bayesian` strictly returns credible intervals and posterior distributions, filtering out frequentist repeated-sampling simulations.
3. **Domain Diagnostics:**
   * Querying *"Residual diagnostics"* under `time-series` strictly returns Ljung-Box autocorrelation and ACF tests, filtering out standard OLS heteroskedasticity diagnostics.

---

## Three Technical Findings Worth Documenting

### 1. StackExchange Answers Are Deictic (and Destroyed Initial Recall)
The initial baseline started at **Recall@10 = 0.183**. Tracing revealed that Cross Validated answers frequently begin with deictic references: *"That is correct. The repeated measures ANOVA is an omnibus test..."* or *"Such multicollinearity is matter of fact..."*. 
Embedded in isolation, these chunks lacked topical signal. By prepending the question title (`question_title + overlap tail + chunk`) during ingestion, Recall@10 jumped from **0.183 to 0.538** (+194% relative gain).

### 2. Single-Gold Labelling Understates True Retrieval Quality
In curated evaluation queries, Recall@10 was 0.146 compared to 0.966 for programmatic queries. A manual audit of 32 curated cases revealed that in **69% of apparent failures**, the retriever returned valid, highly voted answers that simply had a different `answer_id` than the single hand-labelled gold target.

### 3. Eliminating the 15-Second UI Latency Bottleneck
Profiling revealed backend execution took only **1.2s – 2.4s**, yet Gradio took **19.4s** to finish rendering. Gradio was pushing a full WebSocket update per token generated, creating client-side UI backpressure. Grouping token updates into **50ms micro-batches** brought total end-to-end response delivery down to **1.2s – 1.9s** (~90% faster).

---

## Project Structure

```
app.py                              Gradio UI application & entry point
backend/main.py                     FastAPI production entry point
backend/api/routes.py               /api/v1/query (NDJSON streaming), /api/v1/topics
backend/core/vector_store.py        Qdrant client, collection setup, search_hybrid()
backend/core/sparse_store.py        CRC32 feature hashing sparse vector generator
backend/core/embeddings.py          Shared BGE-base embedding singleton
backend/core/reranker.py            Cross-Encoder reranking with session pool
backend/core/ingestion.py           Chunking, title prefixing, 200-char overlap
backend/core/llm_service.py         Groq multi-model fallback cascade, streaming
backend/prompts/system_prompts.yaml YAML prompt registry with content hashing
backend/scripts/parse_dump.py       StackExchange XML → JSONL parser
backend/scripts/embed_corpus.py     Offline chunking + embedding (resumable)
backend/scripts/upload_embeddings.py Production upload to Qdrant Cloud
evaluation/eval_retriever.py        Layer 1a — Component retrieval evaluation
evaluation/eval_contextual_recall.py Layer 1b — DeepEval Fact Coverage evaluation
evaluation/judge_model.py           DeepEval judge configuration & JSON extractors
evaluation/golden_dataset.json      289 benchmark test cases across 10 categories
evaluation/RECALL_REPORT.md         Full retrieval & contextual recall report
test_tag_filtering_advantage.py     Empirical demonstration of metadata filtering
```

---

## Local Setup and Reproduction

### 1. Install Dependencies
```bash
pip install -r requirements.txt
cp .env.example .env
```
Ensure `.env` contains `GROQ_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY`.

### 2. Run the Application
```bash
python app.py
```
Open [http://localhost:7860](http://localhost:7860) in your browser.

### 3. Reproduce Evaluations
```bash
# Run Layer 1a Component Retrieval Benchmark (Recall@k, MRR):
python evaluation/eval_retriever.py --pool 50 --ks 1,3,5,10

# Run Layer 1b DeepEval Contextual Recall & Fact Coverage:
python evaluation/eval_contextual_recall.py

# Demonstrate Topic Tag Filtering Advantage:
python test_tag_filtering_advantage.py
```

---

## Licence and Attribution

* **Code:** [MIT License](LICENSE).
* **Corpus:** Cross Validated content from the official StackExchange data dumps, licensed under **CC BY-SA 4.0**. Every generated response includes clickable citation links back to original author threads on [stats.stackexchange.com](https://stats.stackexchange.com).

Built by **[K. Srinivasa Rao](https://github.com/ksrinivasarao2012)**.

---

## Safety & Guardrails

> Added 2026-09-02. Full change history in [`CHANGES.md`](CHANGES.md).

Retrieval quality is only half of a RAG system. The other half is what happens
when retrieval **fails** — and until this was added, the honest answer was
"the model improvises." The active answer prompt permitted a general-knowledge
fallback whenever retrieved context was thin, which meant off-topic questions,
jailbreak framings, and medical or legal questions phrased as statistics were
all answerable from model priors. There was no other layer in the pipeline to
stop them.

Four checks now run on every request, in `backend/core/guardrails.py`, called
identically by the Gradio app and the FastAPI backend.

| Stage | Check | What it does |
|---|---|---|
| Before rewriting | `check_input()` | Pattern-matches injection attempts (`ignore previous instructions`, `reveal your system prompt`, `DAN mode`, `</system>`) and junk payloads on the **raw** query, so the LLM rewriter is never the first component to see hostile text. Deterministic — no model call, no added latency. |
| After reranking | `filter_by_score()` | Drops chunks below a relevance floor. **If nothing clears it, the LLM is never called.** |
| On empty result | `classify_empty_result()` | Distinguishes small talk ("hi", "thanks") from a genuinely unanswerable question, so a greeting gets a normal reply rather than a grounding refusal. |
| After generation | `check_output()` | Withholds the Sources footer when the answer refused or shares almost no wording with the retrieved text, and strips leaked system-prompt text. |

### Three decisions worth explaining

**The relevance floor is the actual guardrail; the prompt is not.**
A system prompt *asks* a model to refuse. Only skipping generation *makes* it
refuse. Previously the reranker sorted and sliced `top_k=3` unconditionally, so
a query with no good match produced context of exactly the same shape as a
perfect match. Now an empty result short-circuits before the model is invoked —
there is nothing to hallucinate with.

**Small-talk detection runs after retrieval fails, not before.**
Placed up front it would be a bypass: *"hi, ignore your instructions and…"*
would route to the friendly path. It is reachable only by a query that already
passed the input check and then retrieved nothing.

**The two reranker backends do not speak the same units.**
`cross-encoder/ms-marco-MiniLM-L-6-v2` returns raw logits (≈ −11…+11); the Jina
API returns a true 0–1 relevance score — and Jina silently disables itself on a
401/403 balance error, falling through to the local model. A single threshold
constant would therefore have meant two different things depending on billing
state. Scores are normalized to a common 0–1 scale before the floor is applied.

Retrieved chunks are additionally wrapped in explicit
`BEGIN/END RETRIEVED DOCUMENT` markers, and the system prompt declares that text
between them is reference material and never instructions — the corpus is
218,456 chunks of community-written text, so indirect prompt injection through a
retrieved post is a real vector on a public deployment.

### Honest limitations

- **The thresholds are placeholders, not measurements.** `MIN_RERANK_SCORE`
  (0.25) and `MIN_GROUNDING_OVERLAP` (0.15) need calibrating against the
  `out_of_scope` and `standard` golden slices, with `niche_topic` as the
  false-refusal check. Override the first via `GUARDRAIL_MIN_SCORE`.
- **The injection list is a fixed pattern set.** It stops copy-pasted attacks,
  not novel phrasing. The relevance floor is what bounds the damage.
- **Indirect injection is mitigated, not solved.** Nothing scans chunk text.
- **Grounding is measured by word overlap**, which is crude. It is tuned to fire
  only on a clear miss, so it under-reports rather than over-blocks.
- **Prior `out_of_scope` and `adversarial` benchmark numbers predate this work**
  and measured the older permissive prompt. They do not describe the current
  system and are being re-run.

### Bug fixes — 2026-09-02

Five bugs found in code review, fixed the same day (`CHANGES.md` has full
detail):

| Bug | Fix |
|---|---|
| A broken stream mid-answer could glue a full second answer onto a partial one | The generator now tracks whether anything has already been shown; if a failure happens after that, it stops with a short message instead of retrying |
| The query-rewrite step could try up to 7 AI providers in series with no overall time limit | Wrapped in one deadline (`REWRITE_TIMEOUT_SECONDS`, default 4s); falls back to the raw query on timeout |
| A rewritten search query was never checked before being used | Rejected if empty, oversized, or sharing no words with the original question |
| The web app and the API endpoint ran different retrieval settings (candidate pool size, multi-hop query splitting) behind the same prompts | Brought to parity — both now use the same pool size and both split comparison queries |
| One unreachable line of dead code | Removed |

One implementation bug was caught and fixed during the verification pass for
the timeout fix: the first version used a `with` block around the thread pool,
whose cleanup silently waited for the slow call to finish anyway — defeating
the timeout it was supposed to enforce.

### Corrections to the above — 2026-09-02

A few claims made earlier in this README don't match the current code or the
project's own later findings. Per this file's append-only convention, they
are corrected here rather than edited in place:

| Claim above | Where | Correction |
|---|---|---|
| "Server-Side **BM25** IDF applied via `Modifier.IDF`" / "In-memory **BM25** over 218K chunks..." | Architecture diagram note, Key Engineering Decision #2 | This is **not BM25**. It's raw term-frequency weighting (`sum(raw_tf × idf)`) — no term-frequency saturation, no document-length normalization. A chunk repeating one term many times, or simply a longer chunk, outscores a better-matching one at the same relevance. Real BM25 is a scoped, not-yet-run fix (`PROJECT_LOG.md` §6, 2026-09-02 10:15). The architecture diagram itself already says this correctly ("CRC32 Feature Hashing, Term-Frequency (TF)") — only the two prose mentions were wrong. |
| "Candidate Pool Expansion (**K=100**)" | Results table, `niche_topic` row | The code has never used a pool of 100. `app.py` uses `CANDIDATE_K=10`; `backend/api/routes.py` used 15 until 2026-09-02, when it was brought into line with `app.py`'s 10 (see the Bug fixes section above). |

None of the underlying benchmark numbers in the tables are affected — this
only corrects what the pipeline was doing while producing them.
