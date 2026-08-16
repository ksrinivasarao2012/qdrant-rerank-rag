---
title: Cross Validated RAG Assistant
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Cross Validated RAG Assistant

Ask a statistics or machine-learning question in plain English. Get an answer built
from real [Cross Validated](https://stats.stackexchange.com) discussions, with links
back to the source threads.

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue?style=flat-square)](https://huggingface.co/spaces/Srinivasa12/rag-portfolio)
[![Qdrant](https://img.shields.io/badge/Qdrant-hybrid%20search-red?style=flat-square)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/code-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Corpus: CC BY-SA](https://img.shields.io/badge/corpus-CC%20BY--SA-lightgrey.svg?style=flat-square)](https://stackoverflow.com/help/licensing)

**93,455 answers · 218,456 indexed chunks · evaluated on 294 labelled cases**

---

## Why this corpus

Six other corpora were evaluated and rejected before this one. The decision log is in
[`plan.md`](plan.md) Part A; the short version is that most candidate corpora don't
actually need retrieval.

Four editions of the MCC Laws of Cricket, for example, extract to roughly **230K
tokens** against a **131K-token** context window. When the corpus nearly fits in one
prompt, retrieval isn't solving a problem — a good prompt is. Nike/Adidas annual
reports failed for a different reason: Adidas files a 20-F under IFRS with a December
year-end, Nike a 10-K under US GAAP ending in May, so every "compare FY23 margins"
question would have compared different periods under different accounting standards.
The ground truth itself would have been wrong.

Cross Validated passes the three tests those failures produced:

| Test | Why it matters | Result |
|---|---|---|
| **Doesn't fit in a context window** | Otherwise you need a prompt, not a retriever | ~100× a context window ✅ |
| **Vocabulary mismatch** | The only thing dense embeddings do that keyword search can't | Severe, and it's the site's defining search weakness ✅ |
| **No structure to exploit** | Stable IDs tempt you into building a database instead | No IDs, no hierarchy, no versions ✅ |

Someone types *"my model gets 99% on training data but 60% on new data"*. The answer
says **overfitting**, **regularisation**, **bias–variance** — words that appear nowhere
in the question. That gap is the whole reason to build this.

And critically: `AcceptedAnswerId` in the StackExchange dump means **the correct answer
is already labelled, ~200,000 times over.** Ground truth is a download, not a week of
manual annotation.

---

## Architecture

```
user question
     │
     ├─ conversational follow-up? → prepend the last 2 turns to the search query
     │                              ("how do I prevent it?" is unsearchable alone)
     ▼
┌─────────────────── Qdrant, one round trip ───────────────────┐
│  dense: BAAI/bge-small-en-v1.5 (384-d, cosine)               │
│  sparse: CRC32 feature hashing (2^18), TF; IDF applied       │
│          server-side via Modifier.IDF                        │
│  fused server-side by Reciprocal Rank Fusion                 │
└──────────────────────────────┬───────────────────────────────┘
                               │ 15 candidates
                               ▼
              cross-encoder/ms-marco-MiniLM-L-6-v2
                               │ top 3
                               ▼
              Groq · openai/gpt-oss-20b · streamed
                               │
                               ▼
     answer + citations (thread title, vote count, ✓ accepted, link)
```

**Sparse search runs inside the database.** An in-process BM25 index over 218K chunks
is roughly 1 GB of RAM and has to be rebuilt from a full collection scroll on every
boot — it does not survive a free-tier container. Moving to Qdrant native sparse
vectors made startup constant-time and memory flat.

**Ingestion is strictly offline.** `parse_dump.py` → `embed_corpus.py` →
`upload_embeddings.py`. Nothing embeds at request time, so a restart never re-indexes.
Embedding and upload are deliberately separate phases: an earlier single-pass script
lost whole batches to transient network errors with no record of what was dropped.
Point IDs are `uuid5(answer_id + chunk_index)`, so re-uploading is an idempotent
overwrite rather than a duplicate.

---

## Results

Layer 1a (retriever in isolation) of the four-layer evaluation framework in
[`plan.md`](plan.md) Part E. Reference-based, deterministic, no LLM judge.
Full report: [`evaluation/RECALL_REPORT.md`](evaluation/RECALL_REPORT.md).

**294 labelled cases → 314 query instances**, across 10 categories: standard,
paraphrase groups, multi-hop, multi-turn, negation, niche topics, code/traceback,
citation accuracy, out-of-scope and adversarial.

```
n=314   MRR=0.381
recall@10 =0.538   qrecall@10 =0.596
recall@50 =0.656   qrecall@50 =0.729
recall@100=0.666   qrecall@100=0.742
```

`recall@k` requires the exact accepted answer. `qrecall@k` accepts any answer from the
right thread — Cross Validated questions routinely have several good answers, and the
generator is served equally well by any of them.

### The number that matters, and its caveat

| gold construction | n | recall@10 | qrecall@10 |
|---|---:|---:|---:|
| **Programmatic** — the query *is* the gold post's own title | 150 | **0.966** | **0.987** |
| **Curated** — keyword-matched post, hand-written query | 164 | 0.146 | 0.240 |

The 0.966 is close to a ceiling *and flat across k=10/50/100* — nothing is lost to
ranking. But it is near-tautological by construction: the query is byte-identical to
the gold post's title, which is also a prefix of its embedded text. It is a sanity
ceiling, not evidence of paraphrase-level ability.

The curated 0.146 is not what it looks like either — see below.

### Component ablations

| change | recall@10 | MRR | note |
|---|---:|---:|---|
| Prepending `question_title` to embedded text | 0.183 → **0.538** | 0.124 → 0.381 | largest single improvement |
| Cross-encoder reranking | 0.465 → **0.525** | 0.289 → 0.391 | identical pool, identical k |
| LLM query rewriting | 0.465 → **0.417** | 0.289 → 0.311 | **negative** — see below |
| `BAAI/bge-reranker-base` | 0.525 → 0.525 | 0.391 → 0.360 | 2× slower, no gain |
| Conversational history in the search query | multi-turn r@10 0.00 → **0.18** | 0.019 → 0.052 | n=22 slice |

Confounds are stated per-comparison in the full report; the rewriting and reranker-swap
rows were measured at different pool sizes and are directional, not exact.

---

## Three findings worth the write-up

### 1. StackExchange answers are deictic, and it destroyed recall

The pipeline started at **recall@10 = 0.183**. `standard` — the easiest category —
sat at 0.28, which is not "needs tuning", it's "something is structurally wrong".

Ruled out with evidence, in order: ID type mismatch (already handled), a golden set
built from a different corpus snapshot (all gold IDs verified present), and stale
points in the live collection (a clean local index reproduced the numbers to three
decimal places). Then stage-by-stage tracing: when dense-only search missed the gold
answer in the top 50, **no downstream stage ever recovered it.** The loss was at
embedding time.

The cause: Cross Validated answers are frequently deictic replies —
*"That is correct. The repeated measures ANOVA is an omnibus test…"*,
*"Such multicollinearity is matter of fact…"*. Embedded alone they carry almost no
topical signal. `question_title` was stored in payload metadata but never reached the
embedder.

Prepending the title took recall@10 from **0.183 to 0.538**. Chunk overlap (29% of
answers produce more than one chunk) and a separate `display_text` field for citation
snippets shipped alongside it.

### 2. Most of the remaining "failures" were mislabelled, not missed

Curated categories sat at 0.146 while programmatic ones hit 0.966. Reading the actual
output for 32 cases by hand:

| cause | share |
|---|---:|
| **Gold label wrong or arbitrary; retrieval returned good answers** | **69%** |
| Negation semantics not honoured | 12% |
| Genuinely obscure topic | 9% |
| Unresolved pronoun in a follow-up | 9% |

*"Clustering methods that do not require specifying the number of clusters k?"* — the
corpus contains a 46-vote post titled almost exactly that, and retrieval returned it at
**rank 2**. The labelled gold was *"Clustering of large, heavy-tailed dataset"*.

Root cause was in the dataset builder, not the retriever: keyword matching compared raw
lowercase substrings, so `"cross-validation"` failed to match the title
`"Cross Validation - purpose, need and utility"`, and candidates were taken in corpus
order rather than ranked by community signal. Obscure low-vote posts became ground
truth while the canonical thread went unlabelled.

Fixed by normalising punctuation before matching and ranking candidates by
(accepted, vote score). The residual is a harder problem: on a corpus with dozens of
good answers per topic, **single-gold labelling cannot be repaired by better keyword
selection** — two rounds of repairs left paraphrase recall unchanged at 0.15. That
needs multi-gold labelling or human relevance judgement, which is the outstanding work.

### 3. End-to-End Latency Reduction: Fixing a 15-second UI delay in plain English

When users asked questions like *"what is k means"* or *"what is ml"*, the application took **19.4 seconds** and **11.4 seconds** to finish responding.

We conducted a deep latency audit to find out why:

#### The Investigation: Backend vs. UI Speed
When we profiled the backend AI components in isolation, we discovered a surprising result:
- **Vector Search (Qdrant Cloud)**: ~0.35 seconds
- **Re-Ranking (Jina API)**: ~0.50 – 0.90 seconds
- **AI Answer Generation (Groq)**: ~0.35 seconds
- **Actual Backend Execution Time**: Only **1.2 to 2.4 seconds**!

So why was the user waiting 19 seconds? We identified **3 hidden bottlenecks**:

1. **The UI Traffic Jam (10–15 Seconds Lost)**  
   *Imagine a waiter running from the kitchen to your table every time the chef places a single grain of rice on the plate!* That is what Gradio was doing. It was sending a full data update to the web browser for **every single letter** generated by the AI. For a 300-word answer, Gradio attempted 300 network updates, completely clogging the web interface queue.

2. **Unnecessary AI Query Rewriting (1.3–5.0 Seconds Lost)**  
   When asking a question in a multi-turn chat (like *"what is ml"*), the system tried to rewrite the query by calling remote AI services to see if it was a follow-up. When those services hit rate limits, the app paused for 5 seconds retrying. But standalone questions like *"what is ml"* do not need rewriting—the call was completely wasted.

3. **Repeated Network Handshakes (0.5 Seconds Lost)**  
   Every time documents were sent to be re-scored, the app opened a new network connection, performed SSL security handshakes, and closed it. Doing this on every request added unnecessary delay.

---

#### The Solutions (How we brought latency down to 1.2s – 1.9s):

- **Smart UI Batch Updates (Yielding every 50ms)**:  
  Instead of refreshing the web page on every single letter, we group tokens into smooth updates every 50ms (~20 updates per second). The response streams live like a typewriter, and the 15-second UI traffic jam disappears.
- **Smart Query Rewrite Guard & Ultra-Fast Rewriter**:  
  We added a quick rule: does the query contain words like *"it"*, *"they"*, *"this"*, or *"that"* pointing to previous answers? If not (e.g. *"what is ml"*), we skip the extra AI rewriter call completely (**saving 1.5s – 5.0s**). When a follow-up query *does* contain pronouns (e.g. *"what is the problem with it"*), we route query resolution directly through Groq (`llama-3.3-70b-versatile`), which resolves the pronoun referent in **0.20 seconds** instead of hanging on rate-limited remote fallback APIs. Follow-up query time dropped from **20.7s down to 1.9s**!
- **Persistent Network Sessions**:  
  We keep the HTTP network connection open across requests instead of opening and closing a new socket every time.
- **Parallel Search**:  
  Dense vector search and keyword search now run concurrently at the exact same time using multi-threading.
- **Shorter Context Prompts**:  
  We cap context document snippets at 1500 characters so the AI processes less text, delivering the first word of the answer in just **0.20 seconds**.

#### Final Speed Comparison:
| Metric | Before Optimization | After Optimization | Improvement |
| :--- | :--- | :--- | :--- |
| **Backend AI Processing** | 2.40s – 3.58s | **1.21s – 1.90s** | ~50% faster |
| **User Interface Delivery** | 11.4s – 19.4s | **1.2s – 1.9s** | **85% – 90% faster** |

---

## Known limitations

- **Exclusion queries do not work.** `negation` recall@100 = 0.10 — only 2 of 20 gold
  answers appear anywhere in a 100-candidate pool. *"Forecast time series without
  ARIMA"* has its answer under **exponential smoothing**, a term absent from the query
  in both lexical and embedding space. No reranker or embedding upgrade fixes this;
  recovering it requires inferring the alternative to the excluded technique, which is
  reasoning, not similarity. Query expansion over a technique map is the real fix.
- **Multi-hop comparisons retrieve one side.** `both@10 = 0.00` — for *"compare AIC and
  BIC"* the system finds one topic, not both. Query decomposition raises MRR from 0.114
  to 0.180 on that slice and is measured but not yet integrated.
- **Conversational history helps follow-ups and would hurt topic switches.** The golden
  set contains no topic-switch cases, so that cost is unmeasured.
- **Single-gold ground truth understates quality** by roughly the 69% above.
- **n is 20–30 per category.** Single-category differences under ~10 points are noise.
- **Accepted ≠ best.** The asker's choice is one person's judgement.
- **No runtime guardrails.** The adversarial category *tests* for prompt-injection
  resistance; nothing enforces it beyond the strict-grounding system prompt.

---

## Repository

```
app.py                              Gradio UI (Hugging Face Spaces entry point)
backend/main.py                     FastAPI entry point
backend/api/routes.py               /api/v1/query (NDJSON streaming), /api/v1/topics
backend/core/vector_store.py        Qdrant client, collection setup, search_hybrid()
backend/core/sparse_store.py        feature-hashing sparse vector generator
backend/core/embeddings.py          shared embedding singleton
backend/core/reranker.py            cross-encoder reranking
backend/core/ingestion.py           chunking, title prefix, overlap, display_text
backend/core/llm_service.py         Groq/Gemini clients, prompt assembly, streaming
backend/core/prompts.py             YAML prompt registry with variant fingerprinting
backend/scripts/parse_dump.py       StackExchange XML → JSONL
backend/scripts/embed_corpus.py     phase 1: chunk + embed locally (resumable)
backend/scripts/upload_embeddings.py phase 2: upload to Qdrant (idempotent, retrying)
backend/scripts/build_golden_dataset.py  builds the 294-case evaluation set
evaluation/eval_retriever.py        layer 1a — recall/precision/MRR ablations
evaluation/eval_generator.py        layer 1b — generator on hand-fed golden context
evaluation/eval_pipeline.py         layer 2  — RAG triad on the live stack
evaluation/eval_application.py      layer 3  — correctness, completeness, style
evaluation/tune_retrieval.py        fast category-scoped experiments
evaluation/RECALL_REPORT.md         full retrieval evaluation report
plan.md                             corpus decision log, architecture, eval framework
```

**Prompts live in YAML**, not in Python. Each variant is content-hashed, so editing a
prompt changes its fingerprint and stale evaluation rows become visibly attributable to
text that no longer exists.

---

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env      # GROQ_API_KEY, GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY
python app.py             # http://127.0.0.1:7860
```

Building the index from scratch:

```bash
# 1. download stats.stackexchange.com.7z from archive.org, extract Posts.xml
#    into data/raw/
python backend/scripts/parse_dump.py        # → data/processed/posts.jsonl
python backend/scripts/embed_corpus.py      # local only, no network, resumable
python backend/scripts/upload_embeddings.py # network only, idempotent
```

Reproducing the evaluation:

```bash
python backend/scripts/build_golden_dataset.py
python evaluation/eval_retriever.py --local --pool 100 --ks 10,50,100
```

---

## Licence and attribution

Code is MIT. The corpus is Cross Validated content from the official StackExchange data
dumps, licensed **CC BY-SA**; every answer links back to its source thread, and citation
snippets are capped at 300 characters. Contributions belong to their original authors on
[stats.stackexchange.com](https://stats.stackexchange.com).

Built by [K. Srinivasa Rao](https://github.com/ksrinivasarao2012).
