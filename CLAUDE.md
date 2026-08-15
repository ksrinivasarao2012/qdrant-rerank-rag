# CLAUDE.md

## ⚠️ TOP PRIORITY — READ FIRST

**Do not overcook, over-engineer, or overcomplicate anything.**

Always prefer the simplest fix that solves the actual problem in front of you and doesn't create future risk. Concretely:

- No speculative abstractions, config flags, or "future-proofing" for needs that don't exist yet.
- No rewriting working code while fixing an unrelated bug.
- No adding new dependencies, services, or layers unless the task genuinely requires them.
- If a minimal fix and an elaborate fix both solve the problem, do the minimal one.
- When reviewing code, call out real bugs plainly — don't invent problems to justify a bigger change.
- If unsure whether something is overkill, it probably is — ask or default to the smaller change.

This rule overrides general instincts toward "best practice" polish. This is a personal portfolio project, not production infrastructure — correctness and simplicity beat cleverness every time.

---

## Project Overview

**Portfolio RAG Assistant** — a retrieval-augmented generation system answering statistics/ML questions grounded in **Cross Validated** (stats.stackexchange.com), a StackExchange site. Licensed CC-BY-SA. This is the second incarnation of the project — several earlier corpus ideas (Apple/Microsoft filings, Nike/Adidas 10-Ks, TCS/Infosys annual reports, MCC Cricket Laws) were evaluated and rejected because they didn't actually require retrieval (small enough to fit in an LLM context window), had no real vocabulary mismatch to exploit, or had licensing/consistency problems. See `plan.md` for the full decision log (Part A) and the criteria used to select Cross Validated instead (Part B).

## Architecture

**Retrieval pipeline:**
1. Query rewriting (Gemini) using chat history, when present.
2. **Stage 1 — Native hybrid search in Qdrant**: dense vectors (`BAAI/bge-small-en-v1.5`, 384-dim, cosine) + sparse vectors (custom feature-hashing generator, `backend/core/sparse_store.py`), fused server-side via RRF. No in-process BM25 index — Qdrant applies IDF itself via `Modifier.IDF` on the sparse vector config.
3. **Stage 2 — Cross-encoder reranking** (`backend/core/reranker.py`) narrows candidates down to top-k.
4. **Answer generation** via Groq (`openai/gpt-oss-20b`, `reasoning_effort: low`), streamed token-by-token.

**Data source:** StackExchange XML dump (`Posts.xml`), parsed offline by `backend/scripts/parse_dump.py` into `data/processed/posts.jsonl`, then loaded into Qdrant by a seed script. Top 50,000 questions by score are kept; their answers are chunked (`backend/core/ingestion.py`, paragraph-boundary chunking) and embedded. `AcceptedAnswerId` from the dump provides a free, pre-labeled ground-truth set for retrieval evaluation.

**No PDF/document upload.** This was removed — the corpus is entirely offline-ingested StackExchange data. Ingestion is deliberately not a runtime path, so a server restart never re-embeds anything.

**Prompt registry:** `backend/prompts/system_prompts.yaml` + `backend/core/prompts.py`. YAML-based variants with fingerprinting for reproducibility. Active variants: `gptoss_strict_v1` (answers), `describe_qwen_v1` (vision, largely unused now). Retired variants are kept (not deleted) for historical eval reference, each pinned to its original model.

## Key Files

| File | Purpose |
|---|---|
| `app.py` | Gradio frontend |
| `backend/api/routes.py` | FastAPI backend (`/api/v1/query`, `/api/v1/topics`) |
| `backend/core/vector_store.py` | Qdrant client, collection setup, `search_hybrid()` |
| `backend/core/sparse_store.py` | Feature-hashing sparse vector generator |
| `backend/core/llm_service.py` | Groq/Gemini clients, prompt assembly, streaming |
| `backend/core/reranker.py` | Cross-encoder reranking |
| `backend/core/embeddings.py` | Shared singleton dense embedding model |
| `backend/core/ingestion.py` | Text chunking (`DocumentProcessor`) |
| `backend/core/prompts.py` | Prompt registry loader/fingerprinting |
| `backend/scripts/parse_dump.py` | StackExchange XML → JSONL parser |
| `backend/scripts/seed_corpus.py` | Loads `data/processed/posts.jsonl` into Qdrant (chunks + embeds + upserts). Written, not yet run. |
| `backend/main.py` | FastAPI app entrypoint (mounts `backend/api/routes.py`) |
| `backend/core/config.py` | Env/config loading |
| `verify_models.py` | Pre-flight check that Groq models are live and behaving correctly |
| `plan.md` | Master plan: corpus decision log, architecture, 5-layer eval framework, execution roadmap |

`frontend/` and `evaluation/` directories also exist at repo root — not yet wired into the active pipeline (`frontend/` predates the Gradio `app.py`; `evaluation/` predates the DeepEval decision). Check contents before relying on either.

## Operational Notes

- **Groq model deprecations**: `llama-3.1-8b-instant` was retired 2026-08-16. Current active models (`openai/gpt-oss-20b`, `qwen/qwen3.6-27b`) were verified live via `verify_models.py` — last run confirmed all checks pass: model live, reasoning tokens stay in the separate `reasoning` field (no leak into `content`), and the refusal phrase `"do not have enough information"` is intact (this exact string is load-bearing — `app.py`'s citation-suppression logic checks for it literally).
- **No network access to Groq from a sandboxed/CI environment** in some setups — if `verify_models.py` can't reach `api.groq.com`, run it locally instead of assuming the code is broken.
- **No hardcoded paths.** Scripts resolve paths relative to `Path(__file__)`, not hardcoded drive letters — keep it that way so things work outside one machine.
- **Imports use the `backend.*` convention** throughout (not `core.*`). Deployment configs (`Dockerfile`/`render.yaml`) must not add `--app-dir backend`, or imports break.
- **Citations** are built from StackExchange metadata (question title, vote score, accepted flag, thread URL) — there are no page numbers, unlike the old PDF-based version.

## Evaluation Framework

Defined in detail in `plan.md` Part E — five layers: Component (retriever/generator, reference-based), Pipeline (RAG triad: context relevance → faithfulness → answer relevance), Application (correctness/completeness/style/safety), Regression (re-run + diff, not new metrics), Online (live traffic monitoring, no golden answer, feeds back into golden set over time).

**Chosen eval library: DeepEval** (`deepeval` on PyPI). Dependency is now added to both `requirements.txt` and `backend/requirements.txt`, but no eval code has been written yet — nothing imports it. Reasoning: it has ready-made LLM-judge metrics that map directly onto the Pipeline layer's RAG triad (`ContextualRelevancyMetric`, `FaithfulnessMetric`, `AnswerRelevancyMetric`) plus retrieval metrics (contextual precision/recall) for the Component layer, so those don't need to be hand-rolled. Per the top-priority rule: don't wire this in speculatively — add the actual eval code when implementing the eval steps in `plan.md` Part F, not before.

## Project Status (update this section as work progresses)

This section exists so a fresh chat with no prior history can pick up exactly where things left off. Keep it current — when you finish a step, update it here.

**Done:**
- Groq model migration off `llama-3.1-8b-instant` (deprecated 2026-08-16) → `openai/gpt-oss-20b` (answers) / `qwen/qwen3.6-27b` (vision). Verified live via `verify_models.py` (user-run; sandbox cannot reach `api.groq.com`).
- Corpus decision: Cross Validated (stats.stackexchange.com) selected after rejecting 6 other candidates. Full reasoning in `plan.md` Part A/B.
- `backend/scripts/parse_dump.py`: two-pass StackExchange XML parser (top 50K questions by score, their answers) → `data/processed/posts.jsonl`. Memory-safe iterparse, no hardcoded paths.
- `backend/core/sparse_store.py`: custom feature-hashing sparse vector generator (`SPARSE_DIM = 262144`), TF-only — IDF is applied server-side by Qdrant (`Modifier.IDF`), not computed here.
- `backend/core/vector_store.py`: native Qdrant hybrid search (`search_hybrid()`) — dense (`BAAI/bge-small-en-v1.5`) + sparse, fused via RRF inside Qdrant. Replaces old in-process BM25.
- PDF/vision/table upload code fully removed from `app.py`, `backend/api/routes.py`, `backend/core/ingestion.py`. `HybridRetriever`/`backend/core/hybrid_search.py` deleted. `backend/scripts/download_corpus.py` (old TCS/Infosys PDF downloader) deleted. Dead deps (`pymupdf`, `rank-bm25`, `unstructured`) removed from both requirements files.
- Citations across `app.py` and `routes.py` rebuilt around StackExchange metadata (question title, score, accepted flag, thread URL) instead of PDF page numbers.
- This `CLAUDE.md` created.

- `backend/scripts/seed_corpus.py` written: reads `posts.jsonl`, chunks via `DocumentProcessor`, embeds, upserts into Qdrant in batches. Confirmed to exist and look complete on inspection.
- `data/processed/posts.jsonl` exists and has 88,074 answer records (parse_dump.py has been run against the real StackExchange dump, not just tested on synthetic data).
- `deepeval` added as a dependency in both `requirements.txt` and `backend/requirements.txt`.

**Decided, not yet implemented:**
- DeepEval as the eval library — dependency is installed, but no eval code exists yet (see Evaluation Framework above).

**Not started:**
- **Running `embed_corpus.py` → `upload_embeddings.py`.** This is the actual current blocker: the corpus has not been embedded/uploaded to Qdrant Cloud yet. The pipeline code is done (and split into a local embed phase + a network upload phase specifically to survive the network flakiness that corrupted an earlier `seed_corpus.py` run — see git history / conversation log for the full incident); the corpus has not been loaded.
- `plan.md` Part F execution steps 3–11 (beyond parsing + sparse/hybrid search, which are done): building the golden eval set from `AcceptedAnswerId`, wiring up DeepEval metrics, Component/Pipeline/Application/Regression/Online eval runs, CI wiring.
- **Conversational vs. out-of-scope routing.** Deliberately deferred until after the eval work above tells us how often it actually matters. Today the strict-grounding system prompt (`gptoss_strict_v1`) refuses anything not answerable from retrieved context — including plain conversational messages like "hi" or "thanks," which get treated the same as a genuinely off-topic question, since retrieval always runs regardless of what was typed. Planned fix: when a query triggers the out-of-scope/refusal path, classify it further (conversational/small-talk vs. genuinely off-topic vs. other) and let small-talk get a normal conversational reply instead of the strict refusal boilerplate. Do this after the eval framework can actually measure how common this is — don't build it speculatively.
- **Guardrails.** Currently the `adversarial` category in the golden dataset only *tests for* prompt-injection resistance (via `expect_refusal`) — there is no actual runtime guardrail implemented anywhere in the pipeline. Needs real design work: where in the pipeline a guardrail check would live, what it should catch beyond the existing strict-grounding prompt, and how it interacts with the conversational-routing item above (e.g. a "conversational" classification shouldn't become a loophole for injection attempts).
- **Persistent chat history + sidebar (lowest priority, after deployment).** Today conversation state only lives in Gradio's in-memory session state (`app.py`) — nothing survives a page refresh, and the backend (`routes.py`) is fully stateless. Wanted: real persistence (SQLite, no new dependency — stdlib `sqlite3` is enough) storing conversations and messages, plus a sidebar UI to browse/reopen past conversations, ChatGPT-style. Open design question flagged but not resolved: `render.yaml` suggests this may be a public deployment, and there's no auth/login system today — if so, persistence needs at least a lightweight anonymous session identifier (e.g. a cookie) so one visitor doesn't see another visitor's saved conversations in the sidebar; full user accounts are probably overkill for this project. Explicitly sequenced after deployment, not before.
- **Multi-query retrieval for comparison/multi-hop questions.** Today the pipeline does exactly one `search_hybrid()` call per query, even for questions like "compare Lasso and Ridge" that really have two sub-topics. Idea: decompose comparison-style queries into sub-queries (similar in spirit to the existing Gemini query rewriter), retrieve for each independently, then merge/dedupe context before generation — would likely improve real answer quality on multi-hop questions, not just the eval labels. Surfaced while fixing an unrelated bug in `build_golden_dataset.py`'s `multi_hop` category (the golden-dataset fix — searching each comparison side independently to find two genuinely distinct source posts — is unrelated to this and already done; this item is about doing the equivalent at retrieval time, in the live pipeline). Not started — needs its own design pass (where the decomposition step lives, how many sub-queries, how merged context gets deduped) before implementation.

**Cleanup not yet done (lower priority, flagged not fixed):**
- Root still has stray files from the old PDF-upload version of the project: `baking_recipes.pdf`, `pdf-test.pdf`, `sample_portfolio.pdf`, `tax_guide.pdf`, `test_image_report.pdf`, `test_filter.py`, `test_ingest.py`.
- `data/raw_docs/` still holds the old MCC Cricket Laws PDFs/docx from the rejected cricket-corpus idea (`plan.md` Part A.5).
- `pypdf` is still listed in both requirements files with nothing left in the codebase that imports it (PDF processing was removed).
- `tod.md` and `data_pipeline_guide.md` exist at root, purpose/currency unconfirmed.
- None of the above blocks the pipeline — flagged so a fresh chat doesn't mistake them for active code. Per the top-priority rule, don't clean these up speculatively; do it as an explicit, separate task if asked.

**Known constraints to remember:**
- I (Claude, in this sandbox) cannot reach `api.groq.com` — any Groq verification must be run by the user locally.
- Full history of prior corpus rejections, architecture rationale, and the complete step-by-step roadmap lives in `plan.md` — read it for anything not covered here.
