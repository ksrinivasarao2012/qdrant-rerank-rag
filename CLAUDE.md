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
- Groq model migration off `llama-3.1-8b-instant` (deprecated 2026-08-16) → `openai/gpt-oss-20b` (answers) / `qwen/qwen3.6-27b` (vision). Verified live via `verify_models.py`.
- Corpus decision: Cross Validated (stats.stackexchange.com) selected after rejecting 6 other candidates. Full reasoning in `plan.md` Part A/B.
- **Corpus filter** in `backend/scripts/parse_dump.py`: switched from top-50k-by-score to a quality floor (`score >= 1 AND accepted_answer_id is not None`, no upper cap). Reduces popularity bias against niche topics. Result: `data/processed/posts.jsonl` = 93,455 answers.
- **Ingestion fix** (`backend/core/ingestion.py`): `process_answer()` now prepends `question_title` to each chunk's embedded text and applies a 200-char overlap between chunks. Was the root cause of poor recall — StackExchange answers routinely start with deictic phrases ("That is correct...") that carry almost no topical signal when embedded alone. Chunks now also carry a separate `display_text` field (plain chunk, no title/overlap) used for citation snippets so users don't see title-prefixed or overlap-padded text in the UI. Wired through `vector_store.py`, `embed_corpus.py`, `routes.py`, `app.py`, `eval_pipeline.py`.
- **Corpus fully embedded locally** with the ingestion fix: `data/processed/embedded_points.jsonl` = 218,456 chunks. `embed_corpus.py` (local-only, resumable) + `upload_embeddings.py` (network-only, retry-with-backoff, idempotent via `uuid5(answer_id+chunk_index)` point IDs) replaced the old single-pass `seed_corpus.py` (now a deprecation stub). Upload to Qdrant Cloud was completed in a prior run (218,456 points); a stale-point discrepancy (16,140 leftovers from the very old abandoned `seed_corpus.py` run against the pre-quality-floor corpus) is still there but confirmed low-impact and cleanup is unblocked-optional.
- **Live-bug fixes made during file-by-file review**:
  - `backend/core/llm_service.py`: `stream_answer()` (async, the method `routes.py` actually calls in production) used to misroute every request to a dead Hugging Face Inference API fallback whenever `self.model_name` contained "/", which trigger'd on every request since Groq's own IDs like `openai/gpt-oss-20b` contain "/". Fixed to always use Groq, matching the sync path.
  - `backend/core/vector_store.py`: topic filter in `search`/`search_sparse`/`search_hybrid` and the payload index were filtering on `source_file` (leftover PDF field), but no chunk in the current corpus ever has that key. The UI dropdown in `app.py` and the `source_file` field in `routes.py`'s `QueryRequest` therefore silently matched zero results for any specific topic. Fixed the Qdrant filter key to `tags` (the field that actually exists). Kept the parameter name `source_file=` for backward compat with `app.py`/`routes.py` callers rather than a coordinated multi-file rename.
  - `backend/core/vector_store.py`: added `force_local=True` constructor flag to bypass `.env` entirely, since blanking `QDRANT_URL`/`QDRANT_API_KEY` in a Windows PowerShell shell doesn't work (`$env:VAR=""` deletes the variable, then `load_dotenv()` refills it silently, defeating local-index testing). `evaluation/load_local_qdrant.py`/`eval_retriever.py --local`/`diagnose_pipeline_stages.py` all use this now instead of the shell trick.
  - `backend/core/vector_store.py`: cached `_collection_ready` so `collection_exists`/`create_payload_index` don't re-run over the network on every batch (previously ~344 redundant calls per seed run).
  - `backend/core/vector_store.py`: retry-with-backoff on `upsert()` (3 attempts, 2s/4s) — root fix for the original silent data-loss bug where transient Qdrant Cloud errors dropped whole batches unnoticed.
  - `backend/scripts/embed_corpus.py`: never flushes mid-answer; keeps all chunks of one answer atomic across writes, so a crash between flushes can't permanently strand a multi-chunk answer as "done but incomplete" on resume.
- **Cold-start warm-up hooks** in `app.py`, `backend/main.py`, `render.yaml`, `Dockerfile` — force the embedding model + reranker + Qdrant reachability check to happen at container start rather than on the first real request. Only `app.py` and `render.yaml` matter for actual deployment (HF Spaces free tier only allows Gradio SDK spaces, not Docker; Render deploys via `render.yaml`, no Docker).
- **Golden dataset** at `evaluation/golden_dataset.json` = 294 cases across 10 categories (each category ≥20). Built by `backend/scripts/build_golden_dataset.py`. Categories: standard(100), code_traceback(30), citation_accuracy(20), multi_hop(20), multi_turn(22), negation(20), niche_topic(22), out_of_scope(20), adversarial(20), paraphrase_group(20). Exhaustively verified: every gold/negative answer ID exists in `posts.jsonl`, every `multi_hop` case has 2 distinct gold answers from 2 distinct questions, `standard`/`code_traceback` queries byte-exactly match their gold post's title, `citation_accuracy` URLs match, 18 real content mismatches found and hand-fixed via keyword-list tightening in `build_golden_dataset.py`. Full log in `evaluation/results.md`.
- **Evaluation scripts written** (`evaluation/eval_retriever.py`, `eval_generator.py`, `eval_pipeline.py`, `eval_application.py`, `judge_model.py`, `load_local_qdrant.py`, `diagnose_stale_points.py`, `diagnose_pipeline_stages.py`). Judge is a local GGUF Qwen2.5-7B-Instruct via `llama-cpp-python` (deliberately eval-only, in `evaluation/requirements.txt`, never deployed). `eval_retriever.py` has ablation flags (`--method`, `--no-rerank`, `--rewrite`, `--pool`, `--reranker-model`, `--ks`, `--local`) so ablation runs are saved with distinguishable tagged filenames.
- **Git-reset incident (recovered)**: mid-session, an accidental `git reset` rolled the working tree back to a much older snapshot, silently re-introducing already-deleted PDF/vision code. Fully diagnosed and restored via file-by-file review; also caught during that review that `evaluation/run_evals.py` (deprecation stub) and `backend/prompts/system_prompts.yaml` (256-line prompt registry, critical) had both been reset-blanked and needed restoring from commit `2940094`.
- `deepeval` in `requirements.txt` + `backend/requirements.txt`; `llama-cpp-python` in `evaluation/requirements.txt` only.

**Not started:**
- **Actually running the full ablation suite on the re-embedded corpus**, and updating README/resume with the numbers. The default-config run needs to be done first (via `--local` to test locally, then against Qdrant Cloud once satisfied) and compared against the pre-fix baseline in `evaluation/results.md`.
- **Running `eval_generator.py`, `eval_pipeline.py`, `eval_application.py`** end to end. Written and syntax-verified but never actually executed against real data. Requires the local Qwen2.5-7B-Instruct GGUF judge model to be downloaded manually (see `evaluation/judge_model.py` docstring).
- **Cleanup of the 16,140 stale points in the live Qdrant Cloud collection**. Low-impact per `diagnose_stale_points.py` (only ~4% of top-10 hits sampled), but still incorrect data.
- `plan.md` Part F execution steps: Regression/Online eval runs, CI wiring.
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
