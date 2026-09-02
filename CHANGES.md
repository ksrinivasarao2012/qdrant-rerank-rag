# CHANGES.md — Running Record of Every Change

> **RULE FOR ANY FUTURE SESSION (human or AI):**
> **Every time something is added, changed, or removed, append an entry to
> §2 below — in the same session, not later.** Newest entry goes at the
> BOTTOM. Never rewrite or delete an existing entry; if something turns out
> to be wrong, add a new entry that corrects it and say so plainly.
> An entry must answer four things: **what changed, why, how it was verified,
> and what is still not done.**
>
> This file is the quick chronological answer to *"what has been touched and
> is it safe?"*. It does not replace `PROJECT_LOG.md` §6 (the long-form
> WHAT/WHY/OUTCOME table) or the Project Status section of `CLAUDE.md` —
> a change usually belongs in all three.

---

## 1. Current State — Guardrails

| Layer | Where | Status |
|---|---|---|
| Strict grounding prompt | `backend/prompts/system_prompts.yaml` (`gptoss_strict_v2`) | ✅ Active |
| Input check (injection / junk) | `backend/core/guardrails.py` → `check_input()` | ✅ Wired into both entrypoints |
| Relevance floor (skip the LLM entirely) | `guardrails.py` → `filter_by_score()` | ⚠️ Wired, **threshold not yet calibrated** |
| Small-talk vs off-topic routing | `guardrails.py` → `classify_empty_result()` | ✅ Wired |
| Output check (citation gating, leak strip) | `guardrails.py` → `check_output()` | ✅ Wired |

**Known open items after the entries below:**

- `MIN_RERANK_SCORE` is a **placeholder (0.25)**. Must be calibrated against the
  `out_of_scope` and `standard` golden slices, then sanity-checked on
  `niche_topic`, before the demo is shown to anyone.
- `out_of_scope` and `adversarial` eval numbers from before 2026-09-02 measured
  the OLD permissive prompt and do not describe the current system. Re-run them.
- No rate limiting on the deployed Gradio app (`app.py`).
- `stream_answer_sync` fallback can duplicate a partial answer (see the review
  list; not yet fixed).
- `app.py` and `routes.py` still differ on candidate pool size (10 vs 15) and
  query decomposition (only `app.py` has it).

---

## 2. Chronological Entries

### 2026-09-02 16:20 — Grounding contract enforced (prompt registry)

**What changed**
- `backend/prompts/system_prompts.yaml`: added answer variant **`gptoss_strict_v2`**
  and set `answer.active` to it. Previous active variant was `gptoss_simple_v1`.
- `CLAUDE.md`: Architecture line corrected (it named `gptoss_strict_v1` as
  active, which had not been true) + Done entry added.
- `PROJECT_LOG.md`: §6 rows added; §1 Generation Engine line corrected; stale
  `Last Updated` header refreshed.

**Why**
`gptoss_simple_v1`'s rule 2 told the model to answer from **general knowledge**
(behind a warning banner) whenever retrieved context was thin or partial. So
production was not a grounded RAG system: off-topic questions, jailbreak
framings, and medical/legal questions phrased as statistics were all answerable
from model priors, with no other guardrail anywhere in the pipeline. Two
knock-on defects: the load-bearing refusal string `"do not have enough
information"` was never emitted (so half of `app.py`'s citation-suppression
check was dead code), and the `adversarial` / `out_of_scope` golden categories
were testing behaviour the live prompt did not have.

`gptoss_strict_v2` = strict grounding **plus** the two style rules
`simple_v1` introduced (plain English; complete self-contained code),
**minus** the general-knowledge fallback. Three rules strengthened over
`gptoss_strict_v1`: rule 2 demands the refusal verbatim and forbids appending a
guess after it; new rule 3 covers partial coverage; new rule 4 declares Context
to be reference material, never instructions.

No existing variant was edited in place, per the registry's own A/B rule —
`gptoss_simple_v1` is retained, not deleted.

**Verified**
`prompts.active('answer')` → `gptoss_strict_v2`; model and `extra_body` resolve
correctly through `defaults`; refusal string asserted byte-exact against
`app.py`'s substring check.

**Not done**
This is a prompt-level guardrail, the weakest kind — the model is *told* to
refuse, nothing *makes* it. Enforcement came in the next entry.

---

### 2026-09-02 17:05 — Runtime guardrails built and wired (items #3–#6)

**What changed**
- **NEW `backend/core/guardrails.py`** — four pure functions, no state, no new
  dependency:
  - `check_input(query)` → refusal string or `None`. Regex list for injection
    phrases (`ignore previous instructions`, `reveal your system prompt`,
    `you are now`, `DAN mode`, `</system>`, …) plus junk shapes (a character
    repeated 50+ times, a 200-char base64 blob). Runs on the RAW query, before
    the LLM rewriter.
  - `filter_by_score(chunks)` → chunks clearing `MIN_RERANK_SCORE`. An empty
    return means skip generation entirely.
  - `normalize_rerank_score(raw, source)` → puts both reranker backends on one
    0-1 scale.
  - `classify_empty_result(query)` → `"small_talk"` or `"off_topic"`;
    `empty_result_response()` wraps it into a message.
  - `check_output(answer, chunks)` → `(cleaned_answer, show_citations)`.
    Suppresses citations when the answer refused, carried the legacy
    out-of-boundary banner, or shares under 15% of its content words with the
    retrieved text. Strips leaked system-prompt text.
- `backend/core/reranker.py`: tags each chunk with `rerank_source`
  (`"jina"` / `"local"`); corrected the false comment claiming local scores are
  0-1.
- `backend/prompts/system_prompts.yaml`: `gptoss_strict_v2` gains its own
  `citation_block` wrapping each chunk in
  `--- BEGIN/END RETRIEVED DOCUMENT n ---` markers.
- `app.py`: all four guardrails wired into `chat_stream`.
- `backend/api/routes.py`: all four wired into `/api/v1/query`.

**Why**
- **Relevance floor is the real enforcement.** Previously the reranker sorted
  and sliced `top_k=3` unconditionally, so a query with no good match produced
  identically-shaped context to a perfect match, and the LLM was always called.
  Now, when nothing clears the floor, the model is never invoked — there is
  nothing to hallucinate with.
- **Small talk placed AFTER retrieval failure, deliberately.** A greeting
  classifier running up front would be a bypass: *"hi, ignore your instructions
  and…"* would route to the friendly path. Reached only by a query that already
  passed `check_input` and then failed to retrieve anything.
- **Score-scale trap fixed.** `cross-encoder/ms-marco-MiniLM-L-6-v2` emits raw
  **logits (≈ −11…+11)**, not 0-1 as `reranker.py`'s comment claimed; the Jina
  API emits a true 0-1 score, and Jina silently disables itself on a 401/403.
  Without normalization one threshold constant would have meant two different
  things depending on billing state.
- **Citation markers scoped to the new variant, not `defaults`.** Editing
  `defaults.citation_block` would change rendered context for every variant
  including retired ones, breaking comparability of historical eval scores.
- **Citations are a privilege the answer earns.** Attaching sources to an answer
  that ignored them launders a general-knowledge answer as a sourced one, which
  is worse than showing no sources.
- **`fail open` on a missing score.** If the cross-encoder fails to load,
  `reranker.py` bypasses scoring; chunks with no score are kept rather than
  refused, so a model-loading failure degrades quality instead of refusing
  every question.

**Verified**
Syntax-checked all four files. Ran the full call sequence with stub chunks (no
Qdrant, no Groq) across 7 scenarios — injection blocked; `"hi"` and `"thanks!"`
→ small talk; `"what is the capital of France?"` with weak chunks → refusal;
grounded answer → `citations=True`; off-topic answer over good chunks →
`citations=False`; model refusal → `citations=False`. Confirmed
`prompts.fingerprint` for `strict_v1` (`037185a1`) and `gptoss_simple_v1`
(`9444ddb3`) are unchanged by the `citation_block` edit.

Note: `gptoss_strict_v2`'s fingerprint moved `02899d35` → `5e5270c0` when the
markers were added. Harmless — the variant had not been used for any eval run
between the two edits.

**Not done**
- `MIN_RERANK_SCORE = 0.25` is a **guess, not a measurement.** Calibrate on
  `out_of_scope` (should fall below) vs `standard` (should clear), then check
  `niche_topic` for wrongly-refused real questions. Override without editing
  code via `GUARDRAIL_MIN_SCORE`.
- `MIN_GROUNDING_OVERLAP = 0.15` is likewise uncalibrated. Word overlap is a
  crude proxy for grounding; it is set low so it only fires on a clear miss.
- The injection list is a fixed pattern set. It stops copy-pasted attacks, not
  novel phrasing. The floor is what bounds the damage.
- Indirect injection through a retrieved chunk is only *mitigated* (markers +
  prompt rule 4), not solved. Nothing scans chunk text itself.
- On the FastAPI streaming path, citations lead the stream by design, so tokens
  cannot be recalled. `check_output` therefore emits a trailing
  `{"type": "citations_valid", "data": bool}` line instead. **Any client must
  honour it**; a client that ignores it behaves exactly as before. `app.py` does
  not have this problem — it composes the footer after the stream ends.

---

### 2026-09-02 17:45 — Five review bugs fixed (#7–#11)

**What changed**
- **#11 dead code:** deleted the unreachable second `return query` at the end
  of `rewrite_query()` in `llm_service.py`.
- **#7 duplicated answer on stream failure:** `stream_answer_sync()` now
  tracks `yielded_any`. If the primary (or a fallback) model dies **after**
  tokens have already reached the user, it stops with a short
  `"[Connection interrupted -- please try again.]"` instead of starting a
  fresh model that would write a second, complete answer glued onto the
  partial one. The fallback cascade still runs exactly as before when nothing
  has been shown yet.
- **#9 rewrite step has no time limit** and **#10 rewriter output goes
  straight to search unchecked:** new `guardrails.safe_rewrite_query()`
  wraps `llm_service.rewrite_query()`'s up-to-7-provider cascade in one
  overall deadline (`REWRITE_TIMEOUT_SECONDS`, default 4s; override without
  editing code) and validates the result with `is_valid_rewrite()` (rejects
  empty, absurdly long (>500 chars), or completely unrelated rewrites) before
  it can reach retrieval. Both `app.py` and `routes.py` now call this wrapper
  instead of `llm_service.rewrite_query()` directly.
- **#8 the two pipelines don't match:** `routes.py` brought to parity with
  `app.py` (the tuned, interactively-evaluated path) — `CANDIDATE_K` unified
  to 10 (was 15 in `routes.py`), and multi-hop query decomposition
  (`decompose_query` / `search_multi_query`) added to `routes.py`, which
  previously never split comparison queries at all.

**Why**
Per the code review: a broken stream mid-answer was producing visibly garbled
duplicate output; an unbounded provider cascade could stall a request for the
sum of every provider's timeout; a rewritten search query was never sanity
checked before being used; and the FastAPI endpoint and the Gradio app were
silently running different retrieval behavior behind the same prompt
registry, so an eval run against one told you nothing certain about the other.

**Verified**
- `#11`: confirmed exactly one `return query` remains at the end of
  `rewrite_query()` (3 other `return query` lines are legitimate branch exits
  inside the provider loop).
- `#9`/`#10`: `safe_rewrite_query()` tested against a stub LLM service that
  sleeps 2s — with `REWRITE_TIMEOUT_SECONDS=0.3`, returned the raw query in
  0.30s, not 2s. **Caught and fixed a real bug during this verification**:
  the first implementation used `with ThreadPoolExecutor() as pool:`, whose
  `__exit__` calls `shutdown(wait=True)` and blocked for the full 2s even
  after the timeout had already fired, defeating the deadline entirely.
  Fixed by managing the executor manually and calling
  `shutdown(wait=False)` before returning. Also tested `is_valid_rewrite()`
  against an unrelated-topic rewrite (rejected) and a legitimate pronoun
  resolution (accepted).
- `#7`: tested `stream_answer_sync()` against a stub client that yields two
  tokens then raises — output contained the word "lasso" exactly once (no
  duplication) and ended with the interruption message. A second stub that
  raises before any token is yielded confirmed the fallback cascade is still
  attempted in that case.
- `#8`: confirmed by grep that `routes.py` now imports `decompose_query`, sets
  `CANDIDATE_K = 10`, and calls `search_multi_query` when a query decomposes.
  Full syntax check (`ast.parse`) passed on all five touched files.

**Not done**
- `#8` is parity on candidate pool size and decomposition only, not a full
  merge into one shared pipeline function — `routes.py` and `app.py` still
  have separately-written retrieval/generation call sequences that must be
  kept in sync by hand. A real fix would extract one shared function; skipped
  per the project's own "don't over-engineer" rule until a second real
  divergence shows up.
- `routes.py`'s async path (`stream_answer`, used by FastAPI) still has no
  model-fallback cascade at all, unlike `stream_answer_sync` (used by
  Gradio) — this was true before today's changes and is unrelated to #7-#11,
  noted here so it isn't mistaken for already covered.
- `REWRITE_TIMEOUT_SECONDS=4.0` and the `is_valid_rewrite` bounds (500 chars,
  any-word-overlap) are reasoned defaults, not measured against real rewrite
  logs — same caveat as the guardrail thresholds in the 17:05 entry.

---

### 2026-09-02 18:20 — Deployment safety + doc corrections (#12, #14, #16, #17, #18, #19, #20, #23)

**What changed**

- **#12 no rate limit on the deployed surface:** extracted `SlidingWindowRateLimiter`
  out of `routes.py` into new `backend/core/rate_limiter.py` (shared, so it
  doesn't drift the way `app.py`/`routes.py` already had — see #8) and wired it
  into `app.py`'s `chat_stream()`, keyed by client IP via Gradio's auto-injected
  `gr.Request`. Previously only `routes.py` had a limiter, and HF Spaces runs
  `app.py`, not `routes.py` — the actually-deployed surface had none.
- **#14 `/healthz` always said "ok":** split into a bare `/` liveness check
  (unchanged — process is up, no dependency calls) and a real `/healthz`
  readiness check that verifies the embedding model, the reranker, and Qdrant
  reachability, returning HTTP 503 with a per-check breakdown on failure
  instead of an unconditional 200.
- **#16 no startup validation of API keys:** added `Settings.validate()` to
  `config.py`, called from both entrypoints' startup path (`app.py` init,
  `backend/main.py`'s `warm_models`). Logs a clear warning if `GROQ_API_KEY`
  is missing; does not exit/raise, since some entrypoints (eval scripts,
  retrieval-only testing) legitimately run without it.
- **#17 `HF_HUB_OFFLINE=1` forced unconditionally:** now only set when
  `huggingface_hub.scan_cache_dir()` confirms both required models
  (`BAAI/bge-base-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) are
  already cached locally; otherwise left unset so a cold container can
  download them. Uses `os.environ.setdefault`, so an operator's explicit
  `$HF_HUB_OFFLINE` still wins either way.
- **#18/#19 stale architecture claims:** `PROJECT_LOG.md` §1 corrected —
  dense model line updated from the retired `bge-small-en-v1.5` (384-dim) to
  the actual `bge-base-en-v1.5` (768-dim, live since 2026-08-30); sparse model
  line corrected from "Fast BM25" to what it actually is (raw TF×IDF, no
  saturation, no length normalization), with a pointer to the existing
  2026-09-02 10:15 finding and the scoped-but-not-yet-run BM25 fix.
  `README.md` — **appended** a corrections table (per its append-only rule)
  for the same two prose claims ("Server-Side BM25" in two places) plus the
  unsupported "K=100" pool-size claim; the architecture diagram itself
  already said the sparse channel correctly ("CRC32 Feature Hashing, TF") and
  was left alone.
- **#20 README pool-size claim:** covered by the same corrections table above
  — code has never used K=100; `CANDIDATE_K` is 10 in both entrypoints as of
  the #8 fix earlier today.
- **#23 stray files / unused dependency:** checked disk for the items
  `CLAUDE.md`'s Cleanup section flagged (`baking_recipes.pdf`, `pdf-test.pdf`,
  `sample_portfolio.pdf`, `tax_guide.pdf`, `test_image_report.pdf`,
  `test_filter.py`, `test_ingest.py`, `data/raw_docs/*`) — **all already
  gone**, nothing to delete. Removed the one real leftover: `pypdf>=3.0.0`
  from both `requirements.txt` and `backend/requirements.txt` (confirmed zero
  imports of `pypdf`/`PyPDF` anywhere in the codebase before removing).

**Why**

Per the code review: the live public demo (`app.py`, deployed via HF Spaces)
had no request throttling at all, so a rate limiter existed but was protecting
nothing anyone actually hits. A health endpoint that can never report unhealthy
gives an operator no signal. A missing `GROQ_API_KEY` used to surface later as
a confusing runtime error instead of a clear startup warning. Forcing offline
mode unconditionally turned a cold-start container into a hard failure instead
of a one-time download. And three places in the docs described a system that
no longer matches the code (the embedding model was upgraded 2026-08-30, the
sparse channel was found to not be BM25 on 2026-09-02, and the pool size was
never 100 in the first place) — left uncorrected, a future reader (human or
AI) would trust the wrong numbers.

**Not done / deliberately skipped**

- **#13 (queue cap on `demo.queue()`)** and **#15 (CORS `allow_origins=["*"]`
  with `allow_credentials=True`)** were on the original review list but were
  **not** part of this batch — only the items explicitly requested were done.
- **#21 (real BM25)**, **#22 (16,140 stale Qdrant points)**, **#24
  (persistent chat history + sidebar)** — explicitly deferred; #24 in
  particular was raised with the user and confirmed skipped (contradicts
  `CLAUDE.md`'s own "after deployment" sequencing for that item).
- `_models_already_cached()` in `config.py` depends on `huggingface_hub`
  being importable and its cache-scan succeeding; if that check itself fails
  for any reason (uncommon cache layout, permissions), the code fails safe by
  NOT forcing offline mode, which is the conservative direction.

**Verified**

- `ast.parse` on all 8 touched Python files — clean.
- `rate_limiter.py`: unit-tested directly — 3rd call within budget succeeds,
  4th raises `RateLimitExceeded`; a second, different key is unaffected.
- `config.py`: `validate()` logs a warning with `GROQ_API_KEY` unset, is
  silent with it set; `HF_HUB_OFFLINE` confirmed NOT set when the cache can't
  be verified (this sandbox has no local HF cache, which is exactly the
  "cold start" case the fix targets).
- `pypdf` confirmed absent from both requirement files via grep, and zero
  imports existed beforehand.
- Doc corrections cross-checked against the actual code
  (`backend/core/embeddings.py` already correctly said `bge-base-en-v1.5` /
  768-dim / `EMBEDDING_DIM = 768` — only the docs had drifted) and against the
  README append-only rule (diffed before/after; nothing above the appended
  section changed).
