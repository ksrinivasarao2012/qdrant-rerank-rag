# Retrieval Eval — Investigation Log

## Status: unresolved, actively debugging low recall

## What we ran and found

### 1. First `eval_retriever.py` run (default config, against Qdrant Cloud)

```
python evaluation/eval_retriever.py
```

Config: `method=hybrid rerank=True rewrite=False pool=50 ks=[3, 5, 10]`

```
OVERALL: n=306  MRR=0.124
  recall@3=0.134  precision@3=0.045
  recall@5=0.144  precision@5=0.029
  recall@10=0.183  precision@10=0.018

BY CATEGORY:
  citation_accuracy    n= 20  MRR=0.306  recall@10=0.40
  code_traceback       n= 30  MRR=0.196  recall@10=0.27
  multi_hop            n= 20  MRR=0.066  recall@10=0.10
  multi_turn           n= 19  MRR=0.011  recall@10=0.05
  negation             n= 20  MRR=0.006  recall@10=0.05
  niche_topic          n= 17  MRR=0.077  recall@10=0.24
  paraphrase_group     n= 80  MRR=0.015  recall@10=0.05
  standard             n=100  MRR=0.217  recall@10=0.28
```

**This is bad.** `standard` — the easiest, most straightforward category (plain
questions sampled directly from the corpus, no adversarial tricks) — only hits
28% recall@10. A working hybrid+rerank pipeline should be well above that on
straightforward lookups. Something is systemically wrong, not just "retrieval
quality needs tuning."

### 2. Hypothesis: stale/leftover points polluting the live Qdrant collection

Context: an earlier, abandoned run of the old `seed_corpus.py` (before the
quality-floor corpus filter was added to `parse_dump.py`) left the live Qdrant
Cloud collection with 234,596 points, but the current, correct
`embedded_points.jsonl` only has 218,456. The 16,140-point gap was flagged
earlier and never cleaned up. Reasonable first suspect: old chunks from a
different corpus snapshot crowding out the correct ones under RRF fusion.

Ruled out two other candidates before testing this:
- **ID type mismatch** (gold IDs are strings in `golden_dataset.json`, but
  `answer_id` is stored as `int` in Qdrant payload) — checked
  `eval_retriever.py`'s `ranked_answer_ids()`, it already does
  `str(chunk["metadata"].get("answer_id"))` before comparing. Not a bug.
- **Golden dataset built from a different corpus snapshot than what got
  embedded** — checked directly: all 100 `standard` category gold
  `answer_id`s exist in `embedded_points.jsonl`. Not the cause.

### 3. `diagnose_stale_points.py` — checked 5 `standard` queries against live Qdrant

```
$env:QDRANT_URL=...; python evaluation/diagnose_stale_points.py
```

For each query, ran `search_hybrid()` live and checked every top-10 hit's
`answer_id` against the known-good `embedded_points.jsonl`:

| query | gold rank in top10 | stale hits in top10 |
|---|---|---|
| std_001 (ARIMA transform) | not found | 0/10 |
| std_002 (ANOVA) | not found | 0/10 |
| std_003 (design of experiments) | rank 1 (hit) | 0/10 |
| std_004 (multicollinearity) | not found | 2/10 |
| std_005 (2SLS vs OLS variance) | rank 2 (hit) | 0/10 |

**Only 2 stale hits out of 50 total results checked.** Contamination is real
but minor — nowhere near enough to explain an 82% miss rate. This mostly
ruled out the stale-points hypothesis, but wasn't fully conclusive on its own
(small sample, single query variant).

### 4. Built a clean local index and re-ran the full eval, to confirm

Since `VectorDBManager` already supports an embedded/on-disk Qdrant mode
(falls back to `QdrantClient(path=...)` when `QDRANT_URL`/`QDRANT_API_KEY`
are unset), built a fresh local collection directly from the known-good
`embedded_points.jsonl` (218,456 points, zero stale contamination possible)
and re-ran the exact same eval against it instead of the cloud collection.

```
evaluation/load_local_qdrant.py   # seeds local on-disk Qdrant from embedded_points.jsonl
evaluation/eval_retriever.py      # same eval, run with QDRANT_URL/QDRANT_API_KEY blanked
```

Result: **nearly identical numbers to the cloud run.**

```
OVERALL: n=306  MRR=0.124  (cloud: 0.124)
  recall@10=0.183  (cloud: 0.183)

BY CATEGORY (local vs cloud, recall@10):
  citation_accuracy    0.40 vs 0.40
  code_traceback       0.27 vs 0.27
  multi_hop            0.10 vs 0.10
  multi_turn           0.05 vs 0.05
  negation             0.05 vs 0.05
  niche_topic          0.24 vs 0.24
  paraphrase_group     0.05 vs 0.05
  standard             0.28 vs 0.28
```

**Conclusion: stale Qdrant Cloud points are NOT the cause.** The bug is real
and lives in the pipeline itself (embedding, sparse vector generation, RRF
fusion, or reranking) — not in the collection being dirty. This is confirmed,
not just suspected, since local and cloud produce essentially the same
numbers on the same clean data.

## Current step (in progress)

Built `evaluation/diagnose_pipeline_stages.py` to trace a handful of failing
`standard` queries through each individual stage:

1. Dense-only search (top 50) — is the gold answer even embedded close to
   the query?
2. Sparse-only search (top 50) — is keyword overlap working?
3. Hybrid RRF-fused search (top 50) — does fusion keep or lose it?
4. After cross-encoder reranking (top 10) — does reranking keep or lose it?

This should show exactly which stage is dropping the correct answer, rather
than guessing at the whole pipeline.

Run with:
```
$env:QDRANT_URL=""; $env:QDRANT_API_KEY=""; python evaluation/diagnose_pipeline_stages.py
```

## 5. `diagnose_pipeline_stages.py` — traced 6 `standard` queries through every stage

```
$env:QDRANT_URL=""; $env:QDRANT_API_KEY=""; python evaluation/diagnose_pipeline_stages.py
```

| query | dense rank | sparse rank | hybrid rank | rerank rank | top10? |
|---|---|---|---|---|---|
| std_001 (ARIMA transform) | 32 | 9 | 13 | 7 | YES |
| std_002 (ANOVA sig. diff) | None | None | None | None | NO |
| std_003 (design of experiments) | 1 | None | 1 | 1 | YES |
| std_004 (multicollinearity) | None | None | None | None | NO |
| std_005 (2SLS vs OLS) | 1 | 2 | 2 | 2 | YES |
| std_006 (response variable) | 11 | None | 21 | 11 | NO |

Two clear patterns:
- When dense-only already finds the gold answer near the top (std_003,
  std_005), hybrid + rerank works fine end to end.
- When dense-only can't find it AT ALL in the top 50 (std_002, std_004), no
  later stage can recover it — sparse and hybrid also come up empty. The
  loss happens at the embedding step itself, not fusion or reranking.

## 6. Root cause found: chunk text has no question context

Checked `answer_id 103580` (std_002's gold answer) and `374068` (std_004's)
directly in `embedded_points.jsonl`. Both exist, both are single, complete
chunks — not a chunking or missing-data bug. But the actual embedded text:

- `103580`: *"That is correct. The repeated measures ANOVA is an omnibus
  test, so if you reject its null hypothesis..."*
- `374068`: *"Such multicollinearity is matter of fact and does not matter
  substantively. With interactions, you can reduce..."*

Both are direct, deictic answers ("That is correct...", "Such X is...")
that only make sense next to their question. `question_title` is stored in
Qdrant payload metadata for citations, but `backend/core/ingestion.py`'s
`DocumentProcessor.process_answer()` never included it in the text that
actually gets embedded. So the embedding for these chunks carries almost no
signal connecting them to their own topic — dense similarity against the
real query has nothing to latch onto.

This is a known structural issue with StackExchange-style Q&A corpora
specifically (unlike e.g. Wikipedia paragraphs, which are self-contained),
and explains why `standard` — the easiest category — was only hitting 28%
recall@10.

### Fix applied

`backend/core/ingestion.py`, `process_answer()`: now prepends
`question_title` to each chunk's embedded text (`f"{question_title}\n\n{chunk}"`)
before it goes to the embedder, instead of leaving it in metadata only. Since
the same `text` field also feeds citation snippets and generation context
(`routes.py`, `llm_service.py`), this also gives the generator slightly
better topic framing for free — not a retrieval-only change.

## 7. Follow-on fixes made alongside the root-cause fix

While implementing the title-prepend fix, found and fixed two related gaps
in the same file:

- **No chunk overlap.** Checked directly: 29% of answers (27,116 / 93,455)
  produce more than one chunk, so a boundary landing mid-argument was a real,
  non-rare cost, not theoretical. Added a 200-char overlap (`chunk_overlap`
  was already sitting unused as a constructor default) via a new
  `_add_overlap()` step in `ingestion.py`, applied after paragraph-boundary
  splitting.
- **Citation snippets would have shown the title-prefixed, overlap-padded
  text to users.** Split `process_answer()`'s output into two fields: `text`
  (title + overlap + chunk, used for embedding/generation) and
  `display_text` (plain chunk, used for citations). Wired through
  `vector_store.py`, `embed_corpus.py`, `routes.py`, `app.py`, and
  `eval_pipeline.py` (the last one was found out of sync with `routes.py`
  during a later file-by-file review and fixed to match).

## 8. Accidental `git reset` and recovery (unrelated to the retrieval bug)

Separately from all of the above: an accidental `git reset` rolled the
working tree and a subsequent commit back to a much older snapshot,
re-introducing already-deleted PDF/vision code (`ingestion.py`,
`hybrid_search.py`, `app.py`, `routes.py`, and others) and losing the
title-prepend fix from the file on disk (though not from git history --
recovered from commit `2940094`). Fully diagnosed and restored via a
file-by-file review; see git log / conversation history for the full
account. Not a retrieval-quality issue, noted here only because it
interrupted this investigation and is relevant if these numbers ever look
inexplicably different from what's described above.

## 9. Golden dataset also expanded during the same review

Every category now has at least 20 cases (was short on `multi_turn`,
`niche_topic`, `out_of_scope`, `adversarial` -- `out_of_scope`/`adversarial`
only ever had 10/8 hand-written queries; `multi_turn`/`niche_topic` lost a
few to un-matchable keyword sets). Added buffer cases to
`build_golden_dataset.py` covering subfields not already represented
(meta-analysis, spatial stats, Bayes factors, SEM, power analysis, survival
analysis, A/B testing, boosting) rather than padding existing topics.
Dataset is now 294 cases (was 264). Re-running `eval_retriever.py` after the
re-embed will produce numbers against this larger set, not the 264-case one
referenced above -- expect small shifts in aggregate numbers from the sample
size change alone, separate from whatever the fix itself changes.

## 10. Full audit of the 52 flagged golden-dataset cases (all checked, not just 5)

Re-ran the zero-tag-overlap flag against the 294-case dataset (grew from 47
to 52 flagged cases as the dataset grew). Checked every one by comparing the
query against the actual gold post's title/content, not just tags:

- **33 were false alarms** — topically correct, flagged only because tags
  are synonyms/software-specific labels (e.g. `lasso`/`ridge-regression` vs
  declared `regularization`/`regression`). No action needed.
- **17 were real content mismatches** — keyword collisions where the gold
  post shares a word with the query but is about a different topic (e.g.
  `hop_03`'s "power" matching an electrical-engineering "power factor" post
  instead of statistical power; `mturn_15`'s "decay" matching a *weight*
  decay post instead of *learning rate* decay). Fixed by finding real
  matching posts in the corpus and tightening each case's `include`/
  `exclude` keywords in `build_golden_dataset.py` to hit them via tier-1
  title matching, then verified the actual output (not just assumed the fix
  worked) -- one fix (`hop_04`) needed a second pass after the first attempt
  landed on an unrelated "harmonic mean of zero" math trivia post instead of
  the intended F-measure post, and `neg_18`'s `exclude: ["gaussian"]` was
  dropped after it turned out to filter out the correct Epanechnikov-kernel
  post too (almost any real KDE post mentions Gaussian kernels in passing).
- **2 were a structural issue, not a topic mismatch** — `hop_04` and
  `hop_05` each had both "gold posts" turn out to be two different answers
  to the *same* question, not two genuinely distinct sources, defeating
  multi_hop's whole point. Fixed by choosing `include_a`/`include_b`
  keyword pairs specific enough to land on two different questions.

All 294 cases regenerated cleanly after these fixes -- same 4 expected
skips (genuinely obscure terms in `niche_topic`/`multi_turn`, unrelated to
this audit), no new skips introduced by any of the 19 keyword changes.

## 11. Full manual re-verification against posts.jsonl (exhaustive, all 294 cases)

Ran an exhaustive structural/existence check (not sampled) of every case
against the real corpus: every `gold_answer_id`/`negative_answer_id` exists
in `posts.jsonl`, `graded_relevance` keys match `gold_answer_ids` exactly,
every `multi_hop` case has exactly 2 distinct gold answers from 2 distinct
questions, `standard`/`code_traceback` queries are byte-exact matches to
their gold post's own title, `citation_accuracy`'s `expected_url` matches
the real post URL, no duplicate IDs anywhere. **0 errors found.**

Also manually read all 20 `paraphrase_group` variant sets -- all coherent
restatements of their base query, no drift.

Re-ran the tag-overlap check post-fix: dropped from 52 to 43 (9 of the 18
fixes landed on posts whose real tags happened to overlap the case's
declared tags already). The remaining 43 included the 9 fixes whose `tags`
field still described the *old* topic rather than the new gold post -- a
cosmetic-only gap since `tags` isn't used by `eval_retriever.py` for
scoring. Updated those 9 cases' `tags` fields to match their new gold
posts' real StackExchange tags. Final count: 34 flagged, all confirmed
false alarms (synonym-only tag mismatches, already content-verified
correct) -- exactly the false-alarm set from section 10, nothing new.

Golden dataset content is now fully verified and internally consistent.

## Not yet done

- **Full re-embed required.** The title-prepend, overlap, and display_text
  changes all alter what actually gets embedded/stored, so none of them take
  effect until `embed_corpus.py` is re-run (local, ~93k answers) and
  `upload_embeddings.py` re-uploads the new `embedded_points.jsonl` to
  Qdrant Cloud, replacing the current collection. Not yet run.
- Re-run `eval_retriever.py` (and `diagnose_pipeline_stages.py` on the same
  6 queries, for a clean before/after comparison) once re-embedding is done.
- Re-run the full ablation suite for real, trustworthy README/resume numbers
  once the fix is confirmed working.
- Golden-dataset content audit is now complete (see section 10) -- no
  further action needed there.
- Separately, still pending: cleanup of the 16,140 stale points in the live
  Qdrant Cloud collection (confirmed minor impact via
  `diagnose_stale_points.py` — only 2/50 hits checked were stale — but still
  incorrect data that should eventually be removed; not blocking, not
  urgent, and will be moot after the re-embed replaces the collection
  anyway).
