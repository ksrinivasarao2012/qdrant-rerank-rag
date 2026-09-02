# RAG Evaluation Report (generated)

_Generated 2026-09-01 13:42 by `evaluation/generate_report.py`. Every number below is computed from the result files named in each section; none are hand-entered._

## Provenance

| artifact | source file | modified |
|---|---|---|
| Graded evaluation | `contextual_recall_eval_20260901_181021.json` | 2026-09-01 12:40 |
| Retrieval/reranker diagnostic | `retrieval_diagnostic.json` | 2026-09-01 12:56 |
| Reranker bake-off | `reranker_bakeoff.json` | 2026-09-01 13:33 |

## Measurement trust gate

The judge (DeepEval `ContextualRecallMetric`) can fail on quota or malformed output. When it does, the harness records a heuristic fallback score and flags the row. A run is only trustworthy if this count is low.

- Cases scored by the **real judge**: **79/79** (100.0%)
- Cases that fell back to the heuristic: **0/79** (0.0%)

## Headline metrics

| metric | value |
|---|---|
| Cases evaluated | 79 |
| Fact coverage (mean ContextualRecall) | **30.2%** |
| Strict recall@1 | 16.5% |
| Strict recall@5 | 38.0% |
| MRR | 0.245 |

### By category

| category | n | strict R@5 | MRR | fact coverage |
|---|---|---|---|---|
| multi_hop | 23 | 47.8% | 0.366 | 37.4% |
| multi_turn | 18 | 38.9% | 0.224 | 36.3% |
| negation | 20 | 30.0% | 0.210 | 21.6% |
| niche_topic | 18 | 33.3% | 0.152 | 24.6% |

## Where the loss actually is

Splitting fact coverage by whether retrieval surfaced a gold document at all separates *retrieval* failure from *coverage* failure:

| condition | n | mean fact coverage |
|---|---|---|
| gold document **was** in top-5 | 30 | **0.635** |
| gold document **was not** in top-5 | 49 | 0.098 |

When the right document is retrieved, coverage is 0.635. The headline number is therefore bounded by retrieval hit-rate, not by the system's ability to use what it retrieves.

## Retrieval vs. reranker (no LLM calls)

Source: `retrieval_diagnostic.json` — 79 cases.

| stage | recall |
|---|---|
| gold in candidate pool (retrieval ceiling) | **78.5%** |
| gold in final top-5 (after reranking) | **36.7%** |

- Lost by the reranker (in pool, cut before top-5): **33** cases
- Never retrieved (absent from pool): **17** cases

The reranker retains 46.8% of the gold documents retrieval hands it. This is the single largest identified loss in the pipeline.

## Reranker bake-off

Identical cached candidate pools re-ranked by each configuration, so the comparison isolates the reranker. Source: `reranker_bakeoff.json`.

| configuration | R@1 | R@3 | R@5 | MRR | secs |
|---|---|---|---|---|---|
| ms-marco-MiniLM-L-6  @600 | 20.0% | 28.0% | 28.0% | 0.227 | 72 |
| ms-marco-MiniLM-L-6  @2000 | 16.0% | 28.0% | 28.0% | 0.213 | 134 |
| ms-marco-MiniLM-L-12 @600 | 20.0% | 24.0% | 28.0% | 0.223 | 186 |
| no reranker (raw hybrid RRF) | 12.0% | 16.0% | 24.0% | 0.151 | 0 |
| bge-reranker-base    @2000 | 8.0% | 20.0% | 20.0% | 0.133 | 877 |

**Best: ms-marco-MiniLM-L-6  @600 at 28.0% recall@5.** Configurations cluster closely, indicating reranker choice is not the limiting factor on this corpus.
