# Golden dataset changelog

## v2 - 2026-09-01T16:20:10

Derived from `golden_dataset.json` (v1). 39 of 79 cases in the four hard
categories were flagged by `dataset_triage.py` on deterministic red flags
(weak post score, no query/title term overlap, never retrieved into an
80-100 candidate pool, stub-length answer, gold shared across queries,
unverified). All 39 were then reviewed against the query and the documents
retrieval actually returns.

**Review outcome:** 26 KEEP (flag was a false positive), 9 ADD, 4 REPLACE.
So ~84% of labels were already sound; ~5% were genuine mislabels.

Labels were proposed by an LLM assistant with quoted evidence and approved
by the author (`method: llm_proposed_human_approved`). This is weaker than
two independent annotators with measured agreement, and is recorded as such.

| case | action | before | after | rationale |
|---|---|---|---|---|
| `hop_02` | ADD | 372975, 152698 | 372975, 152698, 171296 | [171296] "How to combine weak classifiers to get a strong one?" matches "how they combine weak learners" precisely. |
| `hop_04` | ADD | 55605, 253992 | 55605, 253992, 183510 | [183510] "How to control trade-off between precision and recall? ... Fbeta score" covers both halves of the query. |
| `hop_06` | ADD | 78, 1577, 1579, 1584, 3371, 44453, 83599, 94601, 133806 | 78, 1577, 1579, 1584, 3371, 44453, 83599, 94601, 133806, 95106 | [95106] "How does Factor Analysis explain the covariance while PCA explains the variance?" is exactly the variance-handling question. |
| `hop_12` | REPLACE | 66294, 214315 | 454758 | Golds were "Motivation behind random forest steps" and "importance of the i.i.d. assumption" - the latter is unrelated. [454758] "Decision tree-based  |
| `mturn_12` | ADD | 104746 | 104746, 17104 | [17104] "Estimate confidence interval of mean by bootstrap t method or simply by bootstrap?" is the specific CI-construction answer. |
| `mturn_19` | ADD | 66895 | 66895, 485218 | [485218] "fixed effects vs random effects vs random intercept model" - closest available; corpus coverage of intercept-vs-slope is genuinely thin. |
| `neg_02` | ADD | 10352 | 10352, 1404 | [1404] "Techniques for Handling Incomplete/Missing Data" is the canonical answer; gold [10352] is a reply to one user situation. |
| `neg_03` | REPLACE | 8849 | 15694 | Gold [8849] (score 0, 2 sentences) is about Kruskal-Wallis homoscedasticity, not comparing group means. [15694] "How to test for differences between t |
| `neg_06` | ADD | 125016, 157379 | 125016, 157379, 312407 | [312407] "Alternatives to Using ARIMA for forecasting" is an exact match; gold [157379] is a book list. |
| `neg_16` | REPLACE | 243225, 32421 | 485093 | Gold [243225] is about lmer convergence warnings, not random slopes. [485093] "Is it a must to include a random slope in a mixed model?" directly addr |
| `niche_08` | ADD | 306000 | 306000, 306892 | [306892] describes the downhill simplex algorithm itself; gold covers only stopping criteria. |
| `niche_14` | ADD | 38947 | 38947, 402241 | [402241] on Heckman correction and selection bias is stronger than gold [38947] (score 2, DiD-specific). |
| `niche_15` | REPLACE | 161080, 226123 | 408346 | Golds covered LOESS-vs-LOWESS naming and Nate Silver polynomial tails, neither explains the concept. [408346] "What is the difference between Local Li |

### Not changed

The other 26 flagged cases were verified correct. Notable false positives:

- `mturn_07` - NO_OVERLAP fired because the query says *MLE* and the title says
  *Maximum Likelihood Estimation*. The label is correct.
- `hop_01` - SHARED_GOLD fired on nine answers to one canonical AIC/BIC question.
  That is correct multi-answer coverage, not duplication.
- `neg_05` - gold [13698] *"What are modern, easily used alternatives to stepwise
  regression?"* (score 61) is an exact match for the query. It was flagged
  UNREACHABLE because retrieval failed to surface it within 80 candidates.
  **This is a retrieval defect, not a labelling defect.**
