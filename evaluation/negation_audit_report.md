# Golden Dataset Audit Report: Negation Category

We have systematically verified all **20 queries** in the `negation` category to check if the gold answers cover the user's queries, if they respect the negative/exclusion constraints, and whether concept mismatches exist.

This report records the findings of all **20 cases** evaluated by the local `Qwen2.5-7B-Instruct` judge, along with candidate replacement post IDs from the StackExchange database.

---

## 1. Executive Summary

* **Total Cases Audited**: 20
* **Aligned Cases (`[+] OK`)**: 4 (20.0%)
* **Mismatched / Flagged Cases (`[-] MISMATCH`)**: 16 (80.0%)

### Core Root Cause:
The `negation` queries explicitly ask for alternatives (e.g. *"without using PCA"*, *"excluding ARIMA"*, *"without stepwise regression"*). In 16 out of 20 cases, the seeded gold answers either discussed the exact forbidden concept itself, offered theoretical definitions without answering the alternative method, or were completely off-topic.

---

## 2. Aligned Cases (`[+] OK` - No Action Needed)

1. **`neg_02`**: *"How do I handle missing data without deleting rows?"* -> **`[+] OK`** (Gold ID: `10352`)
2. **`neg_03`**: *"How to compare group means without assuming normal distribution?"* -> **`[+] OK`** (Gold ID: `8849`)
3. **`neg_04`**: *"How to reduce dimensionality without using PCA?"* -> **`[+] OK`** (Gold ID: `325002`)
4. **`neg_09`**: *"How is parameter uncertainty estimated without bootstrapping?"* -> **`[+] OK`** (Gold ID: `21888`)

---

## 3. Mismatched Cases Checklist & Replacement Candidates

All reasons below are recorded **100% verbatim** from the LLM judge logs.

---

### Case `neg_01` (Normality test without Shapiro-Wilk)
* **Query**: `"How to test normality without using the Shapiro-Wilk test?"`
* **Current Gold ID**: `["429"]`
* **LLM Reason (Verbatim)**: `The query asks for methods to test normality without using the Shapiro-Wilk test, but the provided answer does not offer any alternative methods. Instead, it discusses the differences between the Shapiro-Wilk test and the Kolmogorov-Smirnov test, which does not directly answer the query.`
* **Candidate Gold IDs**: `["30053", "74954"]` (Score 136, 33: Anderson-Darling, Jarque-Bera, and graphical diagnostics for normality).
* **Suggested Relevance**: `{"30053": 3}`

---

### Case `neg_05` (Feature selection without stepwise regression)
* **Query**: `"How to select model features without stepwise regression?"`
* **Current Gold ID**: `["213177"]`
* **LLM Reason (Verbatim)**: `The query asks for methods to select model features without using stepwise regression, but the answer does not provide any alternative methods for feature selection. Instead, it discusses the inappropriateness of using LASSO or elastic net for variable selection followed by model fitting on the same data, which is not directly addressing the query.`
* **Candidate Gold IDs**: `["13698", "27310"]` (Score 61: *"What are modern, easily used alternatives to stepwise regression?"*).
* **Suggested Relevance**: `{"13698": 3}`

---

### Case `neg_06` (Forecasting time series without ARIMA)
* **Query**: `"How to forecast time series data without using ARIMA models?"`
* **Current Gold ID**: `["4586"]`
* **LLM Reason (Verbatim)**: `The query asks for methods to forecast time series data without using ARIMA models, but the answer discusses minimizing forecast errors, maximum likelihood estimation, and ARIMA(0,1,1) models, which are all related to ARIMA and not alternatives to it.`
* **Candidate Gold IDs**: `["125016", "157379"]` (Score 54, 46: Exponential smoothing, simple baseline forecasting, and state-space alternatives).
* **Suggested Relevance**: `{"125016": 3}`

---

### Case `neg_07` (Clustering without specifying k)
* **Query**: `"Clustering methods that do not require specifying the number of clusters k?"`
* **Current Gold ID**: `["241387"]`
* **LLM Reason (Verbatim)**: `The answer discusses hierarchical clustering, which does not require specifying the number of clusters k, but it does not directly address other clustering methods that do not require specifying k. The answer also does not provide a comprehensive list of such methods.`
* **Candidate Gold IDs**: `["195481", "3692"]` (Score 74, 58: Selecting clustering methods, DBSCAN, HDBSCAN, and cutting dendrograms).
* **Suggested Relevance**: `{"195481": 3}`

---

### Case `neg_08` (Goodness of fit tests excluding Kolmogorov-Smirnov)
* **Query**: `"Goodness of fit tests excluding Kolmogorov-Smirnov?"`
* **Current Gold ID**: `["215"]`
* **LLM Reason (Verbatim)**: `The query specifically asks for goodness of fit tests excluding Kolmogorov-Smirnov, but the answer does not provide any specific tests or methods that fit this criteria. Instead, it discusses the purpose of fitting distributions and the potential approaches to doing so, which is not directly related to the query.`
* **Candidate Gold IDs**: `["207512", "70208"]` (Score 30, 28: Hosmer-Lemeshow, Anderson-Darling, and Bayesian goodness of fit tests).
* **Suggested Relevance**: `{"207512": 3}`

---

### Case `neg_10` (Nonparametric correlation other than Pearson)
* **Query**: `"Nonparametric correlation coefficients other than Pearson correlation?"`
* **Current Gold ID**: `["1004"]`
* **LLM Reason (Verbatim)**: `The query asks for nonparametric correlation coefficients, but the answer suggests using the Kolmogorov–Smirnov test, which is a goodness-of-fit test, not a correlation test. The answer also mentions an R function ks.test, which is related to the test, but does not address the query's request for correlation coefficients.`
* **Candidate Gold IDs**: `["3733", "2128"]` (Score 114: *"Pearson's or Spearman's correlation with non-normal data"* - covers Spearman's rho and Kendall's tau).
* **Suggested Relevance**: `{"3733": 3}`

---

### Case `neg_11` (Information theory correlation without mutual information)
* **Query**: `"Information theoretic measure of dependency not using mutual information?"`
* **Current Gold ID**: `["6498"]`
* **LLM Reason (Verbatim)**: `The query asks for an information theory metric for correlation without using mutual information, but the answer provides a link to a PDF without any specific information or metric mentioned in the query context.`
* **Candidate Gold IDs**: `["99191", "20856"]` (Distance correlation, maximal information coefficient alternatives).
* **Suggested Relevance**: `{"99191": 3}`

---

### Case `neg_12` (Ensemble prediction without stacking or voting)
* **Query**: `"How to combine ensemble models without using stacking or voting?"`
* **Current Gold ID**: `["88020"]`
* **LLM Reason (Verbatim)**: `The query asks about ensemble prediction combining models without stacking or voting, but the answer does not address this specific topic. Instead, it discusses boosting and bagging, their risks, and how they handle certain types of data issues. It does not provide information on how to combine models without using stacking or voting techniques.`
* **Candidate Gold IDs**: `["102645", "40474"]` (Score 31, 20: Simple blending, weighted linear combinations, and cross-validated out-of-fold averaging).
* **Suggested Relevance**: `{"102645": 3}`

---

### Case `neg_13` (Outlier detection without Isolation Forest)
* **Query**: `"Outlier detection methods that do not use Isolation Forests?"`
* **Current Gold ID**: `["14722"]`
* **LLM Reason (Verbatim)**: `The answer does not directly address the query about outlier detection methods that do not use Isolation Forests. Instead, it discusses the concept of multiple comparisons and how sample size can affect the detection of outliers, which is tangential to the specific method requested.`
* **Candidate Gold IDs**: `["2042", "27804"]` (Score 32, 21: *"What is the best way to identify outliers in multivariate data?"* - Mahalanobis distance, Local Outlier Factor, Cook's distance).
* **Suggested Relevance**: `{"2042": 3}`

---

### Case `neg_14` (Nonlinear dimensionality reduction excluding t-SNE)
* **Query**: `"Non-linear dimensionality reduction methods excluding t-SNE?"`
* **Current Gold ID**: `["140"]`
* **LLM Reason (Verbatim)**: `The query asks for non-linear dimensionality reduction methods excluding t-SNE, but the answer does not mention any specific non-linear methods and instead discusses eigenvalues and PCA, which are linear methods.`
* **Candidate Gold IDs**: `["124545", "364255"]` (Score 56, 18: *"How to understand 'nonlinear' as in 'nonlinear dimensionality reduction'?"* - Isomap, LLE, UMAP, Kernel PCA).
* **Suggested Relevance**: `{"124545": 3}`

---

### Case `neg_15` (Bayesian diagnostics without effective sample size)
* **Query**: `"Bayesian model convergence diagnostics without checking effective sample size?"`
* **Current Gold ID**: `["18567"]`
* **LLM Reason (Verbatim)**: `The query asks about Bayesian model diagnostics without checking effective sample size, but the answer does not address this topic. Instead, it provides advice on how to modify a model to include predicted regressors and suggests a reference for posterior predictive simulations.`
* **Candidate Gold IDs**: `["71037", "194152"]` (Score 72: Gelman-Rubin R-hat, traceplots, autocorrelation plots, posterior predictive checks).
* **Suggested Relevance**: `{"71037": 3}`

---

### Case `neg_16` (Hierarchical models without random slopes)
* **Query**: `"Fitting hierarchical models with random intercepts only, without random slopes?"`
* **Current Gold ID**: `["15730"]`
* **LLM Reason (Verbatim)**: `The query asks about fitting hierarchical models without random slopes, but the answer discusses random effects, fixed effects, and degrees of freedom in mixed effects models. It does not directly address the query about fitting hierarchical models without random slopes.`
* **Candidate Gold IDs**: `["243225", "32421"]` (Score 95, 74: Random intercept models, comparing models with vs without random slopes using ANOVA/LRT).
* **Suggested Relevance**: `{"243225": 3}`

---

### Case `neg_17` (Deep learning sequence modeling without LSTMs)
* **Query**: `"Sequence modeling architectures in deep learning that do not use LSTMs?"`
* **Current Gold ID**: `["1081"]`
* **LLM Reason (Verbatim)**: `The query asks for deep learning sequence modeling without using LSTMs, but the provided answer discusses probability mass functions and scaling of state spaces, which is not related to deep learning or sequence modeling techniques.`
* **Candidate Gold IDs**: `["531971", "472920", "222587"]` (Score 64, 40: Transformers / Self-Attention mechanism, Temporal Convolutional Networks).
* **Suggested Relevance**: `{"531971": 3}`

---

### Case `neg_18` (Kernel density estimation without Gaussian kernels)
* **Query**: `"Non-parametric kernel density estimation without Gaussian kernels?"`
* **Current Gold ID**: `["2500"]`
* **LLM Reason (Verbatim)**: `The query asks about non-parametric kernel density estimation without Gaussian kernels, but the answer provides a detailed derivation for the Epanechnikov kernel, which is a specific type of kernel. It does not address the general case of non-parametric kernel density estimation without Gaussian kernels.`
* **Candidate Gold IDs**: `["244023", "33320"]` (Score 76: Parzen window kernel density estimation with Epanechnikov, biweight, and triangular kernels).
* **Suggested Relevance**: `{"244023": 3}`

---

### Case `neg_19` (Categorical encoding without one-hot encoding)
* **Query**: `"Categorical feature encoding strategies without using one-hot encoding?"`
* **Current Gold ID**: `["1245"]`
* **LLM Reason (Verbatim)**: `The query asks about feature engineering category mapping without one-hot encoding, but the answer discusses creating dummy variables and mentions collinearity, which is related but does not directly address the query's request for an alternative to one-hot encoding.`
* **Candidate Gold IDs**: `["72271", "420730"]` (Target encoding, frequency encoding, ordinal mapping, entity embeddings).
* **Suggested Relevance**: `{"72271": 3}`

---

### Case `neg_20` (Recommender systems excluding collaborative filtering)
* **Query**: `"Recommender system algorithms excluding collaborative filtering?"`
* **Current Gold ID**: `["78"]`
* **LLM Reason (Verbatim)**: `The query asks for recommendation system algorithms excluding collaborative filtering, but the answer discusses standardizing variables using z-scores, which is unrelated to recommendation systems or collaborative filtering.`
* **Candidate Gold IDs**: `["424127", "133694"]` (Content-based filtering, knowledge-based systems, tf-idf item profiling).
* **Suggested Relevance**: `{"424127": 3}`
