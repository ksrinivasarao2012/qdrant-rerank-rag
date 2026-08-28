"""
Script to apply verified golden dataset post ID fixes category-by-category.
Includes backup generation, validation against data/processed/posts.jsonl,
and formatted Before vs. After side-by-side inspection.
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Paths
GOLDEN_PATH = Path("evaluation/golden_dataset.json")
POSTS_PATH = Path("data/processed/posts.jsonl")

# Category 1: multi_turn fixes
MULTI_TURN_FIXES = {
    "mturn_04": {
        "new_category": "multi_hop",
        "gold_answer_ids": ["583"],
        "graded_relevance": {"583": 3},
        "reason": "Standalone comparison between AIC and BIC penalties (no pronouns)."
    },
    "mturn_06": {
        "gold_answer_ids": ["348246", "346984"],
        "graded_relevance": {"348246": 3, "346984": 3},
        "reason": "Answers whether increasing tree count causes Random Forest to overfit."
    },
    "mturn_07": {
        "gold_answer_ids": ["183062", "183011", "183012"],
        "graded_relevance": {"183062": 3, "183011": 3, "183012": 3},
        "reason": "Discusses whether MLE estimator is always unbiased in finite samples."
    },
    "mturn_08": {
        "gold_answer_ids": ["235916", "131244", "137615"],
        "graded_relevance": {"235916": 3, "131244": 3, "137615": 3},
        "reason": "Explains stationarity testing (ADF test vs. KPSS test null hypotheses and Dickey-Fuller equations)."
    },
    "mturn_09": {
        "gold_answer_ids": ["78", "294", "22126"],
        "graded_relevance": {"78": 3, "294": 3, "22126": 3},
        "reason": "Explains feature scaling / standardization choices before running PCA."
    },
    "mturn_10": {
        "gold_answer_ids": ["67210"],
        "graded_relevance": {"67210": 3},
        "reason": "Query drifted to a U-statistic + non-parametric-test definition question. Post 67210 (accepted, score 50, thread 'What exactly does a non-parametric test accomplish...') defines non-parametric tests and explicitly explains what to do with the Mann-Whitney U-statistic (count-then-scale to estimate P(X<Y)). Prior gold 85914 was a null-hypothesis-rejection answer, mismatched to the current query."
    },
    "mturn_11": {
        "new_category": "multi_hop",
        "gold_answer_ids": ["126362", "211359", "298793"],
        "graded_relevance": {"126362": 3, "211359": 3, "298793": 3},
        "reason": "Explains why deep networks prefer ReLU over sigmoid (vanishing gradients)."
    },
    "mturn_12": {
        "gold_answer_ids": ["104746", "104681"],
        "graded_relevance": {"104746": 3, "104681": 3},
        "reason": "Explains constructing confidence intervals using bootstrap samples."
    },
    "mturn_13": {
        "gold_answer_ids": ["107931"],
        "graded_relevance": {"107931": 3},
        "reason": "Explains the consequence of weak instruments in 2SLS regression."
    },
    "mturn_15": {
        "gold_answer_ids": ["200105", "352037"],
        "graded_relevance": {"200105": 3, "352037": 3},
        "reason": "Explains how learning rate decay schedules aid optimization convergence."
    },
    "mturn_16": {
        "gold_answer_ids": ["45868"],
        "graded_relevance": {"45868": 3},
        "reason": "Explains why mean imputation artificially reduces variance."
    },
    "mturn_17": {
        "new_category": "multi_hop",
        "gold_answer_ids": ["402676"],
        "graded_relevance": {"402676": 3},
        "reason": "Explains how UMAP compares to t-SNE for preserving global data structure."
    },
    "mturn_18": {
        "gold_answer_ids": ["90781", "90783", "413505"],
        "graded_relevance": {"90781": 3, "90783": 2, "413505": 3},
        "reason": "Explains how class imbalance affects ROC vs Precision-Recall curves."
    },
    "mturn_19": {
        "new_category": "multi_hop",
        "gold_answer_ids": ["66895"],
        "graded_relevance": {"66895": 3},
        "reason": "Explains when to choose random intercepts vs. random slopes in mixed models."
    },
    "mturn_21": {
        "gold_answer_ids": ["198481", "636153", "198121", "636116"],
        "graded_relevance": {"198481": 3, "636153": 3, "198121": 3, "636116": 3},
        "reason": "Explains how right-censoring affects survival analysis and Kaplan-Meier curves."
    },
    "mturn_22": {
        "gold_answer_ids": ["64486", "64481"],
        "graded_relevance": {"64486": 3, "64481": 3},
        "reason": "Explains how running multiple A/B tests at once increases the false positive rate."
    }
}

# Category 2: negation fixes
NEGATION_FIXES = {
    "neg_01": {
        "gold_answer_ids": ["1648", "74954", "52297"],
        "graded_relevance": {"1648": 3, "74954": 3, "52297": 3},
        "reason": "Provides Anderson-Darling, Jarque-Bera, D'Agostino, KS, and QQ plot alternatives to Shapiro-Wilk."
    },
    "neg_04": {
        "gold_answer_ids": ["124545", "262060", "410405"],
        "graded_relevance": {"124545": 3, "262060": 3, "410405": 3},
        "reason": "Explains non-linear alternatives to PCA: Isomap, deep bottleneck autoencoders, and UMAP/t-SNE/MDS."
    },
    "neg_05": {
        "gold_answer_ids": ["13698", "27310"],
        "graded_relevance": {"13698": 3, "27310": 3},
        "reason": "Explains modern alternatives to stepwise regression: Lasso, Elastic Net, and Random Forest feature importance."
    },
    "neg_06": {
        "gold_answer_ids": ["125016", "157379"],
        "graded_relevance": {"125016": 3, "157379": 3},
        "reason": "Provides non-ARIMA forecasting methods: Exponential Smoothing (Holt-Winters), TBATS state-space, and baselines."
    },
    "neg_07": {
        "gold_answer_ids": ["195481", "3692"],
        "graded_relevance": {"195481": 3, "3692": 3},
        "reason": "Explains clustering without specifying k: DBSCAN/HDBSCAN density reachability and Hierarchical dendrogram cutting."
    },
    "neg_08": {
        "gold_answer_ids": ["207512", "70208"],
        "graded_relevance": {"207512": 3, "70208": 3},
        "reason": "Provides goodness of fit tests excluding Kolmogorov-Smirnov: Hosmer-Lemeshow, Chi-Square, and Anderson-Darling."
    },
    "neg_10": {
        "gold_answer_ids": ["3733", "3744", "3946"],
        "graded_relevance": {"3733": 3, "3744": 3, "3946": 3},
        "reason": "Explains non-parametric rank correlation coefficients: Spearman's rho and Kendall's tau."
    },
    "neg_11": {
        "gold_answer_ids": ["394490", "77016", "50680"],
        "graded_relevance": {"394490": 3, "77016": 3, "50680": 3},
        "reason": "Explains dependency metrics not using mutual information: Distance Correlation (dCor) and Maximal Information Coefficient (MIC)."
    },
    "neg_12": {
        "gold_answer_ids": ["102645", "40474"],
        "graded_relevance": {"102645": 3, "40474": 3},
        "reason": "Explains ensemble combination methods without stacking/voting: Weighted averaging, rank blending, and Bayesian Model Averaging."
    },
    "neg_13": {
        "gold_answer_ids": ["2042", "27804"],
        "graded_relevance": {"2042": 3, "27804": 3},
        "reason": "Explains outlier detection without Isolation Forest: Mahalanobis distance / MCD, Local Outlier Factor (LOF), and Cook's distance."
    },
    "neg_14": {
        "gold_answer_ids": ["124545", "364255"],
        "graded_relevance": {"124545": 3, "364255": 3},
        "reason": "Provides non-linear dimensionality reduction methods excluding t-SNE: UMAP, Isomap, LLE, and Kernel PCA."
    },
    "neg_15": {
        "gold_answer_ids": ["71037", "194152"],
        "graded_relevance": {"71037": 3, "194152": 3},
        "reason": "Explains Bayesian diagnostics without checking ESS: Gelman-Rubin R-hat, trace plot visual checks, and Posterior Predictive Checks."
    },
    "neg_16": {
        "gold_answer_ids": ["243225", "32421"],
        "graded_relevance": {"243225": 3, "32421": 3},
        "reason": "Explains specifying and fitting hierarchical GLMM models with random intercepts only."
    },
    "neg_17": {
        "gold_answer_ids": ["531971", "472920", "222587"],
        "graded_relevance": {"531971": 3, "472920": 3, "222587": 3},
        "reason": "Explains non-LSTM sequence modeling architectures: Transformers (Self-Attention with Positional Encoding), TCNs, and GRUs."
    },
    "neg_18": {
        "gold_answer_ids": ["244023", "33320"],
        "graded_relevance": {"244023": 3, "33320": 3},
        "reason": "Explains non-Gaussian KDE kernels: Epanechnikov kernel (optimal MSE), Biweight/Triweight, and Uniform boxcar kernels."
    },
    "neg_19": {
        "gold_answer_ids": ["411775", "298951"],
        "graded_relevance": {"411775": 3, "298951": 3},
        "reason": "Explains categorical encoding without one-hot: Feature Hashing and Target/Impact Encoding."
    },
    "neg_20": {
        "gold_answer_ids": ["424127", "133694"],
        "graded_relevance": {"424127": 3, "133694": 3},
        "reason": "Explains recommender systems excluding collaborative filtering: Content-based filtering with TF-IDF and item metadata embeddings."
    }
}

# Category 3: multi_hop fixes (sourced from evaluation/multi_hop_audit_report.md,
# cross-checked against fresh golden_audit_report.json — 14 cases flagged REPLACE_GOLD).
# All candidate IDs verified present in data/processed/posts.jsonl.
MULTI_HOP_FIXES = {
    "hop_02": {
        "gold_answer_ids": ["372975", "152698"],
        "graded_relevance": {"372975": 3, "152698": 3},
        "reason": "Bagging = parallel bootstrap averaging vs Boosting = sequential residual correction of weak learners. 372975 has literal 'Bagging:'/'Boosting:' sections with pseudocode; 152698 is a decision guide for when to pick each. (Revised 2026-08-25 after 7B judge flagged the original .md report candidates 77025/348246 as too indirect.)"
    },
    "hop_03": {
        "gold_answer_ids": ["176390", "7404"],
        "graded_relevance": {"176390": 3, "7404": 3},
        "reason": "176390 ('Do underpowered studies have increased likelihood of false positives?', score 40 accepted) directly relates power to Type I error; 7404 ('How do I find the probability of a type II error?', score 33 accepted) covers Type II + beta + power quantitatively. (Revised again 2026-08-25 after 7B judge flagged 1616/14157 as mnemonic + visualization rather than the three-way relationship.)"
    },
    "hop_04": {
        "gold_answer_ids": ["55605", "253992"],
        "graded_relevance": {"55605": 3, "253992": 3},
        "reason": "F1 as the harmonic mean of precision and recall — summarizes the precision-recall trade-off."
    },
    "hop_05": {
        "gold_answer_ids": ["876", "4274"],
        "graded_relevance": {"876": 3, "4274": 3},
        "reason": "Lasso vs Ridge: L1 arbitrary selection among correlated features vs L2 grouping/shrinkage effect."
    },
    "hop_07": {
        "gold_answer_ids": ["184497", "220563"],
        "graded_relevance": {"184497": 3, "220563": 3},
        "reason": "Direct SGD-vs-Adam comparison (184497, 'Difference between GradientDescentOptimizer and AdamOptimizer', score 85 accepted) + Adam mechanism (220563, 'How does the Adam method of stochastic gradient descent work?', score 49). (Revised 2026-08-25 after 7B judge flagged the .md report's 352037 as generic 'network doesn't learn' troubleshooting rather than an SGD-vs-Adam comparison.)"
    },
    "hop_09": {
        "gold_answer_ids": ["77025", "148060"],
        "graded_relevance": {"77025": 3, "148060": 3},
        "reason": "77025 ('Is random forest a boosting algorithm?', score 105 accepted) explicitly frames RF as variance-reduction bagging vs boosting as bias-reduction — exactly the query's dimensions. 148060 covers RF vs Boosting parametric properties. (Revised 2026-08-25 after 7B judge flagged 348246 as a hyperparameter-tuning post.)"
    },
    "hop_11": {
        "gold_answer_ids": ["357974", "409271"],
        "graded_relevance": {"357974": 3, "409271": 3},
        "reason": "H(p,q) = H(p) + D_KL(p||q): cross-entropy = entropy + KL divergence, used interchangeably in classification loss."
    },
    "hop_12": {
        "gold_answer_ids": ["66294", "214315"],
        "graded_relevance": {"66294": 3, "214315": 3},
        "reason": "Random Forest decorrelates trees via random feature subsets (mtry), reducing ensemble variance vs plain bagging."
    },
    "hop_14": {
        "gold_answer_ids": ["402676", "364255"],
        "graded_relevance": {"402676": 3, "364255": 3},
        "reason": "t-SNE (Student-t local probabilities) vs UMAP (fuzzy simplicial sets) — differences in preserving global structure."
    },
    "hop_16": {
        "gold_answer_ids": ["341566", "27870"],
        "graded_relevance": {"341566": 3, "27870": 3},
        "reason": "341566 explains the interplay between random intercept and random slope (correlation under treatment coding, with illustrative pictures); 27870 walks through fitting a mixed model with both random slope and random intercept. (Revised 2026-08-25 after 7B judge flagged 243225 as a convergence-warning post unrelated to the intercept-vs-slope comparison.)"
    },
    "hop_17": {
        "gold_answer_ids": ["531971", "472920"],
        "graded_relevance": {"531971": 3, "472920": 3},
        "reason": "Transformer self-attention O(1) path length vs RNN sequential recurrence O(n) with vanishing gradients on long sequences."
    },
    "hop_18": {
        "gold_answer_ids": ["3954", "19209"],
        "graded_relevance": {"3954": 3, "19209": 3},
        "reason": "3954 is the canonical SVM primer (score 111 accepted); 19209 ('Why bother with the dual problem when fitting SVM?', score 60 accepted) explains dual formulation which is the mathematical machinery the kernel trick exploits. (Revised again 2026-08-25 after 7B judge flagged 2168 as too brief and 3954 alone as generic SVM workflow; dual-problem post is a stronger kernel-trick-relevance signal.)"
    },
    "hop_19": {
        "gold_answer_ids": ["550708", "614855"],
        "graded_relevance": {"550708": 3, "614855": 3},
        "reason": "Target encoding vs one-hot encoding for high-cardinality categoricals: dimensionality explosion vs target leakage / smoothing."
    },
    # hop_20 (Collaborative vs content-based filtering): REMOVED 2026-08-25 —
    # user decision to delete rather than flag. Corpus (stats.stackexchange.com)
    # has abundant CF material but almost no content-based-filtering posts, so
    # no gold pair on this corpus can satisfy the query. See PROJECT_LOG.md.
}

# Category 4: niche_topic fixes (sourced from evaluation/niche_topic_audit_report.md —
# 17 cases flagged REPLACE_GOLD). All candidate IDs verified present in posts.jsonl.
NICHE_TOPIC_FIXES = {
    "niche_01": {
        "gold_answer_ids": ["96741", "357498"],
        "graded_relevance": {"96741": 3, "357498": 3},
        "reason": "Cox PH ties: Efron approximation vs Breslow method — survival analysis tie handling."
    },
    "niche_02": {
        "gold_answer_ids": ["63497", "21225"],
        "graded_relevance": {"63497": 3, "21225": 3},
        "reason": "Benjamini-Hochberg step-up procedure P_(i) <= (i/m) Q controlling the false discovery rate."
    },
    "niche_03": {
        "gold_answer_ids": ["30205", "573589"],
        "graded_relevance": {"30205": 3, "573589": 3},
        "reason": "Sklar's theorem: copulas decompose a joint distribution into uniform marginals + a dependency structure."
    },
    "niche_04": {
        "gold_answer_ids": ["118227", "18772"],
        "graded_relevance": {"118227": 3, "18772": 3},
        "reason": "118227 ('Model building and selection using Hosmer et al. 2013') and 18772 ('Hosmer-Lemeshow vs AIC for logistic regression') both discuss the HL test directly. (Revised 2026-08-25 after 7B judge flagged 3562 as an R^2 post rather than HL-specific.)"
    },
    "niche_07": {
        "gold_answer_ids": ["62147", "20549"],
        "graded_relevance": {"62147": 3, "20549": 3},
        "reason": "62147 is the canonical Mahalanobis explainer (score 255 accepted); 20549 covers the distribution of an observation-level Mahalanobis distance. (Revised 2026-08-25 after 7B judge flagged the pairing with 117463 (ZCA whitening) as off-topic.)"
    },
    "niche_10": {
        "gold_answer_ids": ["236104", "504927"],
        "graded_relevance": {"236104": 3, "504927": 3},
        "reason": "236104 title is literally 'EM-algorithm and missing data'; 504927 walks through EM MLE on bivariate normal WITH missing data. (Revised 2026-08-25 after 7B judge flagged the .md report's 262560/628785 as EM-vs-gradient-descent rather than missing-data-specific.)"
    },
    "niche_11": {
        "gold_answer_ids": ["244902", "104817"],
        "graded_relevance": {"244902": 3, "104817": 3},
        "reason": "244902 title is literally 'When would one use Gibbs sampling instead of Metropolis-Hastings?' — exact query match. 104817 'Gibbs sampling versus general MH-MCMC' is the same comparison. (Revised 2026-08-25 after 7B judge flagged 207 (generic MCMC layperson explainer) and 27374 (LR vs BF) as off-topic.)"
    },
    "niche_12": {
        "gold_answer_ids": ["126661", "65657"],
        "graded_relevance": {"126661": 3, "65657": 3},
        "reason": "126661 'How to use the Hausman test for gender discrimination?' and 65657 'Hausman test for panel data, fe and re' are both Hausman-specific with worked examples. (Revised 2026-08-25 after 7B judge flagged 66895/90759 as generic mixed-effects rather than Hausman-specific.)"
    },
    "niche_13": {
        "gold_answer_ids": ["174144", "176929"],
        "graded_relevance": {"174144": 3, "176929": 3},
        "reason": "Tweedie GLM: compound Poisson-gamma distribution for continuous non-negative data with a point mass at zero."
    },
    "niche_15": {
        "gold_answer_ids": ["161080", "226123"],
        "graded_relevance": {"161080": 3, "226123": 3},
        "reason": "161080 ('Difference between LOESS and LOWESS') and 226123 ('Explanation of what Nate Silver said about loess') focus on LOESS/local regression directly. (Revised 2026-08-25 after 7B judge flagged 25081 (splines) and 517784 (spline interpolation) as not local-linear-specific.)"
    },
    # niche_16 (ARCH vs GARCH specifically): REMOVED 2026-08-25 —
    # user decision, along with 3 other niche corpus gaps (niche_01, 19, 21).
    # Corpus lacks a direct ARCH-vs-GARCH comparison (posts pair with ARMA
    # instead). See PROJECT_LOG.md.
    "niche_17": {
        "gold_answer_ids": ["148290", "192165"],
        "graded_relevance": {"148290": 3, "192165": 3},
        "reason": "Ljung-Box: portmanteau test pooling sample autocorrelations up to lag h against white-noise null."
    },
    # niche_19 (DBSCAN core points vs k-means centroids): REMOVED 2026-08-25 —
    # user decision, along with 3 other niche corpus gaps. Corpus discusses each
    # algorithm separately but has no direct side-by-side comparison. See PROJECT_LOG.md.
    "niche_21": {
        "gold_answer_ids": ["324692", "243014"],
        "graded_relevance": {"324692": 3, "243014": 3},
        "reason": "Meta-analysis heterogeneity: Cochran's Q chi-square, I^2 percentage of variation across studies, forest-plot CI overlap."
    },
    "niche_22": {
        "gold_answer_ids": ["74127", "36146"],
        "graded_relevance": {"74127": 3, "36146": 3},
        "reason": "Moran's I with spatial weights matrix W measuring global spatial autocorrelation (clustering vs dispersion)."
    },
    "niche_23": {
        "gold_answer_ids": ["27374", "204334"],
        "graded_relevance": {"27374": 3, "204334": 3},
        "reason": "27374 ('Likelihood ratio vs Bayes Factor', score 49 accepted) frames BF as the marginal-likelihood ratio for model comparison; 204334 ('Can I make a decision using a Bayes factor?', score 10 accepted) covers the decision-making side. (Revised 2026-08-25 after 7B judge flagged 201591 (alternatives to p-values) as too broad; 27374 retained since it IS the direct comparison ratio explanation.)"
    },
    "niche_24": {
        "gold_answer_ids": ["21507", "63418"],
        "graded_relevance": {"21507": 3, "63418": 3},
        "reason": "21507 ('Introduction to structural equation modeling', score 22 accepted) is a canonical SEM primer that contextualizes path analysis; 63418 ('Difference Between Simultaneous Equation Model and Structural Equation Model', score 16 accepted) is the closest direct comparison. (Revised 2026-08-25 after 7B judge flagged 376925/43502 as Pearl-causality posts rather than SEM-vs-path.)"
    }
}

def load_posts_index(needed_ids):
    """Load only requested post IDs from posts.jsonl for fast inspection."""
    posts_map = {}
    with open(POSTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            aid = str(p.get("answer_id"))
            if aid in needed_ids:
                posts_map[aid] = {
                    "answer_id": aid,
                    "score": p.get("score", 0),
                    "question_title": p.get("question_title", "No Title"),
                    "answer_text": p.get("answer_text", "")
                }
    return posts_map

def apply_category_fixes(category_name, fixes_dict):
    print(f"\n=======================================================")
    print(f" APPLYING FIXES FOR CATEGORY: {category_name.upper()}")
    print(f"=======================================================\n")
    
    # 1. Backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = Path(f"evaluation/golden_dataset.backup_{category_name}_{timestamp}.json")
    shutil.copyfile(GOLDEN_PATH, backup_file)
    print(f"Safety backup created at: {backup_file}\n")
    
    # 2. Load dataset
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    # 3. Gather all old and new post IDs to fetch from posts.jsonl
    all_needed_ids = set()
    for case in cases:
        qid = case["query_id"]
        if qid in fixes_dict:
            for old_id in case.get("gold_answer_ids", []):
                all_needed_ids.add(str(old_id))
            for new_id in fixes_dict[qid].get("gold_answer_ids", []):
                all_needed_ids.add(str(new_id))
                
    print(f"Fetching {len(all_needed_ids)} post IDs from {POSTS_PATH} for verification...")
    posts_db = load_posts_index(all_needed_ids)
    print(f"Successfully loaded {len(posts_db)} posts from database.\n")
    
    # 4. Apply fixes and print side-by-side comparisons
    updated_count = 0
    for idx, case in enumerate(cases):
        qid = case["query_id"]
        if qid in fixes_dict:
            fix = fixes_dict[qid]
            old_category = case.get("category")
            old_gold_ids = [str(x) for x in case.get("gold_answer_ids", [])]
            
            # Update fields
            if "new_category" in fix:
                case["category"] = fix["new_category"]
            case["gold_answer_ids"] = fix["gold_answer_ids"]
            case["graded_relevance"] = fix["graded_relevance"]
            updated_count += 1
            
            # Print side-by-side proof
            query = case.get("query", "")
            print("--------------------------------------------------------------------------------")
            print(f"[{updated_count}] Query ID: {qid}")
            print(f"QUESTION: \"{query}\"")
            if "chat_history" in case and case["chat_history"]:
                last_turn = case["chat_history"][-1]["content"]
                print(f"CONTEXT / PREVIOUS TURN: \"{last_turn}\"")
            print(f"FIX REASON: {fix['reason']}")
            
            print("\n  [BEFORE] Old Gold Post(s):")
            for oid in old_gold_ids:
                post = posts_db.get(oid)
                if post:
                    snippet = post["answer_text"][:140].replace("\n", " ")
                    print(f"    - Post ID: {oid} | Score: {post['score']} | Title: \"{post['question_title']}\"")
                    print(f"      Snippet: \"{snippet}...\"")
                else:
                    print(f"    - Post ID: {oid} (NOT FOUND in database)")
                    
            print("\n  [AFTER - DATABASE PROOF] New Clean Gold Post(s):")
            for nid in fix["gold_answer_ids"]:
                post = posts_db.get(str(nid))
                if post:
                    snippet = post["answer_text"][:160].replace("\n", " ")
                    print(f"    - Post ID: {nid} | Score: {post['score']} | Title: \"{post['question_title']}\"")
                    print(f"      Snippet: \"{snippet}...\"")
                else:
                    print(f"    - Post ID: {nid} (WARNING: NOT FOUND in database)")
            print("--------------------------------------------------------------------------------\n")
            
    # 5. Save updated dataset
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully updated {updated_count} cases for category '{category_name}'.")
    print(f"Cleaned dataset saved to: {GOLDEN_PATH}\n")

if __name__ == "__main__":
    category = "multi_turn"
    if len(sys.argv) > 2 and sys.argv[1] == "--category":
        category = sys.argv[2]
        
    if category == "multi_turn":
        apply_category_fixes("multi_turn", MULTI_TURN_FIXES)
    elif category == "negation":
        apply_category_fixes("negation", NEGATION_FIXES)
    elif category == "multi_hop":
        apply_category_fixes("multi_hop", MULTI_HOP_FIXES)
    elif category == "niche_topic":
        apply_category_fixes("niche_topic", NICHE_TOPIC_FIXES)
    else:
        print(f"Category '{category}' not yet configured in script.")
