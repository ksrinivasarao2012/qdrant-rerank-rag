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
        "gold_answer_ids": ["85914", "85908"],
        "graded_relevance": {"85914": 3, "85908": 3},
        "reason": "Explains what rejecting the null hypothesis tells us about the alternative hypothesis."
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
    else:
        print(f"Category '{category}' not yet configured in script.")
