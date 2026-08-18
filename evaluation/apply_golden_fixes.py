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
        "gold_answer_ids": ["183062"],
        "graded_relevance": {"183062": 3},
        "reason": "Discusses whether MLE estimator is always unbiased in finite samples."
    },
    "mturn_08": {
        "gold_answer_ids": ["235916"],
        "graded_relevance": {"235916": 3},
        "reason": "Explains stationarity testing (ADF test vs. KPSS test null hypotheses)."
    },
    "mturn_09": {
        "gold_answer_ids": ["78"],
        "graded_relevance": {"78": 3},
        "reason": "Explains feature scaling / standardization choices before running PCA."
    },
    "mturn_10": {
        "gold_answer_ids": ["85914"],
        "graded_relevance": {"85914": 3},
        "reason": "Explains what rejecting the null hypothesis tells us about the alternative hypothesis."
    },
    "mturn_11": {
        "new_category": "multi_hop",
        "gold_answer_ids": ["126362", "211359", "298793"],
        "graded_relevance": {"126362": 3, "211359": 3, "298793": 3},
        "reason": "Explains why deep networks prefer ReLU over sigmoid (vanishing gradients)."
    },
    "mturn_12": {
        "gold_answer_ids": ["104746"],
        "graded_relevance": {"104746": 3},
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
    "mturn_19": {
        "gold_answer_ids": ["66895"],
        "graded_relevance": {"66895": 3},
        "reason": "Explains when to choose random intercepts vs. random slopes in mixed models."
    },
    "mturn_21": {
        "gold_answer_ids": ["198481", "636153"],
        "graded_relevance": {"198481": 3, "636153": 3},
        "reason": "Explains how right-censoring affects survival analysis and Kaplan-Meier curves."
    },
    "mturn_22": {
        "gold_answer_ids": ["64486"],
        "graded_relevance": {"64486": 3},
        "reason": "Explains how running multiple A/B tests at once increases the false positive rate."
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
    else:
        print(f"Category '{category}' not yet configured in script.")
