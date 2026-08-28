"""
Systematic Golden Dataset Audit Script.
Uses the local Qwen2.5-7B-Instruct model to verify that each curated case's gold answer
is topically correct, accurate, and covers the query.
"""

import sys
import json
import os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Set threads
for _v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{_v}_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.judge_model import get_judge
from evaluation.eval_contextual_recall import load_gold_posts_map

GOLDEN_JSON_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
POSTS_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "posts.jsonl"
AUDIT_REPORT_PATH = PROJECT_ROOT / "evaluation" / "golden_audit_report.json"


def main():
    if not GOLDEN_JSON_PATH.exists():
        print("Golden dataset not found.")
        return 1

    with open(GOLDEN_JSON_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None, help="Specific category to audit.")
    parser.add_argument("--query_id", type=str, default=None, help="Specific query_id to audit (e.g. mturn_16).")
    args = parser.parse_args()

    if args.query_id:
        evaluable = [c for c in cases if c.get("query_id") == args.query_id]
    elif args.category:
        selected_cats = {c.strip() for c in args.category.split(",")}
        evaluable = [c for c in cases if c.get("category") in selected_cats]
    else:
        curated_categories = {"multi_hop", "niche_topic", "paraphrase_group", "negation", "multi_turn"}
        evaluable = [c for c in cases if c.get("category") in curated_categories]

    print(f"Total golden dataset size: {len(cases)} cases.")
    print(f"Curated cases to verify: {len(evaluable)} cases.")

    needed_ids = {str(gid) for c in evaluable for gid in c.get("gold_answer_ids", [])}
    print(f"Loading answer texts for {len(needed_ids)} post IDs...")
    gold_map = load_gold_posts_map(POSTS_JSONL_PATH, needed_ids)

    print("Initializing LLM Judge (Qwen2.5-7B-Instruct)...")
    judge = get_judge()

    report = []
    mismatches_count = 0

    print("\nStarting systematic audit of curated cases:")
    for idx, case in enumerate(evaluable, start=1):
        qid = case["query_id"]
        category = case["category"]
        query = case.get("query")
        if not query and "variants" in case:
            query = case["variants"][0]
        history = case.get("chat_history", [])
        gold_ids = case.get("gold_answer_ids", [])

        # Fetch gold texts
        gold_texts = [gold_map.get(str(gid), "") for gid in gold_ids if gold_map.get(str(gid))]
        expected_output = "\n\n".join(gold_texts) if gold_texts else None

        if not expected_output:
            print(f"  [{qid}] SKIP: No answer text found in posts.jsonl")
            continue

        # Build prompt for LLM judge
        history_str = ""
        if history:
            history_str = "\n".join(f"- {m['role'].upper()}: {m['content']}" for m in history)
            history_str = f"\nConversation History:\n{history_str}\n"

        prompt = f"""You are a professional statistics and data science audit assistant.
Your task is to verify if the provided StackExchange gold answer is a correct, direct, and complete response to the user's query.

User Query: "{query}"
{history_str}
Gold Answer Text:
\"\"\"
{expected_output}
\"\"\"

Analyze the query and the answer:
1. Is the answer topically aligned with the query? (e.g. if the query asks for "imputation methods", does the answer actually discuss imputation, rather than deletion?)
2. Is the answer factually correct and helpful?
3. Are there any concept mismatches or keyword collisions?

Format your response strictly as a JSON object with these exact keys:
{{
  "is_aligned": true or false,
  "reason": "Brief explanation of alignment or mismatch details",
  "suggested_action": "NO_ACTION" or "CORRECT_QUERY" or "REPLACE_GOLD" or "ADD_GOLDS"
}}
JSON:"""

        try:
            # Call judge (LocalGGUFJudge takes a string and returns a string)
            res_text = judge.generate(prompt).strip()
            
            # Clean JSON if model returned markdown blocks
            clean_text = res_text
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()

            try:
                result = json.loads(clean_text)
            except Exception as parse_err:
                # Regex fallback for nested unescaped quotes
                import re
                result = {}
                # 1. is_aligned
                aligned_match = re.search(r'"is_aligned"\s*:\s*(true|false)', clean_text, re.IGNORECASE)
                if aligned_match:
                    result["is_aligned"] = aligned_match.group(1).lower() == "true"
                else:
                    result["is_aligned"] = False
                
                # 2. suggested_action
                action_match = re.search(r'"suggested_action"\s*:\s*"([^"]+)"', clean_text)
                if action_match:
                    result["suggested_action"] = action_match.group(1)
                else:
                    result["suggested_action"] = "NO_ACTION"
                
                # 3. reason
                reason_match = re.search(r'"reason"\s*:\s*"(.*)"\s*,\s*"suggested_action"', clean_text, re.DOTALL)
                if not reason_match:
                    reason_match = re.search(r'"reason"\s*:\s*"(.*)"\s*\}\s*$', clean_text, re.DOTALL)
                if not reason_match:
                    reason_match = re.search(r'"reason"\s*:\s*"(.*)"', clean_text, re.DOTALL)
                
                if reason_match:
                    reason_val = reason_match.group(1).strip()
                    if reason_val.endswith('"'):
                        reason_val = reason_val[:-1]
                    # Also strip any trailing comma/key residues
                    if '","suggested_action"' in reason_val:
                        reason_val = reason_val.split('","suggested_action"')[0]
                    result["reason"] = reason_val
                else:
                    result["reason"] = f"Regex parsing fallback failed. Raw: {clean_text}"
        except Exception as e:
            result = {
                "is_aligned": False,
                "reason": f"Failed to parse LLM response: {e}. Raw response: {res_text if 'res_text' in locals() else 'None'}",
                "suggested_action": "RETRY"
            }

        is_aligned = result.get("is_aligned", False)
        reason = result.get("reason", "Unknown")
        action = result.get("suggested_action", "NO_ACTION")
        human_verified = bool(case.get("human_verified"))
        human_verified_note = case.get("human_verified_note", "")

        if is_aligned:
            status = "[+] OK"
        elif human_verified:
            status = "[~] MISMATCH (human_verified override)"
            mismatches_count += 1
        else:
            status = "[-] MISMATCH"
            mismatches_count += 1

        print(f"[{idx}/{len(evaluable)}] Case {qid} ({category}): {status} | Action: {action}")
        if not is_aligned:
            print(f"  Reason: {reason}")
            if human_verified:
                print(f"  Human-verified: {human_verified_note}")

        report.append({
            "query_id": qid,
            "category": category,
            "query": query,
            "gold_ids": gold_ids,
            "is_aligned": is_aligned,
            "reason": reason,
            "suggested_action": action,
            "human_verified": human_verified,
            "human_verified_note": human_verified_note,
        })

    # Save report
    with open(AUDIT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    hv_override_count = sum(1 for r in report if not r.get("is_aligned") and r.get("human_verified"))
    raw_pass = len(evaluable) - mismatches_count
    verified_pass = raw_pass + hv_override_count

    print("\n================ AUDIT SUMMARY ================")
    print(f"Total curated cases audited: {len(evaluable)}")
    print(f"Raw judge pass:               {raw_pass}/{len(evaluable)}  ({raw_pass/len(evaluable)*100:.1f}%)")
    print(f"Human-verified overrides:     {hv_override_count} (judge flagged, human review says gold is correct)")
    print(f"Verified pass rate:           {verified_pass}/{len(evaluable)}  ({verified_pass/len(evaluable)*100:.1f}%)")
    print(f"Genuine mismatch/gap:         {mismatches_count - hv_override_count}")
    print(f"Audit report saved to: {AUDIT_REPORT_PATH}")
    print("===============================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
