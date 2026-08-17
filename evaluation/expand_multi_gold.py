"""
Multi-Gold Expansion: discovers thread-sibling answers for curated golden
dataset categories and judges their relevance using the local GGUF model.

Design decisions (locked via /grill-me session):
  - Target categories: multi_hop, niche_topic, paraphrase_group, multi_turn
  - Discovery: thread-sibling (same question_id as existing gold)
  - Score threshold: >= 3
  - Judge: local Qwen2.5-7B GGUF (same as DeepEval metrics)
  - Policy: additive only (never remove existing golds)
  - Audit: full judgment log saved to evaluation/multi_gold_audit.json

Run:
  python evaluation/expand_multi_gold.py              # full run
  python evaluation/expand_multi_gold.py --dry-run    # preview without saving
  python evaluation/expand_multi_gold.py --n 5        # limit to first N target cases
"""

import sys
import os
import json
import shutil
import argparse
import time
from pathlib import Path

# Prevent PyTorch deadlocks on Windows
for _v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{_v}_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_JSON_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
POSTS_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "posts.jsonl"
AUDIT_LOG_PATH = PROJECT_ROOT / "evaluation" / "multi_gold_audit.json"

TARGET_CATEGORIES = {"multi_hop", "niche_topic", "paraphrase_group", "multi_turn"}
MIN_SCORE = 3
MAX_ANSWER_TEXT_FOR_JUDGE = 2000  # truncate long answers to fit context window


def parse_args():
    parser = argparse.ArgumentParser(description="Expand golden dataset with multi-gold labels.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview judgments without saving changes.")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N target cases (for testing).")
    return parser.parse_args()


def load_golden_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_posts_index(posts_path: Path):
    """Builds two indexes from posts.jsonl:
    - answer_id_to_post: {str(answer_id): post_dict}
    - question_id_to_answers: {str(question_id): [post_dict, ...]}
    """
    answer_id_to_post = {}
    question_id_to_answers = {}

    print(f"Building posts index from {posts_path.name}...", flush=True)
    with open(posts_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            post = json.loads(line)
            aid = str(post.get("answer_id"))
            qid = str(post.get("question_id"))

            answer_id_to_post[aid] = post

            if qid not in question_id_to_answers:
                question_id_to_answers[qid] = []
            question_id_to_answers[qid].append(post)

    print(f"  Indexed {len(answer_id_to_post)} answers across "
          f"{len(question_id_to_answers)} questions.", flush=True)
    return answer_id_to_post, question_id_to_answers


def load_judge():
    """Loads the local GGUF judge model (same as judge_model.py but without
    the DeepEval base class dependency -- we only need raw chat completion)."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from llama_cpp import Llama

    default_path = PROJECT_ROOT / "data" / "models" / "qwen2.5-7b-instruct-q4_k_m.gguf"
    model_path = Path(os.getenv("LOCAL_JUDGE_MODEL_PATH", str(default_path)))

    if not model_path.exists():
        print(f"ERROR: Judge model not found at {model_path.resolve()}")
        sys.exit(1)

    n_threads = os.cpu_count() or 4
    print(f"Loading local judge: {model_path.name} (n_threads={n_threads})...", flush=True)
    model = Llama(
        model_path=str(model_path),
        n_ctx=4096,
        n_threads=n_threads,
        verbose=False,
    )
    return model


def judge_relevance(model, query: str, answer_text: str) -> dict:
    """Asks the GGUF judge whether answer_text is a valid response to query.
    Returns {"verdict": "YES"|"NO", "reason": str}."""

    truncated = answer_text[:MAX_ANSWER_TEXT_FOR_JUDGE]
    prompt = (
        f"You are evaluating whether a candidate answer is relevant to a user query.\n\n"
        f"USER QUERY: {query}\n\n"
        f"CANDIDATE ANSWER:\n{truncated}\n\n"
        f"Is this answer a valid, on-topic response to the query above?\n"
        f"Reply with exactly one word on the first line: YES or NO.\n"
        f"Then provide a one-sentence reason on the second line."
    )

    try:
        response = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0,
        )
        text = response["choices"][0]["message"]["content"].strip()
        lines = text.split("\n", 1)
        verdict = lines[0].strip().upper()
        reason = lines[1].strip() if len(lines) > 1 else ""

        # Normalize: accept variations like "YES." or "Yes, because..."
        if verdict.startswith("YES"):
            verdict = "YES"
        elif verdict.startswith("NO"):
            verdict = "NO"
        else:
            verdict = "UNCLEAR"
            reason = text

        return {"verdict": verdict, "reason": reason}
    except Exception as e:
        return {"verdict": "ERROR", "reason": str(e)}


def get_query_for_case(case: dict) -> str:
    """Extracts the primary query string from a case, handling paraphrase_group."""
    if case["category"] == "paraphrase_group":
        variants = case.get("variants", [])
        return variants[0] if variants else case.get("query", "")
    return case.get("query", "")


def expand_golden_dataset(args):
    cases = load_golden_dataset(GOLDEN_JSON_PATH)
    answer_id_to_post, question_id_to_answers = build_posts_index(POSTS_JSONL_PATH)

    # Filter to target categories
    target_cases = [c for c in cases if c.get("category") in TARGET_CATEGORIES
                    and c.get("gold_answer_ids")]
    if args.n:
        target_cases = target_cases[:args.n]

    print(f"\nTarget cases: {len(target_cases)} across categories: "
          f"{', '.join(sorted(TARGET_CATEGORIES))}", flush=True)

    judge_model = load_judge()
    audit_log = []
    total_added = 0
    total_judged = 0

    for i, case in enumerate(target_cases, start=1):
        query = get_query_for_case(case)
        qid = case["query_id"]
        existing_golds = set(case["gold_answer_ids"])

        # Find question_ids for existing golds
        question_ids = set()
        for gid in existing_golds:
            post = answer_id_to_post.get(gid)
            if post:
                question_ids.add(str(post["question_id"]))

        if not question_ids:
            print(f"  [{i}/{len(target_cases)}] {qid}: No question_id found for golds, skipping.", flush=True)
            continue

        # Collect all sibling candidates
        candidates = []
        for q_id in question_ids:
            for sibling in question_id_to_answers.get(q_id, []):
                sid = str(sibling["answer_id"])
                if sid not in existing_golds and sibling.get("score", 0) >= MIN_SCORE:
                    candidates.append(sibling)

        if not candidates:
            print(f"  [{i}/{len(target_cases)}] {qid}: No eligible siblings (score>={MIN_SCORE}).", flush=True)
            continue

        print(f"  [{i}/{len(target_cases)}] {qid} ({case['category']}): "
              f"{len(candidates)} sibling candidates to judge...", flush=True)

        added_this_case = 0
        for cand in candidates:
            cid = str(cand["answer_id"])
            answer_text = cand.get("answer_text", "")

            result = judge_relevance(judge_model, query, answer_text)
            total_judged += 1

            entry = {
                "query_id": qid,
                "category": case["category"],
                "query": query,
                "candidate_answer_id": cid,
                "candidate_score": cand.get("score", 0),
                "candidate_is_accepted": cand.get("is_accepted", False),
                "candidate_text_preview": answer_text[:300],
                "verdict": result["verdict"],
                "reason": result["reason"],
            }
            audit_log.append(entry)

            if result["verdict"] == "YES":
                case["gold_answer_ids"].append(cid)
                if "graded_relevance" not in case:
                    case["graded_relevance"] = {}
                case["graded_relevance"][cid] = 2  # judge-approved (vs 3 for human-curated)
                added_this_case += 1
                total_added += 1
                print(f"    [+] Added {cid} (score={cand.get('score',0)}) -- {result['reason'][:80]}", flush=True)
            else:
                print(f"    [-] Rejected {cid} (score={cand.get('score',0)}) -- {result['reason'][:80]}", flush=True)

        if added_this_case:
            print(f"    => {qid}: {added_this_case} new golds added "
                  f"(total now: {len(case['gold_answer_ids'])})", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"  Cases processed: {len(target_cases)}", flush=True)
    print(f"  Siblings judged: {total_judged}", flush=True)
    print(f"  New golds added: {total_added}", flush=True)
    print(f"  Verdicts: YES={sum(1 for a in audit_log if a['verdict']=='YES')}, "
          f"NO={sum(1 for a in audit_log if a['verdict']=='NO')}, "
          f"UNCLEAR/ERROR={sum(1 for a in audit_log if a['verdict'] not in ('YES','NO'))}", flush=True)
    print(f"{'='*60}", flush=True)

    # Save audit log (always, even in dry-run)
    with open(AUDIT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2, ensure_ascii=False)
    print(f"\nAudit log saved to {AUDIT_LOG_PATH.name} ({len(audit_log)} entries).", flush=True)

    if args.dry_run:
        print("\n[!] DRY RUN -- golden_dataset.json was NOT modified.", flush=True)
    else:
        # Backup original
        backup_path = GOLDEN_JSON_PATH.with_suffix(".pre_multi_gold.json")
        shutil.copy2(GOLDEN_JSON_PATH, backup_path)
        print(f"Backup saved to {backup_path.name}.", flush=True)

        # Save updated dataset
        with open(GOLDEN_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        print(f"Updated golden_dataset.json saved ({total_added} new gold IDs added).", flush=True)

    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(expand_golden_dataset(args))
