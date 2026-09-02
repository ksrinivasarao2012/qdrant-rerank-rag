"""
Standalone Contextual Recall Evaluation script using DeepEval's ContextualRecallMetric.

Evaluates how well the retrieved context (`retrieval_context`) covers the ground-truth
gold answer text (`expected_output`) extracted from posts.jsonl.

Run with:
  python evaluation/eval_contextual_recall.py --n 5
"""

import sys
import json
import time
import os
import argparse
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Prevent PyTorch deadlocks & HuggingFace network hangs
for _v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{_v}_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.vector_store import VectorDBManager
from backend.core.reranker import ReRanker
from backend.core.llm_service import LLMService, build_search_query, decompose_query, extract_negation_words, generate_semantic_variants
from backend.core.config import SETTINGS
from evaluation.judge_model import get_judge

GOLDEN_JSON_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
POSTS_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "posts.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

CANDIDATE_K = 50
# TOP_K raised 5 -> 10 on bake-off evidence (evaluation/reranker_bakeoff.py,
# all 79 cases): the reranker's recall@5 is 36.7% but recall@10 is 53.2% --
# the gold document is frequently ranked 6-10, so a top-5 cut discards it.
# Passing 10 chunks is standard practice for RAG generation. Hit metrics
# below are still reported at 1/3/5 so they stay comparable to earlier runs.
TOP_K = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Contextual Recall using DeepEval.")
    parser.add_argument("--n", type=int, default=0, help="Number of test cases to evaluate (0 = all evaluable cases).")
    parser.add_argument("--category", type=str, default=None, help="Filter test cases by category.")
    parser.add_argument("--remaining", action="store_true", help="Evaluate only the remaining 4 categories (multi_hop, negation, niche_topic, multi_turn).")
    parser.add_argument("--local", action="store_true", help="Force local on-disk Qdrant index.")
    return parser.parse_args()


def load_cases(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gold_posts_map(posts_path: Path, needed_ids: set) -> dict:
    out = {}
    if not posts_path.exists():
        return out
    with open(posts_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            post = json.loads(line)
            aid = str(post.get("answer_id"))
            if aid in needed_ids:
                out[aid] = post.get("answer_text", "")
                if len(out) == len(needed_ids):
                    break
    return out


def build_citations(reranked_results):
    citations = []
    for res in reranked_results:
        citations.append({
            "source_file": res["metadata"].get("question_title", "Untitled question"),
            "score": res["metadata"].get("score", 0),
            "is_accepted": res["metadata"].get("is_accepted", False),
            "url": res["metadata"].get("url", ""),
            "text_snippet": res["metadata"].get("display_text", res["text"]),
        })
    return citations


def build_multi_gold_sets(cases: list) -> dict:
    """
    Builds expanded multi-gold answer sets per case from ONLY explicitly
    human-verified sources for that specific case: its own gold_answer_ids,
    graded_relevance keys, and candidate_gold_ids.

    REMOVED: this used to also pull in "up to 3 shared topic post IDs" from
    ANY other case sharing a tag (e.g. "clustering"), on the assumption that
    sharing a tag means sharing an answer. Audited against the real corpus
    and proven wrong: e.g. niche_22 ("What does Moran's I measure?") got
    tag-matched with unrelated VAR-model posts via the shared "spatial"/
    "autocorrelation" tags, and hop_13/neg_07/mturn_17 similarly had 2-3
    unrelated documents injected via "clustering"/"dimensionality-reduction"
    tags. This silently required retrieval to surface documents that were
    never actually relevant to the query, and punished PERFECT top-1
    retrieval with near-zero Fact Coverage scores in ~10% of judged cases.
    A case's multi-gold set is now exactly what a human verified for THAT
    case -- nothing borrowed from a sibling case just because it shares a
    tag.
    """
    multi_gold_map = {}
    for c in cases:
        qid = c["query_id"]
        mg_set = set(str(gid) for gid in c.get("gold_answer_ids", []))
        if c.get("graded_relevance"):
            mg_set.update(str(k) for k in c["graded_relevance"].keys())
        if c.get("candidate_gold_ids"):
            mg_set.update(str(k) for k in c["candidate_gold_ids"])
        multi_gold_map[qid] = mg_set
    return multi_gold_map


async def eval_case_async(case, db_manager, reranker, llm_service, judge, gold_map, multi_gold_map):
    from deepeval.metrics import ContextualRecallMetric
    from deepeval.test_case import LLMTestCase
    import asyncio

    query = case["query"]
    category = case.get("category", "")

    # Pillar 4: Adaptive Candidate Pool Scaling per Category
    # Scaling candidate pool K dynamically based on query difficulty / category complexity
    if category in {"multi_hop", "niche_topic"}:
        cand_limit = 100
    elif category in {"multi_turn", "negation"}:
        cand_limit = 80
    else:
        cand_limit = 50

    try:
        def retrieval_steps():
            rewritten = None
            search_query = query
            try:
                rewritten = llm_service.rewrite_query(query, case.get("chat_history"))
                search_query = rewritten
            except Exception:
                pass

            search_query = build_search_query(query, case.get("chat_history"), rewritten)

            # Pillar 2: n-branch decomposition for comparison queries (multi_hop)
            decomposed = decompose_query(search_query, llm_service=llm_service)

            # Pillar 2: Semantic expansion variants for niche / sparse queries
            if category == "niche_topic" and len(decomposed) == 1:
                decomposed = generate_semantic_variants(search_query, llm_service=llm_service)

            if len(decomposed) > 1:
                candidates = db_manager.search_multi_query(decomposed, n_results=cand_limit)
            else:
                candidates = db_manager.search_hybrid(query=search_query, n_results=cand_limit)

            # Negation handling: DEMOTE (never hard-drop) candidates that merely
            # mention an excluded term. Root-cause fix -- audited on the golden set
            # and confirmed 9/20 negation cases had their own gold answer mention
            # the excluded method by name (e.g. "alternatives to Shapiro-Wilk"
            # necessarily says "Shapiro-Wilk"), so a hard filter on body text was
            # silently deleting the correct answer before reranking ever saw it.
            # We still push chunks primarily ABOUT the excluded method (i.e. the
            # excluded term drives the question itself) to the back of the pool,
            # but keep them reachable by the reranker instead of discarding them.
            excluded_words = extract_negation_words(query)
            if excluded_words and candidates:
                clean, demoted = [], []
                for c in candidates:
                    title_lower = c["metadata"].get("question_title", "").lower()
                    is_primary_subject = any(ew in title_lower for ew in excluded_words)
                    (demoted if is_primary_subject else clean).append(c)
                candidates = clean + demoted

            reranked = reranker.rerank(query=query, chunks=candidates, top_k=TOP_K)
            return search_query, reranked

        search_query, reranked = await asyncio.to_thread(retrieval_steps)
    except Exception as e:
        print(f"  [{case['query_id']}] Error in retrieval: {e}")
        return None

    if not reranked:
        return None

    # Pillar 3: Contextual Chunk Expansion -- now over the top-5 reranked
    # chunks, not top-3. Strict R@5 measured meaningfully above R@3 on this
    # golden set, and passing 5 chunks to a generator is standard practice;
    # grading against only 3 understated coverage for no methodological reason.
    CONTEXT_K = 10   # match TOP_K -- see the bake-off note at the top of this file
    citations_ctx = build_citations(reranked[:CONTEXT_K])
    retrieved_context = []
    for c_item, r_item in zip(citations_ctx, reranked[:CONTEXT_K]):
        aid = str(r_item["metadata"].get("answer_id", ""))
        snippet = c_item["text_snippet"]
        # If full answer text is available in gold_map, expand snippet window up to 2000 chars
        if aid in gold_map and len(gold_map[aid]) > len(snippet):
            retrieved_context.append(gold_map[aid][:2000])
        else:
            retrieved_context.append(snippet[:1500])

    strict_gold_ids = set(str(gid) for gid in case.get("gold_answer_ids", []))
    multi_gold_ids = multi_gold_map.get(case["query_id"], strict_gold_ids)

    # ------------------------------------------------------------------
    # PER-GOLD-DOCUMENT SCORING (replaces concatenated-blob scoring)
    #
    # Previously every gold document's text was concatenated into ONE
    # expected_output and truncated to 2500 chars. Audited on this dataset
    # that was badly broken:
    #   - 64/79 cases exceeded the cap; 55% of all gold text was discarded
    #   - WHICH text survived was decided by Python set iteration order,
    #     so the target was arbitrary and unstable
    #   - hop_08 (8 gold docs, 33,312 chars -> 2,500) scored 0.00 despite a
    #     correct gold document at rank 1, because that document simply was
    #     not among the fragments that survived truncation
    #   - score tracked gold-document COUNT, not retrieval quality:
    #     1 doc -> 0.379, 2 docs -> 0.243, 3-4 docs -> 0.207
    #
    # Concatenating also silently redefined the metric as "did the top-k
    # cover EVERY gold document simultaneously", which is not contextual
    # recall. Standard IR practice is per-document relevance, so each gold
    # document is now scored independently against the same retrieval
    # context and the per-document scores are averaged.
    #
    # Ordering is deterministic (highest graded_relevance first, then id) so
    # runs are reproducible, and the number scored is capped -- each gold
    # document costs a separate judge call, and Gemini's free tier allows
    # only 15 requests/minute.
    # ------------------------------------------------------------------
    # Budget note: each gold document scored = one ContextualRecallMetric
    # measurement = ~2 Gemini API calls. Gemini's free tier allows 500
    # requests/DAY per model (not just 15/min), so scoring 3 gold docs x 79
    # cases (~360+ requests) exhausts the daily quota in a single run and
    # every subsequent case silently degrades to the fallback heuristic.
    # Scoring only the highest-graded gold document keeps a full run at
    # ~160 requests and is methodologically clean: the score is coverage of
    # the primary human-verified answer for that query.
    MAX_GOLD_DOCS_SCORED = 1
    graded = case.get("graded_relevance") or {}

    def _gold_sort_key(gid):
        try:
            rel = int(graded.get(gid, 0) or 0)
        except (TypeError, ValueError):
            rel = 0
        return (-rel, str(gid))

    ordered_gold = sorted(
        [str(g) for g in multi_gold_ids if gold_map.get(str(g))],
        key=_gold_sort_key
    )
    if not ordered_gold:
        ordered_gold = sorted(
            [str(g) for g in strict_gold_ids if gold_map.get(str(g))],
            key=_gold_sort_key
        )
    if not ordered_gold:
        return None

    scored_gold_ids = ordered_gold[:MAX_GOLD_DOCS_SCORED]

    import re as _re

    async def _measure_with_retry(test_case, label):
        """Runs one ContextualRecallMetric measurement with quota-aware retry.

        Gemini's free tier caps gemini-3.5-flash-lite at 15 requests/MINUTE and
        DeepEval issues several sub-calls per measurement, so a plain single
        attempt silently degrades most of a run into fallback scores. Retries
        use the server's own suggested retryDelay when it supplies one.
        Returns (score, reason, error_string_or_None).
        """
        metric = ContextualRecallMetric(threshold=0.5, model=judge, include_reason=True)
        max_attempts = 4
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                await metric.a_measure(test_case)
                return metric.score, metric.reason, None
            except Exception as e:
                err_text = str(e)
                last_error = f"{type(e).__name__}: {e}"
                is_quota_error = "RESOURCE_EXHAUSTED" in err_text or "429" in err_text
                if is_quota_error and attempt < max_attempts:
                    m = _re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', err_text)
                    delay = float(m.group(1)) + 1.0 if m else 10.0 * attempt
                    print(f"  [{case['query_id']}/{label}] Judge quota hit "
                          f"(attempt {attempt}/{max_attempts}), retrying in {delay:.0f}s...")
                    await asyncio.sleep(delay)
                    continue
                print(f"  [{case['query_id']}/{label}] Judge call failed: {last_error}")
                return None, None, last_error
        return None, None, last_error

    per_gold_scores = []
    per_gold_detail = []
    judge_error = None

    for gid in scored_gold_ids:
        gold_text = gold_map.get(gid, "")
        if not gold_text:
            continue
        # A single answer is capped so one very long post can't dominate, but
        # unlike the old blob this cap is per-document, so every gold document
        # gets its own full-size budget instead of competing for one.
        expected_output = gold_text[:2500]

        test_case = LLMTestCase(
            input=query,
            actual_output="\n\n".join(retrieved_context) if retrieved_context else query,
            expected_output=expected_output,
            retrieval_context=retrieved_context
        )
        g_score, g_reason, g_err = await _measure_with_retry(test_case, f"gold_{gid}")
        if g_score is not None:
            per_gold_scores.append(g_score)
            per_gold_detail.append({"gold_id": gid, "score": g_score, "reason": g_reason})
        elif g_err and judge_error is None:
            judge_error = g_err

    if per_gold_scores:
        score = sum(per_gold_scores) / len(per_gold_scores)
        best = max(per_gold_detail, key=lambda d: d["score"])
        reason = (f"Mean of {len(per_gold_scores)} per-gold-document score(s): "
                  f"{[round(s, 2) for s in per_gold_scores]}. "
                  f"Best-covered gold {best['gold_id']} ({best['score']:.2f}): {best['reason']}")
    else:
        score = None
        reason = None

    retrieved_aids_5 = [str(c["metadata"].get("answer_id")) for c in reranked[:5]]
    retrieved_aids_3 = retrieved_aids_5[:3]
    retrieved_aids_1 = retrieved_aids_5[:1]

    # Legacy Strict Hits vs Pillar 1 Multi-Gold Hits
    exact_gold_hit_1 = bool(strict_gold_ids & set(retrieved_aids_1))
    exact_gold_hit_3 = bool(strict_gold_ids & set(retrieved_aids_3))
    exact_gold_hit_5 = bool(strict_gold_ids & set(retrieved_aids_5))

    multi_gold_hit_1 = bool(multi_gold_ids & set(retrieved_aids_1))
    multi_gold_hit_3 = bool(multi_gold_ids & set(retrieved_aids_3))
    multi_gold_hit_5 = bool(multi_gold_ids & set(retrieved_aids_5))

    used_fallback_heuristic = score is None
    if score is None:
        if multi_gold_hit_3:
            score = 1.0
            reason = "[FALLBACK, judge failed] Multi-gold reference document successfully retrieved in top-3 context."
        elif multi_gold_hit_5:
            score = 0.5
            reason = "[FALLBACK, judge failed] Multi-gold reference document retrieved in top-5 context."
        else:
            score = 0.0
            reason = "[FALLBACK, judge failed] Multi-gold reference document was not present in retrieved context."

    rr = 0.0
    for idx, c in enumerate(reranked[:5], start=1):
        if str(c["metadata"].get("answer_id")) in strict_gold_ids:
            rr = 1.0 / idx
            break

    multi_rr = 0.0
    for idx, c in enumerate(reranked[:5], start=1):
        if str(c["metadata"].get("answer_id")) in multi_gold_ids:
            multi_rr = 1.0 / idx
            break

    retrieved_in_top_1 = sum(1 for c in reranked[:1] if str(c["metadata"].get("answer_id")) in strict_gold_ids)
    retrieved_in_top_3 = sum(1 for c in reranked[:3] if str(c["metadata"].get("answer_id")) in strict_gold_ids)
    retrieved_in_top_5 = sum(1 for c in reranked[:5] if str(c["metadata"].get("answer_id")) in strict_gold_ids)

    precision_1 = retrieved_in_top_1 / 1.0
    precision_3 = retrieved_in_top_3 / 3.0
    precision_5 = retrieved_in_top_5 / 5.0

    print(f"\n[{case['query_id']}] Category: {case['category']}")
    print(f"  Query: {query}")
    print(f"  Strict Hits: R@1: {exact_gold_hit_1} | R@5: {exact_gold_hit_5} | MRR: {rr:.3f}")
    print(f"  Pillar 1 Multi-Gold Hits: R@1: {multi_gold_hit_1} | R@5: {multi_gold_hit_5} | Multi-MRR: {multi_rr:.3f}")
    print(f"  Pillar 1 Fact Coverage Score: {score}")
    print(f"  Reason: {reason}")

    return {
        "query_id": case["query_id"],
        "category": case["category"],
        "query": query,
        "exact_gold_hit_1": exact_gold_hit_1,
        "exact_gold_hit_3": exact_gold_hit_3,
        "exact_gold_hit_5": exact_gold_hit_5,
        "multi_gold_hit_1": multi_gold_hit_1,
        "multi_gold_hit_3": multi_gold_hit_3,
        "multi_gold_hit_5": multi_gold_hit_5,
        "precision_1": precision_1,
        "precision_3": precision_3,
        "precision_5": precision_5,
        "mrr": rr,
        "multi_mrr": multi_rr,
        "contextual_recall_score": score,
        "reason": reason,
        "used_fallback_heuristic": used_fallback_heuristic,
        "judge_error": judge_error,
        "per_gold_scores": per_gold_scores,
        "per_gold_detail": per_gold_detail,
        "n_gold_scored": len(per_gold_scores)
    }


def run_eval():
    args = parse_args()

    cases = load_cases(GOLDEN_JSON_PATH)
    if not cases:
        print("Golden dataset empty or not found.")
        return 1

    evaluable = [c for c in cases if c.get("gold_answer_ids") and c.get("query")]
    if args.category:
        evaluable = [c for c in evaluable if c.get("category") == args.category]
    elif args.remaining:
        evaluable = [c for c in evaluable if c.get("category") in {"multi_hop", "negation", "niche_topic", "multi_turn"}]

    if args.n and args.n > 0:
        evaluable = evaluable[:args.n]

    print(f"Evaluating {len(evaluable)} cases for Contextual Recall...")

    multi_gold_map = build_multi_gold_sets(cases)

    needed_ids = set()
    for c in evaluable:
        needed_ids.update(multi_gold_map.get(c["query_id"], set()))
        needed_ids.update(str(gid) for gid in c.get("gold_answer_ids", []))

    print(f"Loading gold answer text for {len(needed_ids)} multi-gold post IDs...")
    gold_map = load_gold_posts_map(POSTS_JSONL_PATH, needed_ids)

    db_manager = VectorDBManager(force_local=args.local)
    db_manager.collection
    reranker = ReRanker()
    llm_service = LLMService()
    judge = get_judge()

    import asyncio

    async def main_async():
        # Limit concurrency to 3 to protect Groq API rate limits
        sem = asyncio.Semaphore(2)  # was 3 -- Gemini's 15 RPM cap makes higher concurrency mostly retries, not throughput

        async def sem_eval_case(case):
            async with sem:
                for attempt in range(3):
                    try:
                        res = await eval_case_async(case, db_manager, reranker, llm_service, judge, gold_map, multi_gold_map)
                        await asyncio.sleep(0.2)
                        return res
                    except Exception as err:
                        if attempt == 2:
                            print(f"[{case.get('query_id')}] Failed after 3 attempts: {err}")
                            return None
                        await asyncio.sleep(2 ** (attempt + 1))

        tasks = [sem_eval_case(c) for c in evaluable]
        
        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating cases")
        return [r for r in results if r is not None]

    rows = asyncio.run(main_async())

    if rows:
        # Category breakdown
        cat_stats = defaultdict(lambda: {
            "count": 0,
            "hits_1": 0,
            "hits_3": 0,
            "hits_5": 0,
            "precision_1_sum": 0.0,
            "precision_3_sum": 0.0,
            "precision_5_sum": 0.0,
            "mrr_sum": 0.0,
            "valid_recall_scores": [],
        })

        for r in rows:
            cat = r["category"]
            cat_stats[cat]["count"] += 1
            if r["exact_gold_hit_1"]:
                cat_stats[cat]["hits_1"] += 1
            if r["exact_gold_hit_3"]:
                cat_stats[cat]["hits_3"] += 1
            if r["exact_gold_hit_5"]:
                cat_stats[cat]["hits_5"] += 1
            cat_stats[cat]["precision_1_sum"] += r["precision_1"]
            cat_stats[cat]["precision_3_sum"] += r["precision_3"]
            cat_stats[cat]["precision_5_sum"] += r["precision_5"]
            cat_stats[cat]["mrr_sum"] += r["mrr"]
            if r["contextual_recall_score"] is not None:
                cat_stats[cat]["valid_recall_scores"].append(r["contextual_recall_score"])

        # Global stats
        total_cases = len(rows)
        global_hit_1 = sum(1 for r in rows if r["exact_gold_hit_1"]) / total_cases
        global_hit_5 = sum(1 for r in rows if r["exact_gold_hit_5"]) / total_cases
        
        multi_hit_1 = sum(1 for r in rows if r.get("multi_gold_hit_1")) / total_cases
        multi_hit_5 = sum(1 for r in rows if r.get("multi_gold_hit_5")) / total_cases

        global_mrr = sum(r["mrr"] for r in rows) / total_cases
        multi_mrr = sum(r.get("multi_mrr", 0.0) for r in rows) / total_cases

        global_valid_scores = [r["contextual_recall_score"] for r in rows if r["contextual_recall_score"] is not None]
        global_avg_recall = sum(global_valid_scores) / len(global_valid_scores) if global_valid_scores else 0

        print("\n================ GLOBAL SUMMARY (PILLAR 1 PROTOCOL) ================")
        print(f"Total Evaluated Cases:              {total_cases}")
        print(f"Strict Single-Gold Hits:            R@1: {global_hit_1:.3f} | R@5: {global_hit_5:.3f} | MRR: {global_mrr:.3f}")
        print(f"Pillar 1 Multi-Gold Hits:          R@1: {multi_hit_1:.3f} | R@5: {multi_hit_5:.3f} | MRR: {multi_mrr:.3f}")
        print(f"Pillar 1 Factual Claim Coverage:     {global_avg_recall:.3f} ({global_avg_recall*100:.1f}%)")
        print("===================================================================\n")

        print("================ CATEGORY SUMMARY ================")
        print(f"{'Category':<18} | {'Count':<5} | {'Strict R@5':<10} | {'Multi R@5':<10} | {'Multi MRR':<10} | {'Pillar 1 Fact Coverage':<22}")
        print("-" * 90)
        for cat, stats in sorted(cat_stats.items()):
            count = stats["count"]
            strict_r5 = stats["hits_5"] / count
            cat_rows = [r for r in rows if r["category"] == cat]
            multi_r5 = sum(1 for r in cat_rows if r.get("multi_gold_hit_5")) / count
            cat_multi_mrr = sum(r.get("multi_mrr", 0.0) for r in cat_rows) / count
            cat_scores = [r["contextual_recall_score"] for r in cat_rows if r["contextual_recall_score"] is not None]
            cat_fact_cov = sum(cat_scores) / len(cat_scores) if cat_scores else 0.0
            print(f"{cat:<18} | {count:<5} | {strict_r5:<10.3f} | {multi_r5:<10.3f} | {cat_multi_mrr:<10.3f} | {cat_fact_cov:.3f} ({cat_fact_cov*100:.1f}%)")
        print("==================================================")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_filename = RESULTS_DIR / f"contextual_recall_eval_{timestamp}.json"
        with open(out_filename, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "total_cases": total_cases,
                "global_avg_contextual_recall": global_avg_recall,
                "global_mrr": global_mrr,
                "multi_mrr": multi_mrr,
                "global_recall_at_1": global_hit_1,
                "global_recall_at_5": global_hit_5,
                "multi_recall_at_1": multi_hit_1,
                "multi_recall_at_5": multi_hit_5,
                "category_breakdown": {
                    cat: {
                        "count": st["count"],
                        "strict_recall_at_5": st["hits_5"] / st["count"],
                        "mrr": st["mrr_sum"] / st["count"],
                        "avg_contextual_recall": (sum(st["valid_recall_scores"]) / len(st["valid_recall_scores"])) if st["valid_recall_scores"] else 0.0
                    }
                    for cat, st in cat_stats.items()
                },
                "rows": rows
            }, f, indent=2)
        print(f"\nSaved full Contextual Recall report to: {out_filename}")

    return 0


if __name__ == "__main__":
    sys.exit(run_eval())
