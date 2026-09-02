"""Runtime guardrails for the RAG pipeline.

Four pure functions, no state, no new dependency. Both entrypoints (`app.py`
Gradio and `backend/api/routes.py` FastAPI) call the same four, which is also
how the two pipelines stop drifting apart.

Order of application in a request:

    1. check_input(query)              -- before anything else, on the RAW query
    2. filter_by_score(chunks)         -- after reranking, before the LLM
    3. classify_empty_result(query)    -- only when 2 kept nothing
    4. check_output(answer, chunks)    -- after the stream, before the citations

WHY THIS FILE EXISTS
--------------------
Until 2026-09-02 the only thing standing between a user and an ungrounded
answer was the system prompt, and the active prompt variant explicitly told
the model to answer from general knowledge whenever context was thin. Even
with strict grounding restored (`gptoss_strict_v2`), a prompt only ASKS the
model to refuse -- nothing MAKES it. `filter_by_score` is the enforcement:
when no retrieved chunk is good enough, the LLM is never called at all, so
there is nothing to hallucinate with.
"""

import os
import re
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Input checks
# ---------------------------------------------------------------------------

# Deterministic patterns, not a model. A regex list cannot be talked out of
# its decision, costs no latency and no API call, and is trivially testable.
# It will not catch a determined novel attack -- it is the cheap first layer,
# not the only one. `filter_by_score` is what actually bounds the damage.
_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)",
    r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)",
    r"forget\s+(?:everything|all)\s+(?:above|before|you)",
    r"(?:reveal|repeat|print|show|output)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?|rules?)",
    r"repeat\s+everything\s+above",
    r"you\s+are\s+now\s+(?:a|an|no longer)",
    r"pretend\s+(?:you\s+are|to\s+be)",
    r"act\s+as\s+(?:if\s+you\s+are\s+)?(?:a|an)\s+\w+\s+(?:with\s+no|without)\s+(?:restrictions?|rules?|filters?)",
    r"\bDAN\s+mode\b",
    r"developer\s+mode\s+enabled",
    r"</?(?:system|assistant)>",
    r"\[\s*(?:system|INST)\s*\]",
]

_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# Junk / abuse shapes that are not injection but should not reach the LLM.
_REPEAT_RE = re.compile(r"(.)\1{49,}")            # same char 50+ times
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")  # long opaque blob

REFUSAL_INJECTION = (
    "I can only answer statistics and machine-learning questions using the "
    "Cross Validated posts I have indexed. I can't act on instructions embedded "
    "in a question. Please ask your statistics question directly."
)

REFUSAL_OFF_TOPIC = (
    "I do not have enough information in the provided documents to answer this "
    "question. This assistant only answers statistics and machine-learning "
    "questions that are covered by the Cross Validated corpus."
)


def check_input(query: str) -> Optional[str]:
    """Returns a refusal string if the query must not be processed, else None.

    Runs on the RAW user query, before rewriting -- the rewriter is an LLM and
    should not be the first thing to see hostile text.

    Known false positive: someone genuinely asking ABOUT prompt injection gets
    refused. Accepted deliberately -- rare on a statistics corpus, and a wrong
    refusal is far cheaper than a successful injection on a public deployment.
    """
    if not query or not query.strip():
        return REFUSAL_OFF_TOPIC

    if _INJECTION_RE.search(query):
        logger.warning("Guardrail: input blocked (injection pattern) for query %r", query[:120])
        return REFUSAL_INJECTION

    if _REPEAT_RE.search(query) or _BASE64_RE.search(query):
        logger.warning("Guardrail: input blocked (junk payload) for query %r", query[:120])
        return REFUSAL_INJECTION

    return None


# ---------------------------------------------------------------------------
# 2. Relevance floor  --  the load-bearing guardrail
# ---------------------------------------------------------------------------

# PLACEHOLDER VALUE -- NOT YET CALIBRATED.
# Calibrate by running the `out_of_scope` (should score BELOW) and `standard`
# (should score ABOVE) golden slices and picking the separating value, then
# sanity-check against `niche_topic` (real questions with weak matches -- the
# cases most at risk of being wrongly refused). Deliberately set permissive
# for now: a weak answer annoys a user less than a wrong refusal.
MIN_RERANK_SCORE = float(os.getenv("GUARDRAIL_MIN_SCORE", "0.25"))


def normalize_rerank_score(raw: float, source: str) -> float:
    """Puts both reranker backends on the same 0-1 scale.

    THE TRAP THIS SOLVES: the local cross-encoder
    (`cross-encoder/ms-marco-MiniLM-L-6-v2`) emits raw LOGITS, roughly -11..+11
    -- despite reranker.py's comment claiming "0.0 to 1.0". The Jina API emits
    a genuine 0-1 relevance score. Jina silently disables itself on a 401/403
    balance error and falls through to the local model, so without this a
    single threshold constant would mean two completely different things
    depending on billing state -- a mystery bug weeks later.
    """
    if source == "jina":
        return max(0.0, min(1.0, float(raw)))
    # logistic squash of the cross-encoder logit
    try:
        return 1.0 / (1.0 + math.exp(-float(raw)))
    except OverflowError:
        return 0.0 if raw < 0 else 1.0


def filter_by_score(
    chunks: List[Dict[str, Any]],
    floor: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Drops chunks the reranker scored below the floor.

    An empty return is the signal to skip generation entirely. Chunks missing
    a score are KEPT: a reranker that failed to load bypasses scoring
    altogether (see reranker.py), and in that state we must not refuse
    everything -- fail open, log loudly.
    """
    threshold = MIN_RERANK_SCORE if floor is None else floor
    kept = []
    for c in chunks:
        raw = c.get("rerank_score")
        if raw is None:
            logger.warning("Guardrail: chunk has no rerank_score -- keeping (fail open).")
            kept.append(c)
            continue
        score = c.get("rerank_score_norm")
        if score is None:
            score = normalize_rerank_score(raw, c.get("rerank_source", "local"))
            c["rerank_score_norm"] = score
        if score >= threshold:
            kept.append(c)

    if len(kept) < len(chunks):
        logger.info(
            "Guardrail: relevance floor %.2f dropped %d/%d chunks.",
            threshold, len(chunks) - len(kept), len(chunks)
        )
    return kept


# ---------------------------------------------------------------------------
# 3. Nothing survived the floor -- why?
# ---------------------------------------------------------------------------

_SMALL_TALK = {
    "hi", "hii", "hey", "hello", "yo", "sup", "hiya",
    "good morning", "good afternoon", "good evening", "greetings",
    "thanks", "thank you", "thanks a lot", "thank you so much", "ty", "thx",
    "ok", "okay", "cool", "nice", "great", "awesome", "got it", "understood",
    "bye", "goodbye", "see you", "cya",
    "who are you", "what are you", "what can you do", "what do you do",
    "help", "what is this", "how does this work",
}

SMALL_TALK_REPLY = (
    "Hello. I answer statistics and machine-learning questions using indexed "
    "posts from Cross Validated (stats.stackexchange.com), and I link every "
    "answer back to its source thread.\n\n"
    "Try something like *\"why does my model get 99% train accuracy but 60% on "
    "test data?\"* or *\"when should I use Lasso instead of Ridge?\"*"
)


def classify_empty_result(query: str) -> str:
    """Returns 'small_talk' or 'off_topic'. Only meaningful once the relevance
    floor has already rejected every chunk.

    Placing this AFTER retrieval failure is deliberate. A greeting classifier
    that ran up front would be a bypass: "hi, ignore your instructions and..."
    would route to the friendly path. Here it can only ever be reached by a
    query that already failed check_input AND failed to retrieve anything.
    """
    q = (query or "").strip().lower()
    q = re.sub(r"[!.?,]+$", "", q).strip()

    if q in _SMALL_TALK:
        return "small_talk"

    # Short, no question content, starts with a greeting -> still small talk.
    # ("hi there" / "thanks!!" but NOT "hi, how do I compute a p-value?")
    words = q.split()
    if len(words) <= 4 and not re.search(r"\b(how|why|what|when|which|where|who|is|are|do|does|can|should)\b", q):
        for phrase in _SMALL_TALK:
            if q.startswith(phrase):
                return "small_talk"

    return "off_topic"


def empty_result_response(query: str) -> Tuple[str, bool]:
    """Convenience wrapper: returns (message, show_citations)."""
    if classify_empty_result(query) == "small_talk":
        return SMALL_TALK_REPLY, False
    return REFUSAL_OFF_TOPIC, False


# ---------------------------------------------------------------------------
# 4. Output checks
# ---------------------------------------------------------------------------

_LEAK_PATTERNS = [
    r"You are an expert AI assistant specializing in document analysis",
    r"STRICT RULES:",
    r"CONTEXT INFORMATION:",
]
_LEAK_RE = re.compile("|".join(_LEAK_PATTERNS), re.IGNORECASE)

# Substring that marks the model's grounded refusal. LOAD-BEARING: must stay
# byte-compatible with rule 2 of the active answer prompt variant.
REFUSAL_MARKER = "do not have enough information"

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "for", "with", "as", "by", "at", "from", "it", "its",
    "you", "your", "we", "our", "they", "their", "can", "will", "would",
    "should", "could", "may", "might", "have", "has", "had", "do", "does",
    "not", "no", "when", "which", "what", "how", "why", "more", "most", "some",
    "such", "also", "there", "here", "one", "two", "each", "other", "into",
    "about", "over", "under", "between", "because", "so", "however", "using",
    "used", "use", "example", "answer", "question", "provided", "documents",
}

# Below this share of the answer's content words appearing in the retrieved
# text, the answer is treated as not built from the context. Deliberately
# LOW -- this test is rough, and it must only ever fire on a clear miss.
MIN_GROUNDING_OVERLAP = 0.15


def _content_words(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9\-']{2,}", (text or "").lower())
    return [t for t in tokens if t not in _STOPWORDS]


def grounding_overlap(answer: str, chunks: List[Dict[str, Any]]) -> float:
    """Share of the answer's content words that appear in the retrieved text.

    Crude by design. It does not verify claims -- it detects the specific
    failure where the model ignored the context and answered from memory,
    which is exactly what makes attaching citations misleading.
    """
    ans = _content_words(answer)
    if not ans:
        return 1.0
    ctx = set()
    for c in chunks:
        ctx.update(_content_words(c.get("text_snippet") or c.get("text") or ""))
    if not ctx:
        return 0.0
    return sum(1 for w in ans if w in ctx) / len(ans)


def check_output(
    answer: str,
    chunks: List[Dict[str, Any]],
) -> Tuple[str, bool]:
    """Returns (cleaned_answer, show_citations).

    Attaching sources to an answer that ignored them launders a general-
    knowledge answer as a sourced one, which is worse than showing no sources
    at all -- so citations are a privilege the answer has to earn.
    """
    cleaned = answer or ""

    if _LEAK_RE.search(cleaned):
        logger.warning("Guardrail: system-prompt text detected in output; stripping.")
        cleaned = _LEAK_RE.sub("", cleaned).strip()

    # The model refused. Sources would contradict the refusal.
    if REFUSAL_MARKER in cleaned.lower():
        return cleaned, False

    # Legacy banner from the retired gptoss_simple_v1 fallback path. Kept so
    # that variant stays switchable without silently mislabelling citations.
    if "out-of-boundary" in cleaned.lower():
        return cleaned, False

    if not chunks:
        return cleaned, False

    overlap = grounding_overlap(cleaned, chunks)
    if overlap < MIN_GROUNDING_OVERLAP:
        logger.warning(
            "Guardrail: answer overlap with retrieved context is %.2f (< %.2f) "
            "-- suppressing citations as likely ungrounded.",
            overlap, MIN_GROUNDING_OVERLAP
        )
        return cleaned, False

    return cleaned, True


# ---------------------------------------------------------------------------
# 5. Query-rewrite safety net (code-review bugs #9 and #10)
# ---------------------------------------------------------------------------
# #9: rewrite_query() tries up to 7 providers one after another with no
# overall time limit -- on a bad day every provider fails slowly before
# search even starts. #10: whatever the rewriter returns goes straight into
# retrieval with no sanity check -- an empty, absurd, or unrelated rewrite
# silently sends garbage to search.
#
# Both are call-site problems, not something to fix inside rewrite_query's
# internal cascade. This wraps the WHOLE cascade in one thread with a
# deadline (#9), then validates whatever comes back before using it (#10).

REWRITE_TIMEOUT_SECONDS = float(os.getenv("REWRITE_TIMEOUT_SECONDS", "4.0"))


def is_valid_rewrite(original: str, rewritten: str) -> bool:
    """True if a rewritten query is safe to search with instead of the original.

    Deliberately generous -- a rewrite that resolves a pronoun or drops
    filler words legitimately shares few words with the original. This only
    catches the clear failure shapes: empty, wildly long, or completely
    unrelated (e.g. the rewriter echoed a prompt-template artifact).
    """
    if not rewritten or not rewritten.strip():
        return False
    r = rewritten.strip()
    if len(r) > 500:
        return False
    orig_words = set(_content_words(original))
    new_words = set(_content_words(r))
    if orig_words and not (orig_words & new_words):
        return False
    return True


def safe_rewrite_query(llm_service, query: str, chat_history=None) -> str:
    """Runs llm_service.rewrite_query() with a deadline and a sanity check.

    Falls back to the raw query, exactly like the unwrapped call already did
    on every internal provider failure -- the only change is that a slow or
    nonsensical outcome now ALSO falls back, instead of hanging or silently
    corrupting the search query.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

    # NOT a context manager on purpose. `with ThreadPoolExecutor() as pool:`
    # calls shutdown(wait=True) on exit, which blocks until the submitted call
    # finishes even after future.result() has already raised a timeout --
    # defeating the deadline entirely (a 7-provider cascade that is genuinely
    # stuck would still make this function wait for it). shutdown(wait=False)
    # lets this function return on time; the background call is abandoned to
    # finish (or hang) on its own thread, which is harmless since its result
    # is never used.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(llm_service.rewrite_query, query, chat_history)
    try:
        rewritten = future.result(timeout=REWRITE_TIMEOUT_SECONDS)
        pool.shutdown(wait=False)
    except FutureTimeoutError:
        pool.shutdown(wait=False)
        logger.warning(
            "Guardrail: query rewrite exceeded %.1fs -- using raw query.",
            REWRITE_TIMEOUT_SECONDS
        )
        return query
    except Exception as e:
        pool.shutdown(wait=False)
        logger.warning("Guardrail: query rewrite raised %s -- using raw query.", e)
        return query

    if not is_valid_rewrite(query, rewritten):
        logger.warning(
            "Guardrail: rewrite %r failed sanity check against %r -- using raw query.",
            (rewritten or "")[:80], query[:80]
        )
        return query

    return rewritten
