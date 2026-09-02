import os
import secrets
from fastapi import APIRouter, HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
import logging
import json

from backend.schemas.pydantic_models import QueryRequest
from backend.core.vector_store import VectorDBManager
from backend.core.llm_service import LLMService, build_search_query, decompose_query
from backend.core.reranker import ReRanker
from backend.core import guardrails
from backend.core.rate_limiter import SlidingWindowRateLimiter, RateLimitExceeded


# ---------------------------------------------------------------------------
# Auth & Security: Constant-time API Key Verification
# ---------------------------------------------------------------------------
_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: str = Security(_key_header)) -> str:
    """
    Validates X-API-Key header against the API_KEYS environment variable using
    constant-time comparison (secrets.compare_digest) to prevent timing attacks.
    If API_KEYS is not configured in environment, auth is bypassed for dev/demo setups.
    """
    api_keys_str = os.getenv("API_KEYS", "").strip()
    if not api_keys_str:
        return "bypassed"
    valid_keys = {k.strip() for k in api_keys_str.split(",") if k.strip()}
    if not key or not any(secrets.compare_digest(key, valid_key) for valid_key in valid_keys):
        raise HTTPException(status_code=401, detail="Invalid or missing API key in X-API-Key header")
    return key


# ---------------------------------------------------------------------------
# Rate Limiter -- implementation lives in backend/core/rate_limiter.py so the
# Gradio app (app.py) can use the identical logic instead of having none.
# ---------------------------------------------------------------------------
limiter = SlidingWindowRateLimiter(requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "20")))

router = APIRouter(prefix='/api/v1', tags=['RAG'], dependencies=[Depends(require_api_key)])

vector_db = VectorDBManager()
llm_service = LLMService()
reranker = ReRanker()

logger = logging.getLogger(__name__)


@router.get('/topics')
async def list_topics():
    """Returns the distinct tags present in the indexed corpus (O(1) precomputed read)."""
    topics = await run_in_threadpool(vector_db.get_topics)
    return {"topics": topics}

@router.post('/query')
async def query_rag(payload: QueryRequest, request: Request, api_key: str = Depends(require_api_key)):
    # Rate limit check by API Key (or client IP fallback)
    rate_key = api_key if api_key != "bypassed" else (request.client.host if request.client else "global")
    try:
        limiter.check(rate_key)
    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))

    logger.info(f"Received query request: '{payload.query}' (top_k={payload.top_k}, source_file={payload.source_file}, history_len={len(payload.chat_history or [])})")
    
    if not payload.query.strip():
        logger.warning("Empty query received")
        raise HTTPException(status_code = 400, detail = 'Query cannot be empty.')

    # Guardrail 1: check the RAW query before the LLM rewriter sees it.
    # Returned as a normal NDJSON stream (not an HTTP error) so clients handle
    # a refusal exactly like any other answer.
    blocked = guardrails.check_input(payload.query)
    if blocked:
        async def blocked_generator():
            yield json.dumps({"type": "citations", "data": []}) + "\n"
            yield json.dumps({"type": "token", "data": blocked}) + "\n"
        return StreamingResponse(blocked_generator(), media_type="application/x-ndjson")
    
    # Step 1: Rewrite Query (offloaded to threadpool to prevent blocking event loop).
    # guardrails.safe_rewrite_query wraps the up-to-7-provider cascade in a
    # single deadline (bug #9: no rewrite step should be able to stall the
    # request indefinitely) and rejects a nonsensical result before it can
    # reach retrieval (bug #10: nothing previously checked what came back).
    rewritten_query = await run_in_threadpool(
        guardrails.safe_rewrite_query, llm_service, payload.query, payload.chat_history
    )

    # Step 2: Build the SEARCH query. Falls back to concatenating the last two
    # conversation turns when the rewriter did nothing -- a follow-up like
    # "how do I prevent it?" is unsearchable on its own. See build_search_query.
    search_query = build_search_query(payload.query, payload.chat_history, rewritten_query)
    logger.info(f"Search query: '{search_query}'")

    # CANDIDATE_K and multi-hop decomposition matched to app.py (the tuned,
    # interactively-evaluated path) so this endpoint and the Gradio UI run the
    # same retrieval behaviour instead of two pipelines that quietly diverged.
    CANDIDATE_K = 10
    decomposed_queries = await run_in_threadpool(decompose_query, search_query, llm_service)

    # Stage 1: Native Hybrid Search (Dense + Sparse) fused with RRF directly inside Qdrant.
    # Offloaded to threadpool so network/disk latency doesn't block the event loop.
    if len(decomposed_queries) > 1:
        fused_candidates = await run_in_threadpool(
            vector_db.search_multi_query, queries=decomposed_queries, n_results=CANDIDATE_K, source_file=payload.source_file
        )
    else:
        fused_candidates = await run_in_threadpool(
            vector_db.search_hybrid, query=search_query, n_results=CANDIDATE_K, source_file=payload.source_file
        )
    logger.info(f"Stage 1 (Hybrid Search): Retrieved and fused {len(fused_candidates)} candidate chunks from Qdrant")

    # Stage 2: Cross-Encoder Re-Ranking
    # CPU-heavy inference offloaded to threadpool so it doesn't starve concurrent requests
    reranked_results = await run_in_threadpool(
        reranker.rerank,
        query = payload.query,
        chunks = fused_candidates,
        top_k = payload.top_k
    )
    logger.info(f"Stage 2 (Cross-Encoder): Re-ranked candidates down to top {len(reranked_results)} results")

    # Guardrail 2: relevance floor -- skip generation entirely when nothing
    # retrieved is good enough. The system prompt asks the model to refuse;
    # this is what makes it.
    reranked_results = guardrails.filter_by_score(reranked_results)

    # Guardrail 3: route the empty case (small talk vs genuinely off-topic).
    if not reranked_results:
        message, _ = guardrails.empty_result_response(payload.query)
        logger.info("Guardrail: no chunk cleared the relevance floor -- returning canned response without calling the LLM.")
        async def empty_generator():
            yield json.dumps({"type": "citations", "data": []}) + "\n"
            yield json.dumps({"type": "token", "data": message}) + "\n"
        return StreamingResponse(empty_generator(), media_type="application/x-ndjson")
    
    # Build citations list from re-ranked results
    # Stack Exchange answers have no page numbers, so a citation is the question
    # title plus vote score, accepted flag and a link to the original thread.
    citations_list = []
    for res in reranked_results:
        citations_list.append({
            'source_file': res['metadata'].get('question_title', 'Untitled question'),
            'score': res['metadata'].get('score', 0),
            'is_accepted': res['metadata'].get('is_accepted', False),
            'url': res['metadata'].get('url', ''),
            # display_text (plain chunk, no title/overlap prefix) if present;
            # falls back to the full embedded text for chunks indexed before
            # this field existed.
            'text_snippet': res['metadata'].get('display_text', res["text"])
        })
    logger.info(f"Prepared {len(citations_list)} citations for LLM streaming response")
    
    # Asynchronous NDJSON Stream Generator
    async def response_generator():
        logger.info("Starting NDJSON stream response...")
        # Yield citations first as a single JSON line
        yield json.dumps({
            "type": "citations",
            "data": citations_list
        }) + '\n'

        # Stream the LLM tokens as they are generated
        try:
            token_count = 0
            full_text = ""
            async for token in llm_service.stream_answer(
                query = payload.query,
                citations = citations_list,
                chat_history = payload.chat_history
            ): 
                token_count += 1
                full_text += token
                yield json.dumps({
                    "type": "token",
                    "data": token
                }) + '\n'

            # Guardrail 4: the citations were already sent (they lead the
            # stream, by design, so the client can render sources immediately).
            # Tokens cannot be recalled, so instead emit an explicit trailer
            # telling the client whether those citations should be displayed.
            # A client that ignores the trailer behaves exactly as before.
            _, show_citations = guardrails.check_output(full_text, citations_list)
            yield json.dumps({"type": "citations_valid", "data": show_citations}) + '\n'
            logger.info(
                f"Finished NDJSON stream successfully ({token_count} token chunks emitted, "
                f"citations_valid={show_citations})."
            )

        except Exception as e:
            logger.error(f"Streaming error occurred during generation: {e}")
            yield json.dumps({"type": "token", "data": "\n[Stream interrupted due to an error.]"}) + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")