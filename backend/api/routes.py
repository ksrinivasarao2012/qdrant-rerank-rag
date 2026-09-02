import os
import secrets
import time
from collections import defaultdict
from threading import Lock
from fastapi import APIRouter, HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
import logging
import json

from backend.schemas.pydantic_models import QueryRequest
from backend.core.vector_store import VectorDBManager
from backend.core.llm_service import LLMService, build_search_query
from backend.core.reranker import ReRanker


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
# Rate Limiter: Thread-safe Sliding Window Rate Limiter
# ---------------------------------------------------------------------------
class SlidingWindowRateLimiter:
    """Per-key or per-IP sliding window rate limiter."""
    def __init__(self, requests_per_minute: int = 20):
        self.rpm = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = Lock()

    def check(self, key: str):
        now = time.time()
        window_start = now - 60.0
        with self.lock:
            timestamps = [t for t in self.requests[key] if t > window_start]
            if len(timestamps) >= self.rpm:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: maximum {self.rpm} requests per minute."
                )
            timestamps.append(now)
            self.requests[key] = timestamps

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
    limiter.check(rate_key)

    logger.info(f"Received query request: '{payload.query}' (top_k={payload.top_k}, source_file={payload.source_file}, history_len={len(payload.chat_history or [])})")
    
    if not payload.query.strip():
        logger.warning("Empty query received")
        raise HTTPException(status_code = 400, detail = 'Query cannot be empty.')
    
    # Step 1: Rewrite Query (offloaded to threadpool to prevent blocking event loop)
    rewritten_query = await run_in_threadpool(
        llm_service.rewrite_query, payload.query, payload.chat_history
    )

    # Step 2: Build the SEARCH query. Falls back to concatenating the last two
    # conversation turns when the rewriter did nothing -- a follow-up like
    # "how do I prevent it?" is unsearchable on its own. See build_search_query.
    search_query = build_search_query(payload.query, payload.chat_history, rewritten_query)
    logger.info(f"Search query: '{search_query}'")

    CANDIDATE_K = 15

    # Stage 1: Native Hybrid Search (Dense + Sparse) fused with RRF directly inside Qdrant
    # Offloaded to threadpool so network/disk latency doesn't block the event loop
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
            async for token in llm_service.stream_answer(
                query = payload.query,
                citations = citations_list,
                chat_history = payload.chat_history
            ): 
                token_count += 1
                yield json.dumps({
                    "type": "token",
                    "data": token
                }) + '\n'
            logger.info(f"Finished NDJSON stream successfully ({token_count} token chunks emitted).")

        except Exception as e:
            logger.error(f"Streaming error occurred during generation: {e}")
            yield json.dumps({"type": "token", "data": "\n[Stream interrupted due to an error.]"}) + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")