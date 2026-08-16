from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import logging
import json

from backend.schemas.pydantic_models import QueryRequest
from backend.core.vector_store import VectorDBManager
from backend.core.llm_service import LLMService, build_search_query
from backend.core.reranker import ReRanker


router = APIRouter(prefix = '/api/v1',tags = ['RAG'])

# No DocumentProcessor or HybridRetriever here.
# Ingestion is an offline batch job (backend/scripts/seed_corpus.py), and sparse
# search now lives inside Qdrant as a native sparse vector -- so there is no
# in-process BM25 index to build at startup. The old version pulled the whole
# corpus into RAM on every boot to do that.
vector_db = VectorDBManager()
llm_service = LLMService()
reranker = ReRanker()

logger = logging.getLogger(__name__)


@router.get('/topics')
async def list_topics():
    """Returns the distinct tags present in the indexed corpus."""
    chunks = vector_db.get_all_chunks()
    tags = set()
    for chunk in chunks:
        for tag in (chunk.get("metadata", {}).get("tags") or []):
            tags.add(tag)
    return {"topics": sorted(tags)}

@router.post('/query')
async def query_rag(payload: QueryRequest):
    logger.info(f"Received query request: '{payload.query}' (top_k={payload.top_k}, source_file={payload.source_file}, history_len={len(payload.chat_history or [])})")
    
    if not payload.query.strip():
        logger.warning("Empty query received")
        raise HTTPException(status_code = 400, detail = 'Query cannot be empty.')
    
    # Step 1: Rewrite Query (if chat history exists)
    rewritten_query = llm_service.rewrite_query(payload.query, payload.chat_history)

    # Step 2: Build the SEARCH query. Falls back to concatenating the last two
    # conversation turns when the rewriter did nothing -- a follow-up like
    # "how do I prevent it?" is unsearchable on its own. See build_search_query.
    search_query = build_search_query(payload.query, payload.chat_history, rewritten_query)
    logger.info(f"Search query: '{search_query}'")

    CANDIDATE_K = 15

    # Stage 1: Native Hybrid Search (Dense + Sparse) fused with RRF directly inside Qdrant
    fused_candidates = vector_db.search_hybrid(query=search_query, n_results=CANDIDATE_K, source_file=payload.source_file)
    logger.info(f"Stage 1 (Hybrid Search): Retrieved and fused {len(fused_candidates)} candidate chunks from Qdrant")

    # Stage 2: Cross-Encoder Re-Ranking
    # Scored against what the user actually typed, not the history-padded
    # search string -- the padding exists to find candidates, and would
    # otherwise skew relevance toward the previous turn.
    reranked_results = reranker.rerank(
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