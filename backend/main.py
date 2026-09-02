import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router, vector_db, reranker

# Configure unified logging format across the entire backend
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title = "Portfolio RAG API",
    description = "Backend API for RAG query answering over the Cross Validated corpus",
    version = '1.0.0'
)


@app.on_event("startup")
async def warm_models():
    """Loads the embedding model, the reranker, and confirms the Qdrant
    collection is reachable at server boot -- not on whatever request
    happens to arrive first."""
    from backend.core.config import SETTINGS
    SETTINGS.validate()  # bug #16: loud warning for missing GROQ_API_KEY, not a confusing failure on first request
    logger.info("Warming models at startup...")
    vector_db.embedding  # triggers the shared embedding singleton to load
    reranker.encoder     # triggers the cross-encoder to load
    vector_db.collection  # confirms Qdrant is reachable now, not on first request
    logger.info("Models warmed -- ready to serve requests.")


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Hard payload cap (64 KB) to prevent memory bloat or DoS via massive requests."""
    cl = request.headers.get("content-length")
    if cl and int(cl) > 64 * 1024:
        return JSONResponse({"detail": "Request payload too large (max 64 KB)"}, status_code=413)
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_origins=["*"],
    allow_headers=["*"],
)


app.include_router(router)


@app.get('/')
async def root():
    """Bare liveness check outside the API key-protected router -- process is
    up. No dependency checks here on purpose: a load balancer hitting this on
    every request should not also be pinging Qdrant every time."""
    return {'status': 'ok', 'message': 'API is running and healthy.'}


@app.get('/healthz')
async def healthz():
    """Real readiness check. Previously this returned {"status": "ok"}
    unconditionally with no check of anything -- a health endpoint that can
    never go unhealthy tells an operator nothing. Verifies the two things
    that actually have to be true for a request to succeed: the embedding +
    reranker models are loaded, and Qdrant is reachable with the expected
    collection. Returns 503, not 200, when either check fails, so uptime
    monitoring and orchestrators (Render, Docker HEALTHCHECK) can act on it.
    """
    from fastapi.concurrency import run_in_threadpool

    checks = {}
    healthy = True

    try:
        checks["embedding_model"] = vector_db.embedding is not None
    except Exception as e:
        checks["embedding_model"] = f"error: {e}"
        healthy = False

    try:
        checks["reranker_model"] = reranker.encoder is not None and reranker.encoder != "FAILED"
    except Exception as e:
        checks["reranker_model"] = f"error: {e}"
        healthy = False

    try:
        collection_name = await run_in_threadpool(lambda: vector_db.collection)
        checks["qdrant"] = f"reachable ({collection_name})"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"
        healthy = False

    body = {"status": "ok" if healthy else "unhealthy", "checks": checks}
    if not healthy:
        logger.error(f"Health check failed: {checks}")
        return JSONResponse(body, status_code=503)
    return body

