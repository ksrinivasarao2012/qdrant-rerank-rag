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


@app.get('/healthz')
@app.get('/')
async def root():
    """Health check endpoint outside the API key-protected router."""
    return {'status': 'ok', 'message': 'API is running and healthy.'}

