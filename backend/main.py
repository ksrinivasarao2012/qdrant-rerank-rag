import logging
from fastapi import FastAPI
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
    happens to arrive first. Build-time steps (Dockerfile / render.yaml)
    already avoid re-downloading these models, but downloading isn't the
    same as loading: get_embeddings()/reranker.encoder are lazy singletons
    that only actually load into memory on first access, so without this,
    the first real user still eats that load time. Warms the exact same
    instances routes.py uses (imported above), not fresh ones."""
    logger.info("Warming models at startup...")
    vector_db.embedding  # triggers the shared embedding singleton to load
    reranker.encoder     # triggers the cross-encoder to load
    vector_db.collection  # confirms Qdrant is reachable now, not on first request
    logger.info("Models warmed -- ready to serve requests.")

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
    return {'status' : 'ok','message' : 'API is running.'}
