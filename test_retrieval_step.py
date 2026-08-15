import os
import torch
# Prevent PyTorch / OpenMP CPU execution deadlocks on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)

import sys
from pathlib import Path

# Ensure backend can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

print("1. Loading embedding model...")
from backend.core.embeddings import get_embeddings
embedder = get_embeddings()

print("2. Embedding test query...")
vec = embedder.embed_query("What is bias variance tradeoff?")
print(f"   Embedding success! Vector dimension: {len(vec)}")

print("3. Connecting to Qdrant...")
from backend.core.vector_store import VectorDBManager
db = VectorDBManager()
collection_name = db.collection
print(f"   Connected! Collection name: {collection_name}")

print("4. Executing hybrid search...")
results = db.search_hybrid("What is bias variance tradeoff?", n_results=10)
print(f"   Hybrid search success! Chunks returned: {len(results)}")

print("5. Loading reranker...")
from backend.core.reranker import ReRanker
reranker = ReRanker()

print("6. Reranking results...")
reranked = reranker.rerank("What is bias variance tradeoff?", results, top_k=3)
print(f"   Rerank success! Got {len(reranked)} chunks.")
if reranked:
    print(f"   Top reranked chunk: {reranked[0]['text'][:100]}... (Score: {reranked[0]['rerank_score']})")

