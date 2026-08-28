"""
Shared embedding model.

Both DocumentProcessor (for SemanticChunker) and VectorDBManager (for indexing
and query encoding) need BAAI/bge-small-en-v1.5. They used to construct one
each, so the identical weights sat in memory twice -- roughly 130MB wasted, and
two separate warm-up costs on a cold start.

This module owns exactly one instance, created on first use. Loading is guarded
by a lock because Gradio and FastAPI both serve requests concurrently: without
it, two simultaneous first-calls could each start loading the model and one
would be constructed for nothing.

Usage:
    from backend.core.embeddings import get_embeddings
    vec = get_embeddings().embed_query("hello")
"""

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Upgraded from bge-small-en-v1.5 (33M params, 384-dim) to bge-base-en-v1.5
# (110M params, 768-dim) for significantly better paraphrase/semantic recall.
# IMPORTANT: Changing this requires deleting the existing Qdrant collection and
# re-running embed_corpus.py + upload_embeddings.py -- the vector dimension
# changed from 384 to 768, and the two are not compatible in the same collection.
MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEVICE = "cpu"
NORMALIZE = True

# Dimension of MODEL_NAME's output. VectorDBManager creates its Qdrant
# collection with this size, so it lives next to the model name it depends on.
EMBEDDING_DIM = 768

_instance: Optional[Any] = None
_lock = threading.Lock()


def get_embeddings() -> Any:
    """
    Returns the process-wide HuggingFaceEmbeddings instance, loading it on
    first call. Safe to call from multiple threads.
    """
    global _instance

    # Fast path: already loaded, no lock needed.
    if _instance is not None:
        return _instance

    with _lock:
        # Re-check inside the lock -- another thread may have loaded it while
        # we were waiting.
        if _instance is None:
            logger.info(f"Loading shared embedding model ({MODEL_NAME}) on {DEVICE}...")
            from langchain_community.embeddings import HuggingFaceEmbeddings

            if DEVICE == "cpu":
                import os
                import torch
                # sentence-transformers/torch sometimes under-uses available
                # cores on CPU unless told explicitly; harmless if already
                # using all of them.
                torch.set_num_threads(os.cpu_count() or 4)

            _instance = HuggingFaceEmbeddings(
                model_name=MODEL_NAME,
                model_kwargs={"device": DEVICE},
                # batch_size raised from the sentence-transformers default (32):
                # fewer, larger encode() calls means less Python/loop overhead
                # per chunk over an 88k-answer corpus. Output is identical --
                # only the batching of the same computation changes.
                encode_kwargs={"normalize_embeddings": NORMALIZE, "batch_size": 64},
            )
            logger.info("Shared embedding model loaded (single instance, reused process-wide).")

    return _instance


def is_loaded() -> bool:
    """Whether the model has been instantiated yet. Used by tests."""
    return _instance is not None


def reset() -> None:
    """Drops the instance so it reloads on next use. Tests only."""
    global _instance
    with _lock:
        _instance = None
