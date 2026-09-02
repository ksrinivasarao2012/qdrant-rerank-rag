"""
Whole-Answer Embedding Script (Option 2)

Generates clean 768-dim dense embeddings (BAAI/bge-base-en-v1.5) and sparse BM25
vectors for posts.jsonl WITHOUT unnecessary chunking or boilerplate title prefixes.

Posts <= 2,200 chars (~450 tokens) are kept 100% WHOLE.
Only posts > 2,200 chars are cleanly chunked.

Output: data/processed/embedded_points_v2.jsonl
"""

import os
import sys
import json
import time
import uuid
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.embeddings import get_embeddings
from backend.core.sparse_store import SparseVectorGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_POSTS = PROJECT_ROOT / "data" / "processed" / "posts.jsonl"
OUTPUT_POINTS = PROJECT_ROOT / "data" / "processed" / "embedded_points_v2.jsonl"
MAX_WHOLE_CHARS = 2200  # ~450 tokens (91% of posts fit whole)
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def create_chunks_for_post(post: dict) -> list[dict]:
    answer_text = (post.get("answer_text") or "").strip()
    if not answer_text:
        return []

    meta = {
        "answer_id": post["answer_id"],
        "question_id": post.get("question_id"),
        "question_title": post.get("question_title", ""),
        "tags": post.get("tags", []),
        "score": post.get("score", 0),
        "is_accepted": post.get("is_accepted", False),
        "created": post.get("created", ""),
        "url": post.get("url", "")
    }

    # Short to Medium Posts: Keep 100% Whole (No chunking, no boilerplate)
    if len(answer_text) <= MAX_WHOLE_CHARS:
        string_id = f"{post['answer_id']}_0"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))
        return [{
            "id": point_id,
            "text": answer_text,
            "display_text": f"Question: {meta['question_title']}\n\nAnswer: {answer_text}",
            "metadata": {**meta, "chunk_index": 0, "total_chunks": 1}
        }]

    # Long Posts (> 450 tokens): Chunk into 2000-char windows
    chunks = []
    start = 0
    chunk_index = 0
    text_len = len(answer_text)

    while start < text_len:
        end = start + CHUNK_SIZE
        chunk_text = answer_text[start:end].strip()

        if chunk_text:
            string_id = f"{post['answer_id']}_{chunk_index}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))
            chunks.append({
                "id": point_id,
                "text": chunk_text,
                "display_text": f"Question: {meta['question_title']}\n\nAnswer Fragment: {chunk_text}",
                "metadata": {**meta, "chunk_index": chunk_index}
            })
            chunk_index += 1

        start += (CHUNK_SIZE - CHUNK_OVERLAP)

    for c in chunks:
        c["metadata"]["total_chunks"] = len(chunks)

    return chunks


def main():
    if not INPUT_POSTS.exists():
        logger.error(f"Input file not found: {INPUT_POSTS}")
        return 1

    logger.info("Initializing HuggingFace Embeddings (BAAI/bge-base-en-v1.5, 768-dim)...")
    embeddings_model = get_embeddings()
    sparse_gen = SparseVectorGenerator()

    logger.info(f"Reading {INPUT_POSTS} and generating whole-answer units...")
    all_chunks = []
    with open(INPUT_POSTS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                post = json.loads(line.strip())
                all_chunks.extend(create_chunks_for_post(post))

    logger.info(f"Generated {len(all_chunks)} units (reduced from 218,456 chunks to ~{len(all_chunks)})!")

    batch_size = 64
    total = len(all_chunks)
    t0 = time.time()

    logger.info(f"Starting batch embedding into {OUTPUT_POINTS}...")
    OUTPUT_POINTS.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_POINTS, "w", encoding="utf-8") as out_f:
        for i in range(0, total, batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]

            # Compute 768-dim dense embeddings in batch
            dense_vectors = embeddings_model.embed_documents(texts)

            for item, dense_vec in zip(batch, dense_vectors):
                sparse_vec = sparse_gen.to_qdrant_sparse(item["text"])

                record = {
                    "id": item["id"],
                    "dense": dense_vec,
                    "sparse": {
                        "indices": sparse_vec.indices,
                        "values": sparse_vec.values
                    },
                    "payload": {
                        "text": item["text"],
                        "display_text": item["display_text"],
                        **item["metadata"]
                    }
                }
                out_f.write(json.dumps(record) + "\n")

            if (i + batch_size) % 3200 < batch_size or (i + batch_size) >= total:
                elapsed = time.time() - t0
                rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
                logger.info(f"Embedded {i + len(batch)}/{total} points ({rate:.1f} points/sec)...")

    logger.info(f"Successfully generated whole-answer embeddings: {OUTPUT_POINTS} ({total} points total).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
