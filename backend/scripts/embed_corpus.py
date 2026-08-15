"""
Phase 1 of seeding: compute chunks + dense embeddings + sparse vectors
entirely locally, writing results to data/processed/embedded_points.jsonl.

No network calls happen in this script — it cannot fail due to Qdrant Cloud
connectivity. That failure mode (transient DNS/socket errors dropping whole
batches mid-run, see seed_corpus.py history) is why embedding is now split
from uploading: a multi-hour local computation should never be at risk from
a flaky network connection.

Run with: python backend/scripts/embed_corpus.py
Then:     python backend/scripts/upload_embeddings.py
"""

import sys
import json
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.ingestion import DocumentProcessor
from backend.core.embeddings import get_embeddings
from backend.core.sparse_store import SparseVectorGenerator

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        total = kwargs.get("total", 0)
        desc = kwargs.get("desc", "Processing")
        print(f"Starting {desc} (total items to process: {total})...")
        for i, item in enumerate(iterable):
            yield item
            if total > 0 and (i + 1) % 5000 == 0:
                print(f"  {desc}: {i + 1}/{total} completed...")

JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "posts.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "embedded_points.jsonl"


def point_id(answer_id, chunk_index) -> str:
    string_id = f"{answer_id}_{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))


def _already_embedded_answer_ids(output_path: Path) -> set:
    """Reads whatever's already in embedded_points.jsonl (from a prior,
    possibly interrupted run) and returns the set of answer_ids fully
    represented there, so a resumed run can skip them instead of
    recomputing embeddings for answers already done."""
    if not output_path.exists():
        return set()
    answer_ids = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                answer_ids.add(record["payload"]["answer_id"])
            except (json.JSONDecodeError, KeyError):
                # Last line of a prior run may be a partial/truncated write
                # (e.g. crash mid-flush) -- skip it, its answer gets redone.
                continue
    return answer_ids


def embed(jsonl_path: Path, output_path: Path, batch_size: int = 512):
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path.resolve()} not found. Run parse_dump.py first.")
        return 1

    print("Loading embedding model...")
    embedding_model = get_embeddings()
    sparse_generator = SparseVectorGenerator()
    processor = DocumentProcessor()

    total_lines = sum(1 for _ in open(jsonl_path, "r", encoding="utf-8"))
    print(f"Found {total_lines} answers to process.")

    done_answer_ids = _already_embedded_answer_ids(output_path)
    resuming = bool(done_answer_ids)
    if resuming:
        print(f"Resuming: {len(done_answer_ids)} answers already embedded in "
              f"{output_path.name}, skipping them.")

    chunk_count = 0
    text_batch = []          # list of chunk texts pending embedding
    display_text_batch = []  # matching plain (no title/overlap) text, for citations
    meta_batch = []          # matching metadata for each pending chunk

    # Append if resuming (keep prior progress), otherwise start fresh.
    file_mode = "a" if resuming else "w"

    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_path, file_mode, encoding="utf-8") as fout:

        for line in tqdm(fin, total=total_lines, desc="Chunking + embedding"):
            post = json.loads(line.strip())
            if post["answer_id"] in done_answer_ids:
                continue
            metadata = {
                "answer_id": post["answer_id"],
                "question_id": post["question_id"],
                "question_title": post["question_title"],
                "score": post["score"],
                "is_accepted": post["is_accepted"],
                "tags": post["tags"],
                "created": post["created"],
                "url": post["url"],
            }
            chunks = processor.process_answer(post.get("answer_text", ""), metadata)

            # Flush BEFORE adding this answer's chunks if the batch is
            # already full, rather than mid-answer. _already_embedded_answer_ids()
            # marks an answer "done" if ANY of its chunks appear in the output
            # file -- if a crash happened between two flushes that each wrote
            # only part of one multi-chunk answer, resume would wrongly skip
            # it forever, leaving it permanently incomplete. Keeping each
            # answer's chunks atomic across flushes avoids that.
            if text_batch and len(text_batch) + len(chunks) > batch_size:
                _flush_batch(text_batch, display_text_batch, meta_batch, embedding_model, sparse_generator, fout)
                chunk_count += len(text_batch)
                text_batch, display_text_batch, meta_batch = [], [], []

            for chunk in chunks:
                text_batch.append(chunk["text"])
                display_text_batch.append(chunk.get("display_text", chunk["text"]))
                meta_batch.append(chunk["metadata"])

        if text_batch:
            _flush_batch(text_batch, display_text_batch, meta_batch, embedding_model, sparse_generator, fout)
            chunk_count += len(text_batch)

    print(f"\nWrote {chunk_count} embedded chunks to {output_path.resolve()}")
    print("Next: python backend/scripts/upload_embeddings.py")
    return 0


def _flush_batch(text_batch, display_text_batch, meta_batch, embedding_model, sparse_generator, fout):
    dense_vectors = embedding_model.embed_documents(text_batch)
    for text, display_text, meta, dense in zip(text_batch, display_text_batch, meta_batch, dense_vectors):
        indices, values = sparse_generator.generate_sparse_vector(text)
        record = {
            "id": point_id(meta.get("answer_id", "ans"), meta.get("chunk_index", 0)),
            "dense": dense,
            "sparse": {"indices": indices, "values": values},
            "payload": {"text": text, "display_text": display_text, **meta},
        }
        fout.write(json.dumps(record) + "\n")
    fout.flush()


def main():
    return embed(JSONL_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    sys.exit(main())
