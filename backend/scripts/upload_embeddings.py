"""
Phase 2 of seeding: upload the locally-computed points in
data/processed/embedded_points.jsonl to Qdrant (Cloud or local) in batches.

This is the only phase that touches the network. It's fast (no embedding
compute), so if it fails partway, just re-run it — point IDs are deterministic
(uuid5 of answer_id + chunk_index), so re-upserting already-uploaded points is
a no-op overwrite, not a duplicate. VectorDBManager.add_chunks-style retry
(3 attempts, backoff) is applied per batch via the underlying client.

Run with: python backend/scripts/upload_embeddings.py
"""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client.http import models
from backend.core.vector_store import VectorDBManager

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

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "embedded_points_768.jsonl"


def to_point_struct(record: dict) -> models.PointStruct:
    return models.PointStruct(
        id=record["id"],
        vector={
            "dense": record["dense"],
            "sparse": models.SparseVector(
                indices=record["sparse"]["indices"],
                values=record["sparse"]["values"],
            ),
        },
        payload=record["payload"],
    )


def upload(input_path: Path, batch_size: int = 200):
    if not input_path.exists():
        print(f"Error: {input_path.resolve()} not found. Run embed_corpus.py first.")
        return 1

    print("Connecting to Qdrant...")
    db_manager = VectorDBManager()
    collection_name = db_manager.collection  # triggers one-time collection setup
    print(f"Using collection: '{collection_name}'")

    total_lines = sum(1 for _ in open(input_path, "r", encoding="utf-8"))
    print(f"Found {total_lines} embedded points to upload.")

    batch = []
    uploaded = 0
    failed_batches = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Uploading to Qdrant"):
            record = json.loads(line.strip())
            batch.append(to_point_struct(record))

            if len(batch) >= batch_size:
                if _upsert_with_retry(db_manager, collection_name, batch):
                    uploaded += len(batch)
                else:
                    failed_batches += 1
                batch = []
                time.sleep(0.5)  # brief pause to avoid CPU throttling on free-tier

        if batch:
            if _upsert_with_retry(db_manager, collection_name, batch):
                uploaded += len(batch)
            else:
                failed_batches += 1

    print("\n=== UPLOAD SUMMARY ===")
    print(f"Uploaded {uploaded}/{total_lines} points.")
    if failed_batches:
        print(f"WARNING: {failed_batches} batch(es) failed after retries — re-run this "
              f"script to retry them (already-uploaded points are overwritten, not duplicated).")
    else:
        print("All batches uploaded successfully.")

    # Pre-compute and save distinct topics.json at index time for O(1) route reads
    try:
        topics = set()
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    payload = json.loads(line.strip()).get("payload", {})
                    topics.update(payload.get("tags") or [])
        topics_path = input_path.parent / "topics.json"
        with open(topics_path, "w", encoding="utf-8") as f:
            json.dump({"topics": sorted(list(topics))}, f, indent=2)
        print(f"Pre-computed and saved {len(topics)} distinct topic tags to '{topics_path}'.")
    except Exception as e:
        print(f"Warning: Could not pre-compute topics.json: {e}")

    return 0 if failed_batches == 0 else 1


def _upsert_with_retry(db_manager: VectorDBManager, collection_name: str, points, max_attempts=5) -> bool:
    for attempt in range(1, max_attempts + 1):
        try:
            db_manager.client.upsert(collection_name=collection_name, points=points)
            return True
        except Exception as e:
            if attempt < max_attempts:
                wait = 2 ** attempt + 2  # 6s, 10s, 18s, 34s
                print(f"  Batch upload attempt {attempt}/{max_attempts} failed ({e}); retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Batch upload failed after {max_attempts} attempts ({e}); skipping for now.")
    return False


def main():
    return upload(INPUT_PATH)


if __name__ == "__main__":
    sys.exit(main())
