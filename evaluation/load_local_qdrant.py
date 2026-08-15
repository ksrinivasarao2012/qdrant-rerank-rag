"""
One-off diagnostic: build a LOCAL, on-disk Qdrant collection directly from
data/processed/embedded_points.jsonl -- no network, no Qdrant Cloud.

VectorDBManager already supports this: when QDRANT_URL / QDRANT_API_KEY are
unset it falls back to an embedded on-disk client (QdrantClient(path=...)).
This script seeds that local collection from the known-good, current
embedded_points.jsonl, so we can run eval_retriever.py against a guaranteed-
clean index and compare against the live Qdrant Cloud numbers. If recall
jumps back up locally, that confirms the live collection's stale points
(the 16,140-point discrepancy) are the cause, not the retrieval pipeline
itself.

Run locally (no network needed -- this must NOT hit Qdrant Cloud, so make
sure QDRANT_URL / QDRANT_API_KEY are unset for this run):

    (PowerShell)
    $env:QDRANT_URL=""; $env:QDRANT_API_KEY=""; python evaluation/load_local_qdrant.py

Then evaluate against the same local collection the same way:

    $env:QDRANT_URL=""; $env:QDRANT_API_KEY=""; python evaluation/eval_retriever.py
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Belt-and-suspenders: blank these out in-process too, in case the shell
# didn't clear them, so this can never accidentally write to Qdrant Cloud.
os.environ["QDRANT_URL"] = ""
os.environ["QDRANT_API_KEY"] = ""

from qdrant_client.http import models
from backend.core.vector_store import VectorDBManager

EMBEDDED_POINTS_PATH = PROJECT_ROOT / "data" / "processed" / "embedded_points.jsonl"
BATCH_SIZE = 500


def main():
    if not EMBEDDED_POINTS_PATH.exists():
        print(f"Error: {EMBEDDED_POINTS_PATH} not found.")
        return 1

    db = VectorDBManager()
    print(f"Local Qdrant path: {db.db_path}")
    if db.client.__class__.__name__ != "QdrantClient" or os.environ.get("QDRANT_URL"):
        pass  # can't easily introspect mode, but env vars above guarantee local
    db.collection  # creates the collection with the right dense/sparse config

    batch = []
    total = 0
    with open(EMBEDDED_POINTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            batch.append(
                models.PointStruct(
                    id=rec["id"],
                    vector={
                        "dense": rec["dense"],
                        "sparse": models.SparseVector(
                            indices=rec["sparse"]["indices"],
                            values=rec["sparse"]["values"],
                        ),
                    },
                    payload=rec["payload"],
                )
            )
            if len(batch) >= BATCH_SIZE:
                db.client.upsert(collection_name=db.collection, points=batch)
                total += len(batch)
                print(f"  upserted {total}...")
                batch = []

    if batch:
        db.client.upsert(collection_name=db.collection, points=batch)
        total += len(batch)

    print(f"Done. {total} points loaded into local collection '{db.collection_name}' at {db.db_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
