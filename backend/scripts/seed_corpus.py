"""
DEPRECATED — do not use.

This script embedded and upserted to Qdrant in a single interleaved pass,
which meant a transient network failure (Qdrant Cloud DNS/socket errors)
silently dropped whole batches of chunks with no retry and no record of what
was lost. That's what actually happened during a real seeding run.

Replaced by a two-phase approach:
  1. python backend/scripts/embed_corpus.py     (local only, no network,
     resumable if interrupted)
  2. python backend/scripts/upload_embeddings.py (network only, retries
     transient failures, safe to re-run — point IDs are deterministic so
     re-uploading is an overwrite, not a duplicate)

Kept as a stub (not deleted) so the failure history stays visible in the
codebase rather than disappearing silently.
"""

import sys

if __name__ == "__main__":
    print(__doc__)
    sys.exit(1)
