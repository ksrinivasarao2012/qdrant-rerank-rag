import re
import zlib
from typing import List, Dict, Any, Tuple
from qdrant_client.http import models

# Dimensions of the sparse vector (the hashing space).
# 2**18 rather than 65536: a 100K-answer corpus has roughly 150-300K distinct
# tokens, so a 65K space collides heavily and unrelated words share a slot,
# producing false matches. Costs nothing extra -- sparse vectors only store
# non-zero entries, so a larger space does not mean larger vectors.
SPARSE_DIM = 262144  # 2**18

class SparseVectorGenerator:
    """
    State-free sparse vector generator utilizing the Hashing Trick (Feature Hashing).
    
    Why: By hashing words to a fixed dimension (65,536), we avoid storing or sync-ing
    a massive vocabulary mapping file on the server. It is 100% deterministic and stateless.
    """

    def __init__(self, vocabulary_size: int = SPARSE_DIM):
        self.vocab_size = vocabulary_size

    def tokenize(self, text: str) -> List[str]:
        """Splits text into lowercase alphanumeric words."""
        if not text:
            return []
        return re.findall(r'\w+', text.lower())

    def _hash_word(self, word: str) -> int:
        """Hashes a word deterministically to an index within the vocab size.

        CRC32 (zlib, C-implemented) instead of MD5: same determinism and
        cross-platform stability we need, but MD5 is a cryptographic hash and
        is markedly slower per call -- this runs once per word, for every
        chunk, across the whole corpus, so the difference adds up. Safe to
        change because indexing and querying both import this one class, so
        both sides stay consistent with each other regardless of which hash
        is used.
        """
        return zlib.crc32(word.encode('utf-8')) % self.vocab_size

    def generate_sparse_vector(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Generates a sparse vector representation (indices and values) for the given text.
        Values are calculated using normalized Term Frequency (TF).
        """
        words = self.tokenize(text)
        if not words:
            return [], []

        # Count frequencies
        freqs: Dict[int, int] = {}
        for word in words:
            idx = self._hash_word(word)
            freqs[idx] = freqs.get(idx, 0) + 1

        # Sort indices to comply with Qdrant requirements (Qdrant expects sorted indices)
        sorted_indices = sorted(freqs.keys())
        
        # Calculate raw term counts
        values = [float(freqs[idx]) for idx in sorted_indices]

        return sorted_indices, values

    def to_qdrant_sparse(self, text: str) -> models.SparseVector:
        """Helper to return Qdrant-compatible SparseVector structures."""
        indices, values = self.generate_sparse_vector(text)
        return models.SparseVector(
            indices=indices,
            values=values
        )
