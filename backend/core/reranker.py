import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            logger.info(f"Lazy-loading Cross-Encoder model: {self.model_name}")
            from sentence_transformers import CrossEncoder
            self._encoder = CrossEncoder(self.model_name, max_length=512)
            logger.info("Cross-Encoder loaded successfully.")
        return self._encoder
    
    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Takes a list of retrieved chunks and re-scores them using a Cross-Encoder.
        """
        if not chunks:
            return []

        # 1. Format pairs for the Cross-Encoder: [(query, text1), (query, text2), ...]
        sentence_pairs = [[query, chunk["text"]] for chunk in chunks]

        # 2. Predict relevance scores (0.0 to 1.0)
        logger.debug(f"Scoring {len(chunks)} chunks with Cross-Encoder...")
        scores = self.encoder.predict(sentence_pairs)

        # 3. Attach scores to the chunks
        for idx, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[idx])

        # 4. Sort chunks by score in descending order (highest score first)
        reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        # 5. Return only the top_k chunks
        logger.info(f"Re-ranking complete. Returning top {top_k} chunks.")
        return reranked_chunks[:top_k]
        