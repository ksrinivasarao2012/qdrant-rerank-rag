import logging
import requests
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._encoder = None
        self._http_session = requests.Session()

    @property
    def encoder(self):
        if self._encoder is None:
            logger.info(f"Lazy-loading Cross-Encoder model: {self.model_name}")
            try:
                from sentence_transformers import CrossEncoder
                self._encoder = CrossEncoder(self.model_name, max_length=512)
                logger.info("Cross-Encoder loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Cross-Encoder model '{self.model_name}': {e}. Reranking will be bypassed.")
                self._encoder = "FAILED"
        return self._encoder
    
    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Takes a list of retrieved chunks and re-scores them.
        Uses Jina AI API if available, otherwise falls back to local Cross-Encoder.
        """
        if not chunks:
            return []

        from backend.core.config import SETTINGS

        if SETTINGS.JINA_API_KEY:
            logger.info("Using Jina Rerank API with HTTP Session reuse...")
            try:
                headers = {
                    "Authorization": f"Bearer {SETTINGS.JINA_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "top_n": top_k,
                    "documents": [chunk["text"][:600] for chunk in chunks]
                }
                res = self._http_session.post("https://api.jina.ai/v1/rerank", headers=headers, json=payload, timeout=5)
                if res.status_code == 200:
                    results = res.json().get("results", [])
                    
                    # Jina returns results sorted by relevance. We need to map them back to our chunks.
                    # The 'results' array contains dicts like {"index": 2, "relevance_score": 0.89}
                    reranked_chunks = []
                    for item in results:
                        original_idx = item["index"]
                        chunk = chunks[original_idx]
                        chunk["rerank_score"] = float(item["relevance_score"])
                        reranked_chunks.append(chunk)
                    
                    logger.info("Jina API re-ranking complete.")
                    return reranked_chunks
                else:
                    logger.warning(f"Jina API failed ({res.status_code}): {res.text}. Falling back to local Cross-Encoder.")
            except Exception as e:
                logger.warning(f"Jina API connection error: {e}. Falling back to local Cross-Encoder.")

        # --- Fallback to Local Cross-Encoder ---
        encoder = self.encoder
        if encoder == "FAILED" or encoder is None:
            logger.warning("Bypassing Cross-Encoder reranking (model failed to load).")
            return chunks[:top_k]

        # 1. Format pairs for the Cross-Encoder: [(query, text1), (query, text2), ...]
        sentence_pairs = [[query, chunk["text"][:600]] for chunk in chunks]

        # 2. Predict relevance scores (0.0 to 1.0)
        logger.debug(f"Scoring {len(chunks)} chunks with local Cross-Encoder...")
        scores = encoder.predict(sentence_pairs)

        # 3. Attach scores to the chunks
        for idx, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[idx])

        # 4. Sort chunks by score in descending order (highest score first)
        reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        # 5. Return only the top_k chunks
        logger.info(f"Local re-ranking complete. Returning top {top_k} chunks.")
        return reranked_chunks[:top_k]
        