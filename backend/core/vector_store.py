import os
import time
import uuid
import logging
from dotenv import load_dotenv
from typing import List, Any, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models

from backend.core.embeddings import get_embeddings, EMBEDDING_DIM
from backend.core.sparse_store import SparseVectorGenerator

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize sparse vector generator
sparse_generator = SparseVectorGenerator()

class VectorDBManager:
    def __init__(self, db_path: str = "./data/qdrant_db", collection_name: str = "stats_se_rag_docs"):
        """
        Initializes persistent Qdrant connection with lazy-loaded embeddings and sparse indexing.
        """
        os.makedirs(db_path, exist_ok=True)
        self.db_path = db_path
        self.collection_name = collection_name
        
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_url and qdrant_api_key:
            logger.info("Initializing VectorDBManager connecting to Qdrant Cloud...")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            logger.info(f"Initializing VectorDBManager locally at path={db_path}...")
            self.client = QdrantClient(path=db_path)
            
        logger.info("Qdrant client successfully created.")
        self._embedding = None
        # Set once collection_exists/create_payload_index have run successfully,
        # so `collection` doesn't re-check over the network on every batch.
        self._collection_ready = False

    @property
    def embedding(self):
        return get_embeddings()

    @property
    def collection(self):
        if self._collection_ready:
            return self.collection_name

        # Create collection with named vectors ('dense' and 'sparse') if it doesn't exist
        if not self.client.collection_exists(collection_name=self.collection_name):
            logger.info(f"Creating collection {self.collection_name} in Qdrant with hybrid indexing...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=EMBEDDING_DIM,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True  # Keeps RAM consumption very low
                        ),
                        modifier=models.Modifier.IDF
                    )
                }
            )
            logger.info(f"Collection {self.collection_name} created successfully.")
        
        # Ensure payload index on source_file exists for filtering
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source_file",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info("Payload index on 'source_file' verified/created.")
        except Exception as e:
            logger.warning(f"Could not create payload index on 'source_file': {e}")

        self._collection_ready = True
        return self.collection_name

    def delete_by_source(self, source_file: str) -> bool:
        """Deletes all existing vectors originating from a specific source file."""
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_file",
                                match=models.MatchValue(value=source_file),
                            )
                        ]
                    )
                )
            )
            logger.info(f"Cleared existing vectors for source file: {source_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete existing vectors for {source_file}: {e}")
            return False

    def add_chunks(self, chunks: List[Dict[str, Any]], source_file: str = None) -> bool:
        """
        Embeds chunks and uploads both dense and sparse vectors to Qdrant.
        """
        if not chunks:
            logger.warning("add_chunks called with empty chunks list")
            return False
        
        try:
            target_source = source_file or chunks[0]["metadata"].get("source_file")
            if target_source:
                self.delete_by_source(target_source)

            logger.info(f"Preparing to index {len(chunks)} chunks in Qdrant...")
            documents = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            # display_text (plain chunk, no title/overlap) falls back to the
            # embedded text for any chunk that doesn't carry one -- keeps this
            # backward-compatible with chunks built before display_text existed.
            display_texts = [chunk.get("display_text", chunk["text"]) for chunk in chunks]
            embeddings = self.embedding.embed_documents(documents)

            points = []
            for idx, (doc, meta, emb, display_text) in enumerate(zip(documents, metadatas, embeddings, display_texts)):
                # Assign unique deterministic point IDs
                string_id = f"{meta.get('answer_id', 'ans')}_{meta.get('chunk_index', 0)}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

                payload = {
                    "text": doc,
                    "display_text": display_text,
                    **meta
                }
                
                # Fetch sparse vector
                sparse_vec = sparse_generator.to_qdrant_sparse(doc)
                
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "dense": emb,
                            "sparse": sparse_vec
                        },
                        payload=payload
                    )
                )

            # Upsert into Qdrant, retrying on transient network failures so a
            # single dropped connection doesn't silently lose a whole batch.
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    self.client.upsert(
                        collection_name=self.collection,
                        points=points
                    )
                    logger.info(f"Successfully added {len(chunks)} chunks to Qdrant.")
                    return True
                except Exception as e:
                    if attempt < max_attempts:
                        wait = 2 ** attempt  # 2s, 4s
                        logger.warning(
                            f"Upsert attempt {attempt}/{max_attempts} failed ({e}); "
                            f"retrying in {wait}s..."
                        )
                        time.sleep(wait)
                    else:
                        raise
        except Exception as e:
            logger.exception(f"Failed to add chunks to Qdrant after retries ({len(chunks)} chunks lost).")
            return False

    def search(self, query: str, n_results: int = 3, source_file: str = None) -> List[Dict[str, Any]]:
        """
        Performs semantic (dense) search in Qdrant.
        """
        logger.info(f"Performing dense search query: '{query}' (limit={n_results})")
        try:
            query_vector = self.embedding.embed_query(query)
            query_filter = None
            if source_file:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchValue(value=source_file)
                        )
                    ]
                )
            results = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                using="dense",
                query_filter=query_filter,
                limit=n_results
            )
        except Exception as e:
            logger.exception(f"Dense query execution failed for query: '{query}'")
            return []
        
        return self._format_results(results)

    def search_sparse(self, query: str, n_results: int = 3, source_file: str = None) -> List[Dict[str, Any]]:
        """
        Performs keyword (sparse) search in Qdrant.
        """
        logger.info(f"Performing sparse search query: '{query}' (limit={n_results})")
        try:
            sparse_query = sparse_generator.to_qdrant_sparse(query)
            query_filter = None
            if source_file:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchValue(value=source_file)
                        )
                    ]
                )
            results = self.client.query_points(
                collection_name=self.collection,
                query=sparse_query,
                using="sparse",
                query_filter=query_filter,
                limit=n_results
            )
        except Exception as e:
            logger.exception(f"Sparse query execution failed for query: '{query}'")
            return []
        
        return self._format_results(results)

    def search_hybrid(self, query: str, n_results: int = 3, source_file: str = None) -> List[Dict[str, Any]]:
        """
        Performs hybrid search (dense + sparse) fused with RRF directly inside Qdrant.
        """
        logger.info(f"Performing native hybrid search query: '{query}' (limit={n_results})")
        try:
            query_vector = self.embedding.embed_query(query)
            sparse_query = sparse_generator.to_qdrant_sparse(query)
            
            query_filter = None
            if source_file:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchValue(value=source_file)
                        )
                    ]
                )
            
            # Request prefetching of dense and sparse vectors, then fuse them using RRF
            results = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using="dense",
                        limit=n_results * 5  # Fetch more candidates for better fusion
                    ),
                    models.Prefetch(
                        query=sparse_query,
                        using="sparse",
                        limit=n_results * 5
                    )
                ],
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF
                ),
                query_filter=query_filter,
                limit=n_results
            )
        except Exception as e:
            logger.exception(f"Hybrid query execution failed for query: '{query}'")
            return []
        
        return self._format_results(results)

    def _format_results(self, results) -> List[Dict[str, Any]]:
        """Helper to format Qdrant hits into standard RAG dictionary chunks."""
        clean_results = []
        for hit in results.points:
            payload = hit.payload or {}
            text = payload.get("text", "")
            meta = {k: v for k, v in payload.items() if k != "text"}
            clean_results.append({
                "id": str(hit.id),
                "text": text,
                "metadata": meta
            })
        return clean_results

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieves all stored chunks from Qdrant."""
        try:
            chunks = []
            offset = None
            while True:
                records, next_page = self.client.scroll(
                    collection_name=self.collection,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset
                )
                for record in records:
                    payload = record.payload or {}
                    text = payload.get("text", "")
                    meta = {k: v for k, v in payload.items() if k != "text"}
                    chunks.append({"id": str(record.id), "text": text, "metadata": meta})
                
                if next_page is None:
                    break
                offset = next_page
                
            logger.info(f"Retrieved {len(chunks)} chunks from Qdrant.")
            return chunks
        except Exception as e:
            logger.exception("Failed to retrieve all chunks from Qdrant.")
            return []
