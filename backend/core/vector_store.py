import os
import time
import uuid
import json
import logging
from pathlib import Path
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
    DEFAULT_COLLECTION = "stats_se_rag_docs"

    def __init__(self, db_path: str = "./data/qdrant_db", collection_name: str = None,
                 force_local: bool = False):
        """
        Initializes persistent Qdrant connection with lazy-loaded embeddings and sparse indexing.

        force_local: bypasses QDRANT_URL/QDRANT_API_KEY entirely and always uses the
        local on-disk client, regardless of what's in .env. Needed because blanking
        those env vars in the shell before running a script (e.g.
        `$env:QDRANT_URL=""`) does NOT reliably work on Windows -- PowerShell's
        $env: provider (via .NET's SetEnvironmentVariable) treats assigning an
        empty string as DELETING the variable, not setting it to blank. Once
        deleted, python-dotenv's load_dotenv() (called below) sees it as absent
        and silently reloads the real value from .env, defeating the trick and
        connecting to Qdrant Cloud instead of the intended local index. This flag
        sidesteps the whole problem instead of relying on shell env manipulation.
        """
        os.makedirs(db_path, exist_ok=True)
        self.db_path = db_path
        # Collection resolution order: explicit argument > QDRANT_COLLECTION env
        # var > default. The env var exists because every eval/diagnostic script
        # constructs this class with no collection_name, so before this there was
        # no way to point a whole run at an alternate index (e.g. a v2 rebuild)
        # short of editing the default here. Setting $env:QDRANT_COLLECTION and
        # running an eval silently evaluated the OLD collection instead -- no
        # error, just wrong numbers attributed to the wrong index.
        self.collection_name = (
            collection_name
            or os.getenv("QDRANT_COLLECTION")
            or self.DEFAULT_COLLECTION
        )

        qdrant_url = None if force_local else os.getenv("QDRANT_URL")
        qdrant_api_key = None if force_local else os.getenv("QDRANT_API_KEY")

        if qdrant_url and qdrant_api_key:
            logger.info("Initializing VectorDBManager connecting to Qdrant Cloud...")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60.0)
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
        
        # Ensure payload index on 'tags' exists for topic filtering. Was
        # 'source_file' -- a leftover from the old PDF-upload version, where
        # each chunk's payload actually had that field. The current
        # StackExchange corpus (embed_corpus.py) never sets 'source_file' on
        # any chunk; the real filterable field is 'tags'. Leaving the old
        # field name here meant the topic filter dropdown in app.py silently
        # matched nothing for any specific topic -- only "All Topics" (no
        # filter) ever worked.
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="tags",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info("Payload index on 'tags' verified/created.")
        except Exception as e:
            logger.warning(f"Could not create payload index on 'tags': {e}")

        self._collection_ready = True
        return self.collection_name

    def delete_by_source(self, source_file: str) -> bool:
        """Deletes all vectors carrying a given tag. Method name still says
        'source' for backward compatibility with the old PDF-upload version
        of the code -- semantically it's a tag filter now, matching against
        the 'tags' payload field (see _tag_filter above for the same fix)."""
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(filter=self._tag_filter(source_file))
            )
            logger.info(f"Cleared existing vectors for tag: {source_file}")
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

    def _tag_filter(self, source_file):
        """Builds a Qdrant filter for a single tag. Named parameter is still
        called `source_file` in callers (routes.py, app.py) for backward
        compatibility -- semantically it's a tag now, matched against the
        payload's 'tags' field which is the only one actually present on
        chunks in the current StackExchange corpus. Old field name 'source_file'
        never gets set by embed_corpus.py, so the previous filter on that key
        silently matched nothing -- see the bug flagged around the payload
        index creation above."""
        if not source_file:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="tags",
                    match=models.MatchValue(value=source_file)
                )
            ]
        )

    def _combine_filters(self, source_file: str = None, qdrant_filter: Any = None) -> Any:
        """Merges topic (tag) filters and custom Qdrant filters."""
        tag_filt = self._tag_filter(source_file) if source_file else None
        if not tag_filt:
            return qdrant_filter
        if not qdrant_filter:
            return tag_filt
            
        must_conditions = []
        must_not_conditions = []
        
        if tag_filt.must:
            must_conditions.extend(tag_filt.must)
        if tag_filt.must_not:
            must_not_conditions.extend(tag_filt.must_not)
            
        if qdrant_filter.must:
            must_conditions.extend(qdrant_filter.must)
        if qdrant_filter.must_not:
            must_not_conditions.extend(qdrant_filter.must_not)
            
        return models.Filter(
            must=must_conditions if must_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None
        )

    def search(self, query: str, n_results: int = 3, source_file: str = None, qdrant_filter: Any = None) -> List[Dict[str, Any]]:
        """
        Performs semantic (dense) search in Qdrant.
        """
        logger.info(f"Performing dense search query: '{query}' (limit={n_results})")
        try:
            query_vector = self.embedding.embed_query(query)
            results = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                using="dense",
                query_filter=self._combine_filters(source_file, qdrant_filter),
                limit=n_results
            )
        except Exception as e:
            logger.exception(f"Dense query execution failed for query: '{query}'")
            return []

        return self._format_results(results)

    def search_sparse(self, query: str, n_results: int = 3, source_file: str = None, qdrant_filter: Any = None) -> List[Dict[str, Any]]:
        """
        Performs keyword (sparse) search in Qdrant.
        """
        logger.info(f"Performing sparse search query: '{query}' (limit={n_results})")
        try:
            sparse_query = sparse_generator.to_qdrant_sparse(query)
            results = self.client.query_points(
                collection_name=self.collection,
                query=sparse_query,
                using="sparse",
                query_filter=self._combine_filters(source_file, qdrant_filter),
                limit=n_results
            )
        except Exception as e:
            logger.exception(f"Sparse query execution failed for query: '{query}'")
            return []

        return self._format_results(results)

    def search_hybrid(self, query: str, n_results: int = 3, source_file: str = None, qdrant_filter: Any = None) -> List[Dict[str, Any]]:
        """
        Performs hybrid search (dense + sparse) fused with RRF directly inside Qdrant.
        Dense embedding and sparse encoding are executed concurrently for low latency.
        """
        logger.info(f"Performing native hybrid search query: '{query}' (limit={n_results})")
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_dense = executor.submit(self.embedding.embed_query, query)
                f_sparse = executor.submit(sparse_generator.to_qdrant_sparse, query)
                query_vector = f_dense.result()
                sparse_query = f_sparse.result()
            
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
                query_filter=self._combine_filters(source_file, qdrant_filter),
                limit=n_results * 3  # Fetch expanded pool for answer-level deduplication
            )
        except Exception as e:
            logger.exception(f"Hybrid query execution failed for query: '{query}'")
            return []
        
        return self._format_results(results)[:n_results]

    def _format_results(self, results, deduplicate_by_answer: bool = True) -> List[Dict[str, Any]]:
        """
        Helper to format Qdrant hits into standard RAG dictionary chunks.
        Deduplicates by answer_id to return unique post hits rather than multiple
        fragmented chunks of the same post cluttering the candidate pool.
        """
        clean_results = []
        seen_answer_ids = set()
        for hit in results.points:
            payload = hit.payload or {}
            text = payload.get("text", "")
            meta = {k: v for k, v in payload.items() if k != "text"}
            answer_id = meta.get("answer_id")
            
            if deduplicate_by_answer and answer_id:
                if answer_id in seen_answer_ids:
                    continue
                seen_answer_ids.add(answer_id)
                
            clean_results.append({
                "id": str(hit.id),
                "text": text,
                "metadata": meta
            })
        return clean_results

    def search_multi_query(
        self,
        queries: List[str],
        n_results: int = 10,
        source_file: str = None,
        qdrant_filter: Any = None
    ) -> List[Dict[str, Any]]:
        """
        Executes parallel hybrid searches across multiple sub-queries (e.g. decomposed multi-hop comparisons)
        and merges candidates using Reciprocal Rank Fusion (RRF).
        """
        if not queries:
            return []
        if len(queries) == 1:
            return self.search_hybrid(queries[0], n_results=n_results, source_file=source_file, qdrant_filter=qdrant_filter)

        logger.info(f"Executing Multi-Query Parallel Hybrid Search for {len(queries)} sub-queries: {queries}")
        from concurrent.futures import ThreadPoolExecutor
        
        all_query_results = []
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = [
                executor.submit(self.search_hybrid, q, n_results=n_results * 2, source_file=source_file, qdrant_filter=qdrant_filter)
                for q in queries
            ]
            for f in futures:
                try:
                    all_query_results.append(f.result())
                except Exception as e:
                    logger.warning(f"Sub-query execution failed: {e}")

        # Reciprocal Rank Fusion across all sub-queries
        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Dict[str, Any]] = {}
        RRF_K = 60

        for sub_results in all_query_results:
            for rank, doc in enumerate(sub_results, start=1):
                doc_id = doc["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + rank))
                if doc_id not in doc_lookup:
                    doc_lookup[doc_id] = doc

        # Sort documents by fused RRF score descending
        fused_sorted_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)
        return [doc_lookup[did] for did in fused_sorted_ids[:n_results]]


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

    def get_topics(self) -> List[str]:
        """
        O(1) pre-computed read of topics.json.
        Falls back to payload-only scroll if topics.json does not exist.
        """
        try:
            topics_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "topics.json"
            if topics_path.exists():
                with open(topics_path, "r", encoding="utf-8") as f:
                    return json.load(f).get("topics", [])
        except Exception as e:
            logger.warning(f"Could not read precomputed topics.json ({e}); falling back to payload-only scroll.")

        # Fallback: payload-only scroll over Qdrant (with_payload=["tags"], limit=10_000)
        try:
            tags, offset = set(), None
            while True:
                records, offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=10_000,
                    with_payload=["tags"],
                    with_vectors=False,
                    offset=offset
                )
                for r in records:
                    tags.update((r.payload or {}).get("tags") or [])
                if offset is None:
                    break
            return sorted(tags)
        except Exception as e:
            logger.exception("Failed to fetch topics via payload scroll.")
            return []

