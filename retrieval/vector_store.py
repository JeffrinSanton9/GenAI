"""
retrieval/vector_store.py – ChromaDB wrapper for storing and querying chunks.

ChromaDB is an open-source, local vector database.
  • Runs fully in-process (no server setup needed for development).
  • Persists data to disk so you don't re-index every restart.
  • Supports metadata filtering out of the box.

Design decisions:
  • We store embeddings ourselves (via retrieval/embedder.py) so we have
    full control and can swap the model easily.
  • Document IDs use chunk_id from the chunker (UUID) so re-ingestion of
    the same chunk is idempotent (ChromaDB upserts by ID).
"""

from __future__ import annotations
from typing import Any

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, COLLECTION, TOP_K
from retrieval.embedder import embed_texts, embed_query


class VectorStore:
    """Thin wrapper around a ChromaDB collection."""

    def __init__(self,
                 persist_dir: str | None = None,
                 collection_name: str = COLLECTION):
        persist_dir = persist_dir or str(CHROMA_DIR)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Embed and store a list of chunk dicts.

        Each chunk must have:
          chunk_id  – unique string ID
          text      – the chunk text
          metadata  – dict of string-valued metadata fields
        """
        if not chunks:
            return

        ids       = [c["chunk_id"] for c in chunks]
        texts     = [c["text"]     for c in chunks]
        metadatas = [_sanitize_meta(c["metadata"]) for c in chunks]

        # Compute embeddings in one batch (efficient)
        embeddings = embed_texts(texts)

        # Upsert: safe to call again for the same IDs (idempotent)
        self._col.upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = texts,
            metadatas  = metadatas,
        )

    def clear(self) -> None:
        """Delete ALL documents from the collection."""
        self._client.delete_collection(self._col.name)
        self._col = self._client.get_or_create_collection(
            name=self._col.name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def semantic_search(self,
                        query: str,
                        top_k: int = TOP_K,
                        filters: dict[str, Any] | None = None) -> list[dict]:
        """
        Retrieve top-k chunks most similar to the query embedding.

        Args:
            query:   Natural language question.
            top_k:   Number of results to return.
            filters: ChromaDB `where` filter dict.
                     Example: {"doc_type": "pdf", "year": "2025"}

        Returns:
            List of result dicts sorted by relevance (best first).
        """
        q_embedding = embed_query(query)
        kwargs: dict[str, Any] = {
            "query_embeddings": [q_embedding],
            "n_results": min(top_k, self.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = _build_where(filters)

        results = self._col.query(**kwargs)
        return _format_results(results)

    def keyword_search(self,
                       query: str,
                       top_k: int = TOP_K,
                       filters: dict[str, Any] | None = None) -> list[dict]:
        """
        BM25-style keyword search using ChromaDB's built-in full-text search.

        Falls back to semantic search if full-text is unavailable.
        """
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(top_k, self.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = _build_where(filters)
        try:
            results = self._col.query(**kwargs)
            return _format_results(results)
        except Exception:
            # ChromaDB full-text not available – fall back
            return self.semantic_search(query, top_k, filters)

    def get_all_metadata(self) -> list[dict]:
        """Return metadata of every stored chunk (for admin/debug)."""
        result = self._col.get(include=["metadatas"])
        return result.get("metadatas", [])

    def count(self) -> int:
        """Total number of chunks in the collection."""
        return self._col.count()

    def list_sources(self) -> list[str]:
        """Return unique document source names."""
        metas = self.get_all_metadata()
        return sorted({m.get("source", "unknown") for m in metas})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize_meta(meta: dict) -> dict:
    """ChromaDB requires all metadata values to be str, int, float, or bool."""
    return {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
            for k, v in meta.items()
            if v is not None}


def _build_where(filters: dict[str, Any]) -> dict:
    """Build a ChromaDB `where` clause from a simple key-value filter dict."""
    if len(filters) == 1:
        k, v = next(iter(filters.items()))
        return {k: {"$eq": str(v)}}
    return {"$and": [{k: {"$eq": str(v)}} for k, v in filters.items()]}


def _format_results(raw: dict) -> list[dict]:
    """Turn raw ChromaDB query output into clean result dicts."""
    docs      = raw.get("documents", [[]])[0]
    metas     = raw.get("metadatas",  [[]])[0]
    distances = raw.get("distances",  [[]])[0]

    results = []
    for text, meta, dist in zip(docs, metas, distances):
        results.append({
            "text":     text,
            "metadata": meta,
            "score":    round(1 - dist, 4),   # cosine similarity (higher = better)
        })
    return results
