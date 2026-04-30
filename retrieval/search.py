"""
retrieval/search.py – Keyword, semantic, and hybrid search interfaces.

Search modes:
  • keyword  – BM25-based exact/term match (great for codes, IDs, exact phrases).
  • semantic  – Dense embedding search (great for paraphrases, meaning).
  • hybrid    – Weighted fusion of both (best of both worlds).

Hybrid search uses Reciprocal Rank Fusion (RRF), a simple, effective method
that combines ranked lists without needing score normalisation.
"""

from __future__ import annotations
from typing import Any

from rank_bm25 import BM25Okapi

from config import TOP_K, HYBRID_ALPHA
from retrieval.vector_store import VectorStore


# ── Public API ────────────────────────────────────────────────────────────────

def search(query: str,
           mode: str = "hybrid",
           top_k: int = TOP_K,
           filters: dict[str, Any] | None = None,
           store: VectorStore | None = None) -> list[dict]:
    """
    Unified search entry point.

    Args:
        query:   The user's natural-language question.
        mode:    "keyword" | "semantic" | "hybrid"  (default: hybrid)
        top_k:   Number of results to return.
        filters: Metadata filter dict, e.g. {"doc_type": "pdf", "year": "2025"}.
        store:   VectorStore instance (creates one if not provided).

    Returns:
        List of result dicts: {text, metadata, score}
    """
    store = store or VectorStore()

    if store.count() == 0:
        return []

    if mode == "semantic":
        return store.semantic_search(query, top_k, filters)

    elif mode == "keyword":
        # BM25 over all stored chunks (pulled from ChromaDB)
        return _bm25_search(query, top_k, filters, store)

    else:  # hybrid (default)
        sem_results = store.semantic_search(query, top_k * 2, filters)
        kw_results  = _bm25_search(query, top_k * 2, filters, store)
        return _rrf_fusion(sem_results, kw_results, top_k)


# ── BM25 search ───────────────────────────────────────────────────────────────

def _bm25_search(query: str,
                 top_k: int,
                 filters: dict[str, Any] | None,
                 store: VectorStore) -> list[dict]:
    """
    BM25 keyword ranking over all chunks retrieved from ChromaDB.

    We pull all chunks from ChromaDB (with optional metadata filter) and
    run BM25 locally. This is fine for a college corpus (hundreds of docs);
    for millions of docs you'd use a dedicated search engine like Elasticsearch.
    """
    # Pull all chunks matching the filter
    raw = store._col.get(
        include=["documents", "metadatas"],
        where=_build_where(filters) if filters else None,
    )
    docs   = raw.get("documents", [])
    metas  = raw.get("metadatas",  [])

    if not docs:
        return []

    # Tokenise
    tokenised_corpus = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenised_corpus)
    scores = bm25.get_scores(query.lower().split())

    # Sort by score, take top_k
    ranked = sorted(
        zip(docs, metas, scores),
        key=lambda x: x[2],
        reverse=True,
    )[:top_k]

    return [{"text": d, "metadata": m, "score": round(s, 4)}
            for d, m, s in ranked if s > 0]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf_fusion(sem_results: list[dict],
                kw_results:  list[dict],
                top_k: int,
                k: int = 60) -> list[dict]:
    """
    Combine two ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank))  for each list that contains the chunk.
    k=60 is the standard recommended value.
    """
    # Use chunk text as the de-duplication key
    scores: dict[str, float] = {}
    lookup: dict[str, dict]  = {}

    def _add(results: list[dict], weight: float):
        for rank, res in enumerate(results, start=1):
            key = res["text"][:120]   # first 120 chars as fingerprint
            scores[key]  = scores.get(key, 0.0) + weight * (1 / (k + rank))
            lookup[key]  = res

    _add(sem_results, HYBRID_ALPHA)
    _add(kw_results,  1 - HYBRID_ALPHA)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [lookup[key] for key, _ in ranked]


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_where(filters: dict[str, Any]) -> dict | None:
    if not filters:
        return None
    if len(filters) == 1:
        k, v = next(iter(filters.items()))
        return {k: {"$eq": str(v)}}
    return {"$and": [{k: {"$eq": str(v)}} for k, v in filters.items()]}
