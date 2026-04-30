"""
retrieval/embedder.py – Compute dense vector embeddings for text.

Uses sentence-transformers locally (no API key required).
The same model must be used for both indexing and querying.

Why sentence-transformers?
  • Free, open-source, runs on CPU.
  • all-MiniLM-L6-v2 is only ~80 MB but scores well on retrieval benchmarks.
  • Consistent: same library → same vector space for docs and queries.
"""

from __future__ import annotations
from functools import lru_cache

from config import EMBED_MODEL


@lru_cache(maxsize=1)
def _get_model():
    """Load the embedding model once and cache it (lazy load)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed.\n"
            "Run:  pip install sentence-transformers"
        )
    print(f"🔤  Loading embedding model '{EMBED_MODEL}' …")
    return SentenceTransformer(EMBED_MODEL)


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Embed a list of strings → list of float vectors.

    Args:
        texts:      Strings to embed (chunks or queries).
        batch_size: How many to embed per batch (reduces memory pressure).

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,   # unit-norm → cosine ≡ dot product
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string. Convenience wrapper."""
    return embed_texts([query])[0]
