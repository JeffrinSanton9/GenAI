"""
ingest/chunker.py – Split document text blocks into overlapping chunks.

Why chunking matters:
  • LLMs have a context window; we can't feed a 200-page PDF at once.
  • Smaller chunks improve retrieval precision.
  • Overlap preserves context that would be lost at split boundaries.

Strategy used here: character-level sliding window with sentence-boundary
awareness (we avoid cutting mid-sentence where possible).
"""

from __future__ import annotations
import re
import uuid
from config import CHUNK_SIZE, CHUNK_OVERLAP


def _sentence_split(text: str) -> list[str]:
    """Naïve sentence splitter – split at '.', '!', '?' followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str,
               metadata: dict,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split a text block into overlapping chunks.

    Each chunk dict contains:
      {
        "chunk_id":   <uuid str>,
        "text":       <chunk text>,
        "metadata":   <copy of doc metadata + chunk_index>,
      }
    """
    sentences   = _sentence_split(text)
    chunks      = []
    buffer      = ""
    chunk_index = 0

    for sentence in sentences:
        # Would adding this sentence exceed the limit?
        if len(buffer) + len(sentence) + 1 > chunk_size and buffer:
            chunks.append(_make_chunk(buffer.strip(), metadata, chunk_index))
            chunk_index += 1
            # Keep the last `overlap` characters as context for the next chunk
            buffer = buffer[-overlap:] + " " + sentence
        else:
            buffer = (buffer + " " + sentence).strip()

    # Flush remaining text
    if buffer.strip():
        chunks.append(_make_chunk(buffer.strip(), metadata, chunk_index))

    return chunks


def _make_chunk(text: str, metadata: dict, index: int) -> dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "text":     text,
        "metadata": {**metadata, "chunk_index": index},
    }


def chunk_documents(doc_blocks: list[dict],
                    chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Chunk a list of document blocks (output of a loader).

    Args:
        doc_blocks: list of {"text": ..., "metadata": ...} dicts.
        chunk_size: max chars per chunk.
        overlap:    overlap chars between consecutive chunks.

    Returns:
        Flat list of chunk dicts ready for embedding + indexing.
    """
    all_chunks = []
    for block in doc_blocks:
        chunks = chunk_text(block["text"], block["metadata"],
                            chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks
