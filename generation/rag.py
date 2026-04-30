"""
generation/rag.py – Retrieval-Augmented Generation pipeline.

Flow:
  1. Retrieve: find the top-k relevant chunks for the user query.
  2. Augment:  build a context string from the retrieved passages.
  3. Generate: call the LLM with [system prompt + context + user question].
  4. Cite:     every answer references the source document(s).

The LLM is instructed to ONLY use the retrieved context, which minimises
hallucinations – a key requirement for a trusted academic knowledge base.
"""

from __future__ import annotations
from typing import Any, Generator

import anthropic

from config import ANTHROPIC_API_KEY, LLM_MODEL, MAX_TOKENS, SYSTEM_PROMPT, TOP_K
from retrieval.search import search
from retrieval.vector_store import VectorStore


# ── Main RAG function ─────────────────────────────────────────────────────────

def ask(query: str,
        mode: str = "hybrid",
        top_k: int = TOP_K,
        filters: dict[str, Any] | None = None,
        store: VectorStore | None = None,
        stream: bool = False) -> dict:
    """
    Full RAG pipeline: retrieve → augment → generate.

    Args:
        query:   The student's question.
        mode:    Search mode – "hybrid" | "semantic" | "keyword".
        top_k:   Number of chunks to retrieve.
        filters: Metadata filter (e.g. {"doc_type": "handbook"}).
        store:   VectorStore instance (created if not provided).
        stream:  If True, returns a streaming generator in result['stream'].

    Returns:
        {
          "answer":   <generated answer string>,
          "sources":  [{"source": ..., "snippet": ..., "score": ...}, ...],
          "chunks":   <raw retrieved chunks>,
          "stream":   <generator | None>,
        }
    """
    store = store or VectorStore()

    # ── Step 1: Retrieve ──────────────────────────────────────────────────────
    chunks = search(query, mode=mode, top_k=top_k, filters=filters, store=store)

    if not chunks:
        return {
            "answer":  ("I could not find any relevant documents. "
                        "Please make sure documents have been ingested first."),
            "sources": [],
            "chunks":  [],
            "stream":  None,
        }

    # ── Step 2: Augment (build context) ──────────────────────────────────────
    context, sources = _build_context(chunks)

    # ── Step 3: Generate ─────────────────────────────────────────────────────
    user_message = (
        f"Context passages from official college documents:\n\n"
        f"{context}\n\n"
        f"---\n"
        f"Student question: {query}"
    )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if stream:
        def _stream_gen() -> Generator[str, None, None]:
            with client.messages.stream(
                model      = LLM_MODEL,
                max_tokens = MAX_TOKENS,
                system     = SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": user_message}],
            ) as s:
                for text in s.text_stream:
                    yield text

        return {"answer": None, "sources": sources, "chunks": chunks,
                "stream": _stream_gen()}

    response = client.messages.create(
        model      = LLM_MODEL,
        max_tokens = MAX_TOKENS,
        system     = SYSTEM_PROMPT,
        messages   = [{"role": "user", "content": user_message}],
    )
    answer = response.content[0].text

    return {
        "answer":  answer,
        "sources": sources,
        "chunks":  chunks,
        "stream":  None,
    }


# ── Multi-turn conversation ───────────────────────────────────────────────────

def ask_with_history(query: str,
                     history: list[dict],
                     **kwargs) -> dict:
    """
    RAG with conversation history for multi-turn chat.

    Args:
        query:   Latest student question.
        history: List of {"role": "user"|"assistant", "content": ...} dicts.
        **kwargs: Passed to ask().

    Returns:
        Same dict as ask(), with updated history appended.
    """
    store   = kwargs.pop("store", None) or VectorStore()
    chunks  = search(query,
                     mode    = kwargs.get("mode",    "hybrid"),
                     top_k   = kwargs.get("top_k",   TOP_K),
                     filters = kwargs.get("filters", None),
                     store   = store)

    context, sources = _build_context(chunks)

    system = SYSTEM_PROMPT + (
        "\n\nYou are in a multi-turn conversation. Use chat history for context "
        "but ALWAYS ground answers in the provided document passages."
    )

    messages = list(history) + [{
        "role": "user",
        "content": (
            f"Context passages:\n\n{context}\n\n---\n"
            f"Student question: {query}"
        ),
    }]

    client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model      = LLM_MODEL,
        max_tokens = MAX_TOKENS,
        system     = system,
        messages   = messages,
    )
    answer = response.content[0].text

    updated_history = list(history) + [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": answer},
    ]

    return {
        "answer":  answer,
        "sources": sources,
        "chunks":  chunks,
        "history": updated_history,
        "stream":  None,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """Format retrieved chunks into a numbered context block + source list."""
    context_parts = []
    sources       = []

    for i, chunk in enumerate(chunks, start=1):
        meta    = chunk.get("metadata", {})
        source  = meta.get("source", "Unknown")
        snippet = chunk["text"][:300].replace("\n", " ")

        context_parts.append(
            f"[Passage {i}] (Source: {source})\n{chunk['text']}"
        )
        sources.append({
            "source":  source,
            "snippet": snippet,
            "score":   chunk.get("score", 0.0),
            "meta":    meta,
        })

    return "\n\n".join(context_parts), sources
