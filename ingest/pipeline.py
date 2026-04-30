"""
ingest/pipeline.py – End-to-end ingestion: load → chunk → embed → store.

Usage:
    from ingest.pipeline import ingest_document, ingest_folder

    # Single file with optional metadata
    ingest_document("docs/student_handbook.pdf",
                    extra_meta={"doc_type": "handbook", "year": "2025"})

    # Whole folder
    ingest_folder("docs/")
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ingest.loaders  import load_document
from ingest.chunker  import chunk_documents
from retrieval.vector_store import VectorStore


def ingest_document(path: str | Path,
                    extra_meta: dict[str, Any] | None = None,
                    store: VectorStore | None = None) -> int:
    """
    Ingest a single document into the vector store.

    Args:
        path:       Path to the document.
        extra_meta: Metadata fields to attach (doc_type, department, year, …).
        store:      VectorStore instance. Creates a new one if not provided.

    Returns:
        Number of chunks ingested.
    """
    store = store or VectorStore()
    path  = Path(path)

    print(f"📄  Loading   {path.name} …")
    blocks = load_document(path, extra_meta)

    print(f"✂️   Chunking  {len(blocks)} block(s) …")
    chunks = chunk_documents(blocks)

    print(f"🗄️   Storing   {len(chunks)} chunk(s) …")
    store.add_chunks(chunks)

    print(f"✅  Done – {len(chunks)} chunks indexed from '{path.name}'")
    return len(chunks)


def ingest_folder(folder: str | Path,
                  extra_meta: dict[str, Any] | None = None,
                  store: VectorStore | None = None,
                  recursive: bool = False) -> dict[str, int]:
    """
    Ingest all supported documents inside a folder.

    Args:
        folder:     Path to the folder.
        extra_meta: Shared metadata applied to every document.
        store:      VectorStore instance (shared across all files).
        recursive:  If True, also scan sub-folders.

    Returns:
        Dict {filename: chunk_count} for every successfully ingested file.
    """
    store  = store or VectorStore()
    folder = Path(folder)
    supported = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}

    pattern = "**/*" if recursive else "*"
    files   = [f for f in folder.glob(pattern)
               if f.is_file() and f.suffix.lower() in supported]

    if not files:
        print(f"⚠️  No supported documents found in '{folder}'")
        return {}

    results: dict[str, int] = {}
    for f in tqdm(files, desc="Ingesting documents"):
        try:
            n = ingest_document(f, extra_meta=extra_meta, store=store)
            results[f.name] = n
        except Exception as exc:
            print(f"❌  Failed to ingest '{f.name}': {exc}")
            results[f.name] = 0

    total = sum(results.values())
    print(f"\n🎉  Ingestion complete – {total} chunks from {len(files)} file(s).")
    return results


def clear_collection(store: VectorStore | None = None) -> None:
    """Delete all documents from the vector store (use with caution)."""
    store = store or VectorStore()
    store.clear()
    print("🗑️  Vector store cleared.")
