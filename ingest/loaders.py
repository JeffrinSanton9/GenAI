"""
ingest/loaders.py – Load raw text + metadata from different file types.

Each loader returns a list of dicts:
  {
    "text": <str>,        # raw extracted text
    "metadata": {         # document-level metadata
        "source":     <filename>,
        "doc_type":   <pdf|docx|txt|html>,
        "page":       <int | None>,
        ...any extra keys passed in
    }
  }
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any


# ── helpers ──────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Remove excess whitespace / control chars while keeping paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)       # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)    # max 2 consecutive newlines
    return text.strip()


def _base_meta(path: Path, extra: dict[str, Any]) -> dict[str, Any]:
    return {"source": path.name, **extra}


# ── PDF loader ────────────────────────────────────────────────────────────────

def load_pdf(path: Path, extra_meta: dict[str, Any] | None = None) -> list[dict]:
    """Extract text page-by-page using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Install pdfplumber:  pip install pdfplumber")

    extra_meta = extra_meta or {}
    results = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = _clean(text)
            if not text:
                continue
            results.append({
                "text": text,
                "metadata": {**_base_meta(path, extra_meta),
                             "doc_type": "pdf", "page": i},
            })
    return results


# ── DOCX loader ───────────────────────────────────────────────────────────────

def load_docx(path: Path, extra_meta: dict[str, Any] | None = None) -> list[dict]:
    """Extract text paragraph-by-paragraph from a .docx file."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Install python-docx:  pip install python-docx")

    extra_meta = extra_meta or {}
    doc = Document(path)
    blocks = []
    current_heading = None

    for para in doc.paragraphs:
        style = para.style.name.lower()
        text  = para.text.strip()
        if not text:
            continue
        if "heading" in style:
            current_heading = text
        else:
            blocks.append({
                "text": text,
                "metadata": {**_base_meta(path, extra_meta),
                             "doc_type": "docx",
                             "section": current_heading or "General"},
            })
    return blocks


# ── TXT loader ────────────────────────────────────────────────────────────────

def load_txt(path: Path, extra_meta: dict[str, Any] | None = None) -> list[dict]:
    """Load a plain-text file as one block (chunker will split it later)."""
    extra_meta = extra_meta or {}
    text = _clean(path.read_text(encoding="utf-8", errors="replace"))
    return [{
        "text": text,
        "metadata": {**_base_meta(path, extra_meta), "doc_type": "txt"},
    }]


# ── HTML loader ───────────────────────────────────────────────────────────────

def load_html(path: Path, extra_meta: dict[str, Any] | None = None) -> list[dict]:
    """Strip HTML tags and return visible text."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install beautifulsoup4:  pip install beautifulsoup4 lxml")

    extra_meta = extra_meta or {}
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = _clean(soup.get_text(separator="\n"))
    return [{
        "text": text,
        "metadata": {**_base_meta(path, extra_meta), "doc_type": "html"},
    }]


# ── Router ────────────────────────────────────────────────────────────────────

_LOADERS = {
    ".pdf":  load_pdf,
    ".docx": load_docx,
    ".txt":  load_txt,
    ".md":   load_txt,       # treat markdown as plain text
    ".html": load_html,
    ".htm":  load_html,
}


def load_document(path: str | Path,
                  extra_meta: dict[str, Any] | None = None) -> list[dict]:
    """
    Automatically pick the right loader based on file extension.

    Args:
        path:       Path to the document file.
        extra_meta: Optional dict of metadata to merge (e.g. doc_type, year).

    Returns:
        List of page/block dicts with 'text' and 'metadata' keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}. "
                         f"Supported: {list(_LOADERS)}")
    return loader(path, extra_meta)
