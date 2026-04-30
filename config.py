"""
config.py – Central configuration for the College RAG system.
Edit these values to tune behaviour without touching the core code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DOCS_DIR      = BASE_DIR / "sample_docs"          # Default documents folder
CHROMA_DIR    = BASE_DIR / ".chromadb"            # Persistent ChromaDB storage
COLLECTION    = "college_knowledge"               # ChromaDB collection name

# ─── LLM ─────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL         = "claude-sonnet-4-20250514"
MAX_TOKENS        = 1024

# ─── Embeddings ──────────────────────────────────────────────────────────────
# Free, runs locally – no API key needed.
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality

# ─── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # Characters per chunk
CHUNK_OVERLAP = 80    # Overlap between consecutive chunks

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K          = 5    # Number of chunks to retrieve
HYBRID_ALPHA   = 0.5  # 0 = pure BM25, 1 = pure semantic; 0.5 = balanced

# ─── Metadata keys (used in ChromaDB filters) ────────────────────────────────
# Supported filter keys when ingesting documents:
#   source, doc_type, department, year, version, restricted
META_KEYS = ["source", "doc_type", "department", "year", "version", "restricted"]

# ─── System prompt for the RAG generator ─────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful Knowledge Assistant for college students.
You ONLY answer questions using the retrieved document passages provided below.
If the answer is not present in the passages, say:
  "I could not find this information in the official documents. Please check with the admin office."
Always cite your sources using [Source: <document name>] at the end of each fact.
Be concise, friendly, and student-focused."""
