# 🎓 College Student Knowledge Assistant (RAG)

A **Retrieval-Augmented Generation (RAG)** system that helps students find accurate, document-grounded answers about college policies, exams, placements, scholarships, and more.

---

## 📐 Architecture

```
                     ┌──────────────────────────────────────────────┐
                     │               RAG PIPELINE                   │
                     │                                              │
  Student Question   │  ┌─────────┐   ┌──────────┐   ┌──────────┐ │
  ─────────────────► │  │ Search  │──►│ Augment  │──►│ Generate │ │──► Answer + Citations
                     │  │(hybrid) │   │(context) │   │ (Claude) │ │
                     │  └────┬────┘   └──────────┘   └──────────┘ │
                     │       │                                      │
                     └───────┼──────────────────────────────────────┘
                             │
               ┌─────────────▼─────────────┐
               │      ChromaDB             │
               │   (Vector Store)          │
               │   chunks + embeddings     │
               └─────────────┬─────────────┘
                             │
               ┌─────────────▼─────────────┐
               │     Ingestion Pipeline    │
               │ Load → Chunk → Embed →    │
               │         Store             │
               └─────────────┬─────────────┘
                             │
               ┌─────────────▼─────────────┐
               │    College Documents      │
               │ PDF / DOCX / TXT / HTML   │
               └───────────────────────────┘
```

---

## 📁 Project Structure

```
college_rag/
├── config.py                      # Central configuration
├── requirements.txt               # Python dependencies
├── app.py                         # Streamlit web UI
├── cli.py                         # Command-line interface
│
├── ingest/
│   ├── loaders.py                 # PDF, DOCX, TXT, HTML document loaders
│   ├── chunker.py                 # Sliding-window text chunker
│   └── pipeline.py                # Orchestrates ingest_document / ingest_folder
│
├── retrieval/
│   ├── embedder.py                # sentence-transformers embedding model
│   ├── vector_store.py            # ChromaDB wrapper (CRUD + search)
│   └── search.py                  # Keyword, semantic, and hybrid search (RRF)
│
├── generation/
│   └── rag.py                     # Retrieve → Augment → Generate pipeline
│
└── sample_docs/
    ├── student_handbook.txt        # Attendance, exams, discipline, library
    ├── placement_policy.txt        # Eligibility, internships, TPC rules
    ├── academic_calendar_scholarships.txt  # Dates, fees, scholarship schemes
    └── course_syllabi.txt          # DS, OS, DBMS course details
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd college_rag
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or create a `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Ingest sample documents

```bash
python cli.py ingest                    # Load all sample_docs/
python cli.py ingest path/to/my.pdf     # Load a specific file
```

### 4. Ask a question

```bash
python cli.py ask "What is the minimum attendance required?"
python cli.py ask "Am I eligible for placements with 1 backlog?"
python cli.py ask "Which scholarships are available for 2nd year students?"
```

### 5. Multi-turn chat

```bash
python cli.py chat
```

### 6. Launch the Web UI

```bash
streamlit run app.py
```

---

## ⚙️ Configuration (`config.py`)

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model (local, free) |
| `CHUNK_SIZE` | `400` | Max characters per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap chars between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `HYBRID_ALPHA` | `0.5` | 0=pure BM25, 1=pure semantic |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Anthropic model |
| `CHROMA_DIR` | `.chromadb/` | Persistent vector DB path |

---

## 🔍 Search Modes

| Mode | Best For | How It Works |
|---|---|---|
| `hybrid` (default) | Most queries | RRF fusion of semantic + BM25 |
| `semantic` | Paraphrase queries | Dense vector cosine similarity |
| `keyword` | Exact terms (IDs, codes) | BM25 over all stored chunks |

---

## 🏷️ Metadata Filters

When ingesting, you can tag documents with metadata:

```bash
python cli.py ingest scholarship.pdf --doc-type scholarship --year 2025
```

Then filter at query time:

```bash
python cli.py ask "What scholarships exist?" --doc-type scholarship --year 2025
```

Supported filter keys: `source`, `doc_type`, `department`, `year`, `version`, `restricted`

---

## 🧩 Extending the System

### Add a new document loader

Edit `ingest/loaders.py` and add to `_LOADERS`:
```python
def load_csv(path, extra_meta=None): ...
_LOADERS[".csv"] = load_csv
```

### Swap the embedding model

Change `EMBED_MODEL` in `config.py`. Popular options:
- `all-MiniLM-L6-v2` (80 MB, fast, good) ← default
- `all-mpnet-base-v2` (420 MB, better quality)
- `paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

### Scale to Pinecone / Qdrant

Replace `VectorStore` in `retrieval/vector_store.py` with a Pinecone or Qdrant client. The interface (`add_chunks`, `semantic_search`, `count`) stays the same.

---

## 📊 Syllabus Coverage

| Topic | Where in Code |
|---|---|
| Information Retrieval | `retrieval/search.py` – keyword, semantic, hybrid |
| Corpus & Indexing | `ingest/pipeline.py`, `ingest/chunker.py` |
| Embeddings | `retrieval/embedder.py` – sentence-transformers |
| Vector DB | `retrieval/vector_store.py` – ChromaDB |
| RAG Pipeline | `generation/rag.py` – retrieve → augment → generate |
| Evaluation (precision/recall) | Top-K retrieval, source citation in results |

---

## 🔐 Hallucination Prevention

The system prompt instructs the LLM:
> *"ONLY answer using the retrieved document passages. If the answer is not present, say so."*

Every answer includes **citations** (source filename + snippet) so students can verify.

---

## 📝 Example Queries

```
"What is the minimum attendance required to sit for the semester exam?"
"If I miss an internal test, can I request a re-test?"
"Which scholarships are available for 2nd-year students?"
"What is the late fee if I miss the semester fee deadline?"
"Am I eligible for placements if I have 1 backlog?"
"What documents are required for internship approval?"
"What are the prerequisites for Data Structures?"
"Give me the unit-wise syllabus for Operating Systems."
"What are library fine rules for late book return?"
```
