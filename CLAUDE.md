# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Hybrid Agentic RAG pipeline** for the BBTC (Bethesda Bedok-Tampines Church) sermon archive. It scrapes sermon documents (PDF, PPTX, DOCX) from the BBTC website, extracts text and LLM-derived metadata, stores them in a dual-layer storage system (SQLite + ChromaDB), and exposes a Gradio chat interface backed by a LangGraph ReAct agent.

## Environment Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # set API keys: GROQ_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY
```

`.env` keys used:
- `GROQ_API_KEY` — Groq cloud inference (default LLM)
- `GEMINI_API_KEY` / `GOOGLE_API_KEY` — Gemini cloud inference (either key works; the app maps GEMINI_API_KEY → GOOGLE_API_KEY automatically)
- Ollama running locally is the fallback when no cloud keys are present

## Running the Application

```bash
# Launch the Gradio chat UI
python app.py

# Run the full ingestion pipeline manually (scrape + extract + vectorize)
dagster asset materialize --select sermon_ingestion_summary -m dagster_pipeline

# Run the Dagster web UI (to trigger/monitor the pipeline)
DAGSTER_HOME=$(mktemp -d) dagster dev -m dagster_pipeline

# One-shot: vectorize already-extracted sermons without re-scraping
python quick_ingest.py

# Scrape a single year
python src/scraper/bbtc_scraper.py 2024
```

## Architecture

### Data Flow

```
BBTC Website → BBTCScraper → data/staging/ (raw files)
                           → data/sermons/ (.txt extracts)
                           → SQLite (data/sermons.db) [status: extracted]
                               ↓ MetadataExtractor (LLM)
                           → SQLite [status: indexed, with speaker/date/verse]
                           → ChromaDB (data/chroma_db/) [vector chunks]
                               ↓
                        Gradio UI → LangGraph ReAct Agent → Tools → Response
```

### Key Components

| Component | File | Purpose |
|---|---|---|
| `SermonRegistry` | `src/storage/sqlite_store.py` | SQLite CRUD; tracks ingestion status |
| `SermonVectorStore` | `src/storage/chroma_store.py` | ChromaDB with Ollama embeddings + CrossEncoder reranker |
| `BBTCScraper` | `src/scraper/bbtc_scraper.py` | Cloudflare-bypass scraper; downloads + text-extracts PDFs/PPTX/DOCX |
| `MetadataExtractor` | `src/ingestion/metadata_extractor.py` | LLM extracts speaker/date/series/verse from first 500 chars |
| `get_llm()` | `src/llm.py` | Factory returning Groq/Gemini/Ollama LangChain chat model |
| `dagster_pipeline.py` | root | Dagster asset that orchestrates full scrape+ingest; weekly Sunday schedule |
| `app.py` | root | Gradio UI + LangGraph ReAct agent wired to three tools |

### Agent Tools (LangGraph ReAct)

- **`sql_query_tool`** — executes arbitrary SQL against `data/sermons.db`; use for counts, stats, date lookups
- **`search_sermons_tool`** — semantic search over ChromaDB `sermon_collection`; use for "what was said about X"
- **`matplotlib_tool`** — generates PNG charts from live SQLite data; supports `sermons_per_speaker`, `sermons_per_year`, `top_bible_books`

### SQLite Schema

```sql
sermons(
  sermon_id TEXT PRIMARY KEY,  -- slugified filename
  filename TEXT,
  url TEXT UNIQUE,
  speaker TEXT,
  date TEXT,                   -- YYYY-MM-DD
  series TEXT,
  bible_book TEXT,
  primary_verse TEXT,          -- e.g. "Romans 8:28"
  language TEXT,               -- "English" | "Mandarin"
  file_type TEXT,              -- pdf | pptx | docx
  year INTEGER,
  status TEXT,                 -- extracted → indexed | failed
  date_scraped TEXT
)
```

### Ingestion Statuses

- `extracted` — text pulled from document, not yet LLM-processed
- `processed` — synonym for extracted (treated equivalently in pipeline)
- `indexed` — metadata extracted + vectorized in ChromaDB
- `failed` — text extraction failed

### ChromaDB Collections

- `sermon_collection` — chunked sermon text (chunk_size=1000, overlap=100)
- `bible_collection` — reserved for Bible text (not yet populated)

Embeddings default to `nomic-embed-text` via Ollama; falls back to ChromaDB's built-in embeddings if Ollama is unavailable.

## Notable Quirks

- `MetadataExtractor` uses Groq by default and falls back to Ollama (`llama3.2:3b`) on rate-limit errors (HTTP 429).
- The `Reranker` in `src/storage/reranker.py` uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for real CrossEncoder reranking.
- `matplotlib_tool` queries live data from SQLite for three chart types: `sermons_per_speaker`, `sermons_per_year`, `top_bible_books`.
- The scraper uses `cloudscraper` to bypass Cloudflare protection on the BBTC website.
- All imports of `langchain_google_genai` happen after `GEMINI_API_KEY` is remapped to `GOOGLE_API_KEY` in `app.py`.
- Dagster creates temporary `DAGSTER_HOME` directories (`.tmp_dagster_home_*`) in the project root; these can be ignored or cleaned up.
