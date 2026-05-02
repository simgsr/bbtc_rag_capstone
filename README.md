# BBTC Sermon Intelligence

> **Capstone Project — NTU MSCS 2026**
> A production-grade Hybrid Agentic RAG pipeline over a decade of church sermon archives.

📚 800 sermons &nbsp;·&nbsp; 👤 35 speakers &nbsp;·&nbsp; 📅 2015 – 2026 &nbsp;·&nbsp; 🌐 1 languages

A fully local, privacy-preserving AI system that ingests, indexes, and answers natural-language questions about the BBTC (Bethesda Bedok-Tampines Church) sermon archive from 2015 to present. Built end-to-end: from PDF scraping through LLM-powered metadata extraction, dual-layer storage, and a ReAct agent that intelligently routes queries between SQL, vector search, visualisation, and Bible lookup tools.

---

## What it does

Ask questions like:

- *"How many sermons has Pastor Daniel preached on Romans?"*
- *"What did the church teach about anxiety in 2023?"*
- *"Show me a breakdown of sermons by speaker over the last 3 years."*
- *"Compare how NIV and ESV translate John 3:16."*
- *"Find Bible passages about perseverance in suffering."*

The LangGraph ReAct agent decides in real time which tool to invoke — SQL for structured facts, ChromaDB for semantic content, Plotly for charts, or the Bible archive for cross-translation verse lookup — then synthesises a coherent answer.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Gemma 4 (via Ollama, fully local) |
| **Embeddings** | BGE-M3 (1.2 GB, multilingual, via Ollama) |
| **Reranking** | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| **Vector store** | ChromaDB |
| **Structured store** | SQLite |
| **Agent framework** | LangGraph ReAct (`langgraph.prebuilt`) |
| **Pipeline scheduler** | Dagster (weekly cron) |
| **Chat UI** | Gradio |
| **Scraper** | HTTPX + BeautifulSoup (Cloudflare-bypass) |

---

## Architecture

```
BBTC Website → BBTCScraper (classify-before-download, skips handouts)
    ↓
data/staging/  (NG + PS PDF files)
    ↓
ingest.py
  ├── CLASSIFY  (file_classifier.py)  → ng | ps | handout
  ├── GROUP     (sermon_grouper.py)   → SermonGroup(ng, ps[])
  ├── EXTRACT   (ng_extractor.py)     → TOPIC/SPEAKER/THEME/DATE via regex
  │             (ps_extractor.py)     → key verse from PS filename
  ├── SUMMARIZE (gemma4:latest)       → unified NG+PS summary
  └── EMBED     (chroma_store.py)     → BGE-M3 → sermon_collection
    ↓
SQLite (data/sermons.db)              ← structured metadata + verses
ChromaDB (data/chroma_db/)            ← body chunks + summaries + bible verses
    ↓
LangGraph ReAct Agent (5 tools)
    ↓
Gradio Chat UI  →  http://localhost:7860
```

### Sermon Unit Model

Each Sunday, BBTC posts two files that form one **sermon unit**:
- **NG** (Notes/Guide): PDF with labeled `TOPIC`, `SPEAKER`, `THEME`, `DATE` fields + body text
- **PS** (Slides/PPT): PDF whose filename encodes the key verse

The pipeline pairs these by date proximity and topic overlap before ingestion.

### Agent Tools

| Tool | When the agent uses it |
|---|---|
| `sql_query_tool` | Counts, lists, date ranges, speaker stats, verse aggregations |
| `search_sermons_tool` | "What did the church teach about X?" — semantic content search |
| `viz_tool` | "Show me a chart of…" — live Plotly charts from SQLite |
| `get_bible_versions_tool` | "How do different translations render Luke 9:23?" |
| `search_bible_tool` | "Find passages about perseverance" — semantic Bible search |

### Key Components

| Component | File | Purpose |
|---|---|---|
| `SermonRegistry` | `src/storage/sqlite_store.py` | SQLite CRUD; sermons, verses, and reference tables |
| `SermonVectorStore` | `src/storage/chroma_store.py` | ChromaDB with BGE-M3 embeddings + CrossEncoder reranker |
| `BBTCScraper` | `src/scraper/bbtc_scraper.py` | Cloudflare-bypass scraper; classify-before-download |
| `classify_file` | `src/ingestion/file_classifier.py` | Returns `ng` \| `ps` \| `handout` |
| `group_sermon_files` | `src/ingestion/sermon_grouper.py` | Pairs NG+PS files by date proximity and topic overlap |
| `extract_ng_metadata` | `src/ingestion/ng_extractor.py` | Regex on labeled PDF fields; filename fallback |
| `parse_verses_from_filename` | `src/ingestion/ps_extractor.py` | Verse regex on PS filenames |
| `normalize_book` | `src/storage/normalize_book.py` | Canonical 66-book name normalization |
| `ingest_bible` | `src/ingestion/bible/bible_ingest.py` | Fetches Scrollmapper JSON + parses EPUBs → `bible_collection` |
| `BibleEpubParser` | `src/ingestion/bible/epub_parser.py` | Extracts verse-by-verse text from EPUB files |
| `run_pipeline` | `ingest.py` | Orchestrates full classify → group → extract → embed |
| `dagster_pipeline.py` | root | Weekly Saturday schedule wrapping `ingest.py` |
| `app.py` | root | Gradio UI + LangGraph ReAct agent |

---

## Local Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- Make (comes pre-installed on macOS/Linux)

```bash
ollama pull bge-m3           # embeddings — 1.2 GB, multilingual
ollama pull gemma4:latest    # LLM — metadata extraction, summarisation, chat
```

### Install and run (One-Click Setup)

We have provided a `Makefile` to handle environment creation, dependency installation, scraping, and ingestion automatically.

```bash
git clone <repo-url>
cd bbtc_rag_capstone

# Full setup (installs deps, scrapes current year, ingests data)
make setup

# Launch the Gradio chat UI
make run
```

Open [http://localhost:7860](http://localhost:7860).

*Note: Missing directories (like `data/staging`, `data/sermons.db`, or `data/chroma_db`) are created automatically during setup.*

### Bible archive (optional)

KJV, ASV, and YLT are downloaded automatically from Scrollmapper (public domain). NIV and ESV require EPUB files you supply yourself (copyrighted — not included in this repo).

```
data/bibles/NIV.epub
data/bibles/ESV The Holy Bible.epub
```

```bash
# All 5 translations (NIV/ESV skipped automatically if files are absent)
python -m src.ingestion.bible.bible_ingest

# Public-domain only
python -m src.ingestion.bible.bible_ingest --versions KJV ASV YLT
```

---

## Data Pipeline

For more granular control, you can run individual `make` commands or direct python scripts:

```bash
# Set up virtual environment and install dependencies
make install

# Scrape current year's sermons (or specify: YEAR=2024 make scrape)
make scrape

# Full sermon ingest from scratch (wipe + rebuild)
make ingest

# Dagster web UI — weekly Saturday scheduler
make dagster
```

---

## Database Schema

### SQLite (`data/sermons.db`)

```sql
sermons(
  sermon_id  TEXT PRIMARY KEY,  -- "2024-01-06-the-heart-of-discipleship"
  date       TEXT,              -- YYYY-MM-DD
  year       INTEGER,
  language   TEXT,              -- "English" | "Mandarin"
  speaker    TEXT,
  topic      TEXT,
  theme      TEXT,
  summary    TEXT,              -- LLM-generated from NG+PS
  key_verse  TEXT,              -- first verse from PS filename
  ng_file    TEXT,
  ps_file    TEXT,
  status     TEXT               -- grouped → extracted → indexed | failed
)

verses(
  id           INTEGER PRIMARY KEY,
  sermon_id    TEXT,            -- FK → sermons
  verse_ref    TEXT,            -- "Luke 9:23"
  book         TEXT,            -- "Luke"
  chapter      INTEGER,
  verse_start  INTEGER,
  verse_end    INTEGER,
  is_key_verse INTEGER          -- 1 = key verse
)

bible_books(
  book_name  TEXT PRIMARY KEY,  -- canonical name e.g. "1 Samuel"
  testament  TEXT,              -- "OT" | "NT"
  book_order INTEGER            -- 1–66
)

book_aliases(
  alias     TEXT PRIMARY KEY,   -- lowercase variant e.g. "1sam", "gen"
  canonical TEXT                -- FK → bible_books
)

bible_versions(
  version_id   TEXT PRIMARY KEY, -- "KJV", "NIV", etc.
  filename     TEXT,
  status       TEXT,             -- "indexed"
  date_indexed TEXT
)
```

### ChromaDB (`data/chroma_db/`)

**`sermon_collection`**
- Chunks: NG body text (800 tokens / 150 overlap) + LLM summary (single chunk) per sermon
- Metadata: `{sermon_id, doc_type, speaker, date, year, topic, theme, language, key_verse}`
- Embeddings: BGE-M3 via Ollama

**`bible_collection`**
- ~102,790 chunks across 5 translations (~31,000 verses each)
- Sources: KJV, ASV, YLT from Scrollmapper (public domain); NIV, ESV from local EPUBs
- Metadata: `{book, chapter, verse, version, reference}`
- Embeddings: BGE-M3 via Ollama

---

## Tests

```bash
python -m pytest tests/ -v
```

103 tests covering file classification, filename parsing, metadata extraction, verse normalization, sermon grouping, vector retrieval, UI helpers, and SQLite storage.

---

## Project Structure

```
.
├── app.py                        # Gradio UI + LangGraph agent
├── ingest.py                     # Sermon ingestion pipeline
├── dagster_pipeline.py           # Weekly Dagster schedule
├── requirements.txt
├── .env.example
├── src/
│   ├── ingestion/
│   │   ├── bible/
│   │   │   ├── bible_ingest.py   # Bible translation ingestion
│   │   │   └── epub_parser.py    # EPUB verse extractor
│   │   ├── file_classifier.py    # ng | ps | handout classifier
│   │   ├── filename_parser.py    # Fallback metadata from filename
│   │   ├── ng_extractor.py       # Regex metadata from NG PDFs
│   │   ├── ps_extractor.py       # Verse extraction from PS filenames
│   │   └── sermon_grouper.py     # Pairs NG+PS by date/topic
│   ├── scraper/
│   │   └── bbtc_scraper.py       # Cloudflare-bypass scraper
│   ├── storage/
│   │   ├── chroma_store.py       # ChromaDB + BGE-M3 + reranker
│   │   ├── normalize_book.py     # Canonical 66-book name normalization
│   │   ├── normalize_speaker.py  # Speaker name normalization
│   │   ├── reranker.py           # CrossEncoder reranking
│   │   └── sqlite_store.py       # SQLite CRUD
│   ├── tools/
│   │   ├── bible_tool.py         # Bible verse + search tools
│   │   ├── sql_tool.py           # SQL query tool
│   │   ├── vector_tool.py        # Sermon semantic search tool
│   │   └── viz_tool.py           # Plotly chart tool
│   ├── llm.py                    # Unified LLM client (Ollama / Groq / Gemini)
│   └── ui_helpers.py             # Gradio rendering helpers
├── tests/                        # 103 unit tests
├── scripts/
│   └── normalize_books.py        # One-time book-name migration utility
└── docs/
    ├── design/                   # Architecture and feature design notes
    └── plans/                    # Implementation plans
```

---

## Notable Design Decisions

- **Classify-before-download**: The scraper classifies filenames against a regex before downloading, so handout PDFs are never fetched.
- **~50% image-based PDFs**: Many PS slide files have no extractable text — verse extraction relies entirely on filename regex parsing.
- **Fully local by default**: Ollama handles both embeddings and LLM inference. Groq/Gemini are optional cloud fallbacks configured via `.env`.
- **NG labeled fields are reliable from 2022+**: Pre-2022 files fall back to `filename_parser.py` heuristics.
- **CrossEncoder reranking**: Top-20 BGE-M3 candidates are reranked by a cross-encoder before returning to the agent, improving precision without sacrificing recall.
