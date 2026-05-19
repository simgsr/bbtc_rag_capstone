# BBTC Sermon Intelligence

> **Capstone Project — NTU DSAI 2026**
> A production-grade Hybrid Agentic RAG pipeline over a decade of church sermon archives.

800+ sermons &nbsp;·&nbsp; 35 speakers &nbsp;·&nbsp; 2015 – 2026 &nbsp;·&nbsp; English + Mandarin

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
| **Chat LLM** | Gemma 4 / Qwen 3 (via Ollama, fully local) |
| **Ingest LLM** | Apple MLX (`Qwen3-4B-4bit`) or Ollama — configurable via `INGEST_PROVIDER` |
| **Embeddings** | BGE-M3 (1.2 GB, multilingual, via Ollama) |
| **Reranking** | CrossEncoder (`ms-marco-MiniLM-L-6-v2`, via sentence-transformers) |
| **Vector store** | ChromaDB |
| **Structured store** | SQLite |
| **Agent framework** | LangGraph ReAct (`langgraph.prebuilt`) |
| **Pipeline scheduler** | Dagster (weekly cron) |
| **Chat UI** | Gradio |
| **Scraper** | cloudscraper + BeautifulSoup (Cloudflare-bypass) |

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
  ├── SUMMARIZE (MLX or Ollama LLM)  → unified NG+PS summary
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
- [Ollama](https://ollama.ai) running locally (`ollama serve`)
- Make (pre-installed on macOS/Linux)

Pull the required Ollama models before running setup:

```bash
ollama pull bge-m3           # embeddings — 1.2 GB, multilingual (always required)
ollama pull gemma4:latest    # chat LLM — ~9.6 GB (skip if using a smaller model)
```

> **Hardware guide** — see `.env.example` for model recommendations based on available RAM (8 GB → 96 GB+).

**Apple Silicon (M-series) — faster ingest via MLX**

On Apple Silicon you can use [mlx-lm](https://github.com/ml-explore/mlx-lm) instead of Ollama for ingest LLM inference. MLX runs natively on the Neural Engine and is significantly faster for summarisation and verse extraction.

```bash
pip install mlx-lm
```

Then set in `.env`:
```
INGEST_PROVIDER=mlx
MLX_INGEST_MODEL=mlx-community/Qwen3-4B-4bit   # downloaded automatically on first run
```

The MLX model (~2.5 GB) is downloaded from HuggingFace on first use and cached locally. Ollama is still required for BGE-M3 embeddings and the chat agent.

### One-click setup (fresh clone)

```bash
git clone <repo-url>
cd bbtc_rag_capstone
make setup   # installs deps + scrapes all years (2015–present) + ingests
make run     # launches Gradio UI at http://localhost:7860
```

`make setup` runs three steps automatically:
1. Creates a `.venv` and installs `requirements.txt`
2. Copies `.env.example` → `.env` if no `.env` exists
3. Scrapes **all sermon years 2015–present** from the BBTC website
4. Wipes any existing data and rebuilds SQLite + ChromaDB from scratch

> Scraping all years takes ~10–20 minutes. Ingestion takes 2–5 hours with Ollama or ~30–60 minutes with MLX on Apple Silicon (~800 sermons).

---

## Makefile Reference

| Command | What it does |
|---|---|
| `make setup` | Full setup: install deps + scrape all years + ingest |
| `make install` | Create `.venv` and install dependencies only |
| `make scrape` | Scrape current year (override with `YEAR=2024 make scrape`) |
| `make ingest` | Incremental ingest of any new files in `data/staging/` |
| `make run` | Launch Gradio chat UI at http://localhost:7860 |
| `make dagster` | Open Dagster web UI for the weekly scheduler |
| `make test` | Run pytest suite |
| `make clean` | Delete `.venv`, `data/chroma_db/`, `data/sermons.db`, `data/staging/` |

### Full rebuild from scratch

```bash
make clean && make setup
```

### Incremental update (add a new year)

```bash
YEAR=2025 make scrape   # download 2025 files into staging
make ingest             # pick up only the new files
```

---

## Bible Archive (optional)

KJV, ASV, and YLT are downloaded automatically from Scrollmapper (public domain). NIV and ESV require EPUB files you supply yourself (copyrighted — not included in this repo).

Place them at:
```
data/bibles/NIV.epub
data/bibles/ESV The Holy Bible.epub
```

Then ingest:
```bash
# All 5 translations (NIV/ESV skipped automatically if files are absent)
python -m src.ingestion.bible.bible_ingest

# Public-domain only
python -m src.ingestion.bible.bible_ingest --versions KJV ASV YLT

# Wipe and re-ingest bible_collection
python -m src.ingestion.bible.bible_ingest --wipe
```

---

## Environment Variables

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_CHAT_MODEL` | `gemma4:latest` | LLM for the Gradio agent |
| `OLLAMA_INGEST_MODEL` | `gemma4:latest` | LLM for ingest when `INGEST_PROVIDER=ollama_local` |
| `INGEST_PROVIDER` | `ollama_local` | Ingest LLM backend: `ollama_local` \| `mlx` \| `groq` \| `gemini` |
| `MLX_INGEST_MODEL` | `mlx-community/Qwen3-4B-4bit` | MLX model for ingest (Apple Silicon only) |
| `GROQ_API_KEY` | *(empty)* | Optional Groq cloud inference |
| `GOOGLE_API_KEY` | *(empty)* | Optional Gemini cloud inference |

---

## Weekly Scheduler (Dagster)

```bash
make dagster   # opens http://localhost:3000
```

The Dagster pipeline runs three assets on a weekly Saturday 22:00 schedule:

1. **`sermon_scraping`** — scrapes current month's sermons
2. **`sermon_ingestion`** — incremental ingest of new files
3. **`bible_ingestion`** — checks for new EPUB files in `data/bibles/`

Dagster state is stored in `.dagster/` (committed config, gitignored runtime data). The heartbeat timeout is set to 30 minutes in `.dagster/dagster.yaml` so long-running LLM ingestion jobs don't cause the code server to shut down prematurely.

To trigger a manual run or configure `all_years: true` for a full backfill, use the Dagster UI's **Launchpad** and set the asset config:

```json
{"ops": {"sermon_scraping": {"config": {"all_years": true}}}}
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
make test
# or: python -m pytest tests/ -v
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
├── Makefile
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
│   ├── llm.py                    # Unified LLM client (MLX / Ollama / Groq / Gemini)
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
- **Fully local by default**: Ollama handles embeddings and chat LLM inference. For ingest, Apple Silicon users can switch to MLX (`INGEST_PROVIDER=mlx`) for significantly faster summarisation without any server overhead. Groq/Gemini are optional cloud fallbacks.
- **NG labeled fields are reliable from 2022+**: Pre-2022 files fall back to `filename_parser.py` heuristics.
- **CrossEncoder reranking**: Top-20 BGE-M3 candidates are reranked by a cross-encoder before returning to the agent, improving precision without sacrificing recall.
- **Manifest-based pairing**: The scraper writes `_manifest_*.json` files that record which PDFs came from the same sermon page. The grouper reads these first for exact pairing, then falls back to fuzzy date/topic matching.
