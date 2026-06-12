# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Hybrid Agentic RAG pipeline** for the BBTC (Bethesda Bedok-Tampines Church) sermon archive.

Scrapes sermon documents from the BBTC website, groups them into **sermon units** (one Notes/Guide + one Slides/PPT per Sunday), extracts structured metadata, stores in SQLite + ChromaDB, and exposes a Gradio chat interface backed by a LangGraph ReAct agent.

## Environment Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # optional: GROQ_API_KEY, GEMINI_API_KEY for cloud fallback
```

Ollama must be running locally: `ollama serve`

Required Ollama models:
- Chat LLM only — configurable via `OLLAMA_CHAT_MODEL` in `.env`:
  - default: `gemma4:latest` (9.6 GB); high-spec 96 GB+ RAM: `qwen3.5:122b`; 32 GB: `gemma4:31b`
  - Ollama context window: `OLLAMA_NUM_CTX` (default `32768`) — Ollama's own default is 2048, too small for ReAct + 3-exchange history.

**Chat LLM (Gradio agent)** — picked at runtime via the "Inference Engine" radio. Four backends:
- `mlx` (default, Apple Silicon) — uses `MLX_CHAT_MODEL` (default `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`). Boots `mlx_lm.server` as a subprocess on `MLX_SERVER_HOST:MLX_SERVER_PORT` (default `127.0.0.1:8081`) exposing an OpenAI-compatible API; LangChain connects via `ChatOpenAI`. KV prompt cache: `MLX_PROMPT_CACHE_SLOTS` (default `4`) and `MLX_PROMPT_CACHE_BYTES` (default `8_000_000_000`). Pre-warmed at app startup.
- `ollama_local` — uses `OLLAMA_CHAT_MODEL`. `ChatOllama` is constructed with `timeout=600` (raised from 120s) so large models like `qwen3.5:122b` / `gpt-oss:120b` don't time out on long generations.
- `groq` / `gemini` — set `GROQ_API_KEY` / `GOOGLE_API_KEY`

**Ingest LLM** — controlled by `INGEST_PROVIDER` in `.env`:
- `mlx` (default, Apple Silicon): uses `MLX_INGEST_MODEL` (default `mlx-community/Qwen3-4B-4bit`); model auto-downloads from HuggingFace on first run
- `ollama_local`: uses `OLLAMA_INGEST_MODEL` (default `gemma4:latest`); causes Ollama model-swap with BGE-M3 — slower
- `groq` / `gemini`: cloud fallbacks — set `GROQ_API_KEY` or `GOOGLE_API_KEY`

**Embeddings** — BGE-M3 via `sentence-transformers` on MPS (Apple Silicon GPU). No Ollama required for embeddings. Model auto-downloads from HuggingFace on first run.

## Running the Application

```bash
# Launch Gradio chat UI
python app.py

# Full ingest from scratch (wipe + rebuild)
python ingest.py --wipe

# Incremental ingest (new files only)
python ingest.py

# Ingest a specific year
python ingest.py --year 2024

# Dagster web UI (weekly scheduler)
DAGSTER_HOME=$(pwd)/.dagster dagster dev -m dagster_pipeline

# Scrape a single year from BBTC website
python src/scraper/bbtc_scraper.py 2024

# Ingest all Bible translations (KJV, ASV, YLT, NIV, ESV)
python -m src.ingestion.bible.bible_ingest

# Wipe and re-ingest bible_collection
python -m src.ingestion.bible.bible_ingest --wipe

# Ingest specific translations only
python -m src.ingestion.bible.bible_ingest --versions KJV WEB NIV
```

## Architecture

### Sermon Unit Model

Every weekend (Sat/Sun), BBTC posts two files:
- **NG** (Notes/Guide): PDF with labeled fields `TOPIC`, `SPEAKER`, `THEME`, `DATE` + body text
- **PS** (Slides/PPT): PDF exported from PowerPoint; filename encodes the key verse

Together they form one **sermon unit** — the atomic unit of ingestion.

### Data Flow

```
BBTC Website → BBTCScraper (classify-before-download: skip handouts)
    ↓
data/staging/  (NG + PS files only)
    ↓
ingest.py
  ├── CLASSIFY  (file_classifier.py)  → ng | ps | handout
  ├── GROUP     (sermon_grouper.py)   → SermonGroup(ng, ps[])
  ├── EXTRACT   (ng_extractor.py)     → TOPIC/SPEAKER/THEME/DATE via regex
  │             (ps_extractor.py)     → verses from filename + LLM on text
  ├── SUMMARIZE (MLX or Ollama LLM)   → unified NG+PS summary
  └── EMBED     (chroma_store.py)     → BGE-M3 → sermon_collection
    ↓
SQLite (data/sermons.db)  ← structured metadata + verses table
ChromaDB (data/chroma_db/) ← body chunks (800/150) + summary chunk per sermon
    ↓
LangGraph ReAct Agent (3 tools)
    ↓
Gradio UI
```

### Key Components

| Component | File | Purpose |
|---|---|---|
| `SermonRegistry` | `src/storage/sqlite_store.py` | SQLite CRUD; sermons + verses tables |
| `SermonVectorStore` | `src/storage/chroma_store.py` | ChromaDB with lazy-initialized BGE-M3 embeddings |
| `BBTCScraper` | `src/scraper/bbtc_scraper.py` | Cloudflare-bypass scraper; classify-before-download |
| `classify_file` | `src/ingestion/file_classifier.py` | Returns `ng` \| `ps` \| `handout` |
| `group_sermon_files` | `src/ingestion/sermon_grouper.py` | Pairs NG+PS by date proximity/topic overlap |
| `extract_ng_metadata` | `src/ingestion/ng_extractor.py` | Regex on labeled fields; filename fallback |
| `parse_verses_from_filename` | `src/ingestion/ps_extractor.py` | Verse regex on PS filenames |
| `normalize_book` | `src/storage/normalize_book.py` | Canonical 66-book name normalization |
| `ingest_bible` | `src/ingestion/bible/bible_ingest.py` | Fetches Scrollmapper JSON + parses EPUBs → `bible_collection` |
| `BibleEpubParser` | `src/ingestion/bible/epub_parser.py` | Extracts verse-by-verse text from NIV/ESV EPUB files |
| `make_bible_tool` | `src/tools/bible_tool.py` | `get_bible_versions_tool` + `search_bible_tool` for the agent |
| `MLXChatModel` / `get_ingest_llm` | `src/llm.py` | MLX-backed LangChain chat model (text-only, used for ingest); `get_ingest_llm()` reads `INGEST_PROVIDER` to select backend |
| `get_chat_llm` / `_ensure_mlx_server` | `src/llm.py` | Chat-agent LLM factory; for `provider="mlx"` lazily spawns `mlx_lm.server` and returns `ChatOpenAI` (native tool-calling). Cleanup via `atexit` + `SIGTERM`/`SIGINT`/`SIGHUP` registered at module load (main thread). `mlx_lm.server` stdout/stderr stream to the parent terminal at `INFO` level for debugging |
| `_ensure_ollama` / `_shutdown_ollama` | `app.py` | Tracks `ollama serve` subprocess if we spawned it; cleanup only fires for self-spawned daemons (never touches a pre-existing system `ollama serve`). Same signal coverage as MLX |
| `run_pipeline` | `ingest.py` | Orchestrates full classify→group→extract→embed |
| `dagster_pipeline.py` | root | Weekly Saturday schedule wrapping `ingest.py` |
| `app.py` | root | Gradio UI + LangGraph ReAct agent |

### Agent Tools

- **`sql_query_tool`** — SQL against `data/sermons.db`; use for counts, lists, verse aggregations
- **`search_sermons_tool`** — BGE-M3 semantic search over `sermon_collection`; use for content queries
- **`viz_tool`** — Plotly interactive charts: `sermons_per_speaker`, `sermons_per_year`, `verses_per_book`, `sermons_scatter`; accepts optional `top_n: int` (default 15) to control how many results are shown in ranked charts (`sermons_per_speaker`, `verses_per_book`)
- **`get_bible_versions_tool`** — Returns all stored translations (NIV, ESV, KJV, ASV, YLT) of a specific verse from `bible_collection`; use for Translation Audit / version comparison
- **`search_bible_tool`** — BGE-M3 semantic search over `bible_collection`; use for "find passages about [topic]" queries

### SQLite Schema

```sql
sermons(
  sermon_id TEXT PRIMARY KEY,  -- "2024-01-06-the-heart-of-discipleship"
  date      TEXT,              -- YYYY-MM-DD
  year      INTEGER,
  language  TEXT,              -- "English" | "Mandarin"
  speaker   TEXT,
  topic     TEXT,
  theme     TEXT,
  summary   TEXT,              -- LLM-generated from NG+PS
  key_verse TEXT,              -- first verse from PS
  ng_file   TEXT,              -- staging filename of NG
  ps_file   TEXT,              -- staging filename of PS (nullable)
  status    TEXT               -- grouped → extracted → indexed | failed
)

verses(
  id          INTEGER PRIMARY KEY,
  sermon_id   TEXT,            -- FK → sermons
  verse_ref   TEXT,            -- "Luke 9:23"
  book        TEXT,            -- "Luke"
  chapter     INTEGER,
  verse_start INTEGER,
  verse_end   INTEGER,
  is_key_verse INTEGER         -- 1 = key verse (first in PS)
)

bible_versions(
  version_id   TEXT PRIMARY KEY,  -- "KJV", "ASV", "YLT", "NIV", "ESV"
  filename     TEXT,              -- scrollmapper/{id}.json or data/bibles/*.epub
  status       TEXT,              -- "indexed"
  date_indexed TEXT               -- ISO timestamp
)
```

### ChromaDB

**`sermon_collection`**
- Chunks: NG body text (800/150) + LLM summary (single chunk) per sermon
- Metadata per chunk: `{sermon_id, doc_type, speaker, date, year, topic, theme, language, key_verse}`
- Embeddings: `BGE-M3` via sentence-transformers on MPS

**`bible_collection`**
- ~102,790 chunks (5 translations × ~31,000 verses each)
- Sources: KJV, ASV, YLT from Scrollmapper JSON (public domain); NIV, ESV from local EPUB files
- Metadata per chunk: `{book, chapter, verse, version, reference}`
- IDs: `{VERSION}_{Book} {chapter}:{verse}` (e.g. `NIV_John 3:16`)
- Embeddings: `BGE-M3` via sentence-transformers on MPS

## Notable Quirks

- NG labeled fields (`TOPIC`, `SPEAKER`, etc.) are reliable for 2022+ files. Older files fall back to `filename_parser.py`.
- Some older BBTC pages (pre-2020) only posted one file (slides, no cell guide). These produce PS-only sermon groups with no topic/speaker — this is expected, not a bug. Genuinely PS-only groups are skipped correctly in incremental mode via `ps_file_indexed`.
- `Member27s` in filenames is a URL-decoded apostrophe (`%27s` → `27s`). The classifier regex handles this — `members?(?:27s?)?` matches "member's guide" in all encoded forms.
- ~50% of PS files are image-based PDFs with no extractable text — verse extraction relies on filename regex.
- The scraper skips handouts before downloading (classify-before-download).
- `create_react_agent` from `langgraph.prebuilt` is used — NOT `langchain.agents.create_agent`.
- BGE-M3 embedding model: 1.2 GB, multilingual (handles English + Mandarin sermons). Runs via `sentence-transformers` on MPS — no Ollama required.
- MLX ingest: `MLXChatModel` wraps `mlx_lm.generate` as a LangChain `BaseChatModel`. Qwen3 thinking mode is disabled via `enable_thinking=False` in `apply_chat_template` (with `<think>` stripping as fallback) to avoid wasting tokens on reasoning during structured ingest tasks.
- MLX chat: `MLXChatModel` does NOT implement `bind_tools`, so it can't drive the ReAct agent. The chat path uses `mlx_lm.server` (OpenAI-compat) + `ChatOpenAI` instead — see `_ensure_mlx_server()` in `src/llm.py`. The subprocess is spawned lazily on the first MLX chat request and shut down via `atexit` / `SIGTERM` / `SIGINT` / `SIGHUP` (handlers must be registered at module load on the main thread; `signal.signal()` is a no-op from Gradio worker threads). `SIGKILL` and hard crashes bypass cleanup — orphan `mlx_lm.server` / `ollama serve` processes can be detected with `pgrep -fl "mlx_lm|ollama serve"`.
- Agent caching (`_agent_cache` / `_llm_cache` in `app.py`): cached **per provider** for `ollama_local` / `groq` / `gemini`, but **rebuilt per call for MLX**. Reason: `mlx_lm.server`'s OpenAI-compat layer can leave the httpx client of `ChatOpenAI` in a closed state after some streamed responses, surfacing as `"Cannot send a request, as the client has been closed."` — rebuilding per call avoids stale handles (model stays loaded in the server, so the overhead is millis of Pydantic construction). `_llm_cache` also holds a **strong reference to the LLM instance** to prevent garbage-collection from triggering httpx `__del__` close on the underlying client while LangGraph still holds the wrapped runnable.
- "Client closed" retry: `respond()` in `app.py` wraps `agent.invoke()` in a 3-attempt loop that, on `"client has been closed"`, evicts both `_agent_cache[provider]` and `_llm_cache[provider]` and rebuilds before retrying. Other exceptions bubble immediately.
- ReAct agent reuses ~3,238 tokens of system prompt + tool schemas per LLM call. For MLX, `--prompt-cache-size 4 --prompt-cache-bytes 8000000000` is passed to `mlx_lm.server` so the static preamble is only prefilled once — second/third calls are ~70-80% faster (5s → 1s in benchmarks).
- Chat history window: `app.py:respond()` passes the last 6 history entries (3 user + 3 assistant exchanges) to the agent. Bump the slice if you need longer memory.
- `OLLAMA_CHAT_MODEL` and `OLLAMA_INGEST_MODEL` are auto-detected via `_auto_detect_ollama_model()` in `src/llm.py`: if not set in `.env`, it queries `localhost:11434/api/tags` and picks the first available model, raising `RuntimeError` if none are found.
- BGE-M3 embedding init in `SermonVectorStore` is lazy — deferred to first `_upsert_in_batches` or `_search` call, so importing the class does not require Ollama to be running.
- `bible_ingest.py` treats `status="skipped"` as equivalent to `"indexed"` in `_is_indexed()`, so missing EPUB files are not retried on every run.
