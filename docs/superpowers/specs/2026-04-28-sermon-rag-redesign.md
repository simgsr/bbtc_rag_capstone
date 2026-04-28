# BBTC Sermon RAG — Full Redesign Spec

**Date:** 2026-04-28  
**Status:** Approved

---

## 1. Problem Statement

The current RAG pipeline ingests each file (NG or PS) as an independent record. There is no concept of a "sermon unit" — the NG and PS for the same Sunday are unrelated in the data model. This causes:

- Structured queries ("list verses by speaker", "most preached book") to fail or return garbage
- Semantic search to return irrelevant chunks because metadata is incomplete or wrong
- LLM-extracted metadata to be unreliable (current approach guesses speaker from filename)

The NG PDFs already contain clearly labeled structured fields (`TOPIC`, `SPEAKER`, `THEME`, `DATE`) — the current pipeline ignores this and uses fragile LLM guessing instead.

---

## 2. Definitions

| Term | Meaning |
|---|---|
| **NG** | Notes / Cell Guide / Members Guide / Members Copy — the weekly sermon notes PDF |
| **PS** | PPT / Slides — the PowerPoint or slide deck PDF for the same sermon |
| **Sermon Unit** | One NG + one PS for the same weekend (Saturday or Sunday service), or a standalone PS for holidays/special occasions |
| **Handout** | Supplementary material — **ignored** during scraping and ingestion |

---

## 3. Architecture Overview

```
BBTC Website
    ↓
BBTCScraper (classify-before-download: skip handout/unknown)
    ↓
data/staging/  (NG + PS files only)
    ↓
ingest.py
  ├── CLASSIFY  → NG | PS per file
  ├── GROUP     → sermon units (NG + PS paired by date proximity / topic overlap)
  ├── EXTRACT   → regex from NG labeled fields + LLM verses from PS
  ├── SUMMARIZE → llama3.1:8b generates unified NG+PS summary
  └── EMBED     → BGE-M3 → ChromaDB sermon_collection
    ↓
SQLite  ← structured metadata + verses table
ChromaDB ← semantic search over full NG body + LLM summaries
    ↓
LangGraph ReAct Agent (3 tools)
    ↓
Gradio UI
```

---

## 4. Data Model

### 4.1 SQLite — `sermons` table

```sql
CREATE TABLE sermons (
  sermon_id    TEXT PRIMARY KEY,   -- slugified: "2024-01-06-the-heart-of-discipleship"
  date         TEXT,               -- YYYY-MM-DD (first day of the weekend)
  year         INTEGER,
  language     TEXT,               -- "English" | "Mandarin"
  speaker      TEXT,
  topic        TEXT,
  theme        TEXT,
  summary      TEXT,               -- LLM-generated from full NG + PS text combined
  key_verse    TEXT,               -- e.g. "Luke 9:23" — first verse in PS
  ng_file      TEXT,               -- staging filename of NG (nullable)
  ps_file      TEXT,               -- staging filename of PS (nullable)
  status       TEXT                -- grouped → extracted → indexed | failed
);
```

### 4.2 SQLite — `verses` table

```sql
CREATE TABLE verses (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  sermon_id    TEXT REFERENCES sermons(sermon_id),
  verse_ref    TEXT,               -- normalized "Luke 9:23"
  book         TEXT,               -- "Luke"
  chapter      INTEGER,
  verse_start  INTEGER,
  verse_end    INTEGER,            -- for ranges "Luke 9:23-27" → verse_end = 27
  is_key_verse INTEGER DEFAULT 0
);
```

The `verses` table enables clean SQL aggregations:
- `SELECT book, COUNT(*) FROM verses GROUP BY book ORDER BY COUNT(*) DESC` — most preached book
- `SELECT verse_ref, COUNT(*) FROM verses JOIN sermons USING(sermon_id) WHERE speaker = ? GROUP BY verse_ref` — verses by speaker

### 4.3 ChromaDB — `sermon_collection`

Two document types per sermon, both in the same collection:

| Type | Content | chunk_size | overlap |
|---|---|---|---|
| `body` | Full NG body text (Introduction, outline, discussion Qs) | 800 | 150 |
| `summary` | LLM-generated unified NG+PS summary | single chunk | — |

**Metadata on every chunk:**
```json
{
  "sermon_id": "2024-01-06-the-heart-of-discipleship",
  "doc_type": "body" | "summary",
  "speaker": "SP Chua Seng Lee",
  "date": "2024-01-06",
  "year": 2024,
  "topic": "The Heart of Discipleship",
  "theme": "#CanIPrayForYou",
  "language": "English",
  "key_verse": "Luke 9:23"
}
```

**Embedding model:** `BGE-M3` (via Ollama) — handles English and Mandarin, superior multilingual retrieval vs nomic-embed-text.

---

## 5. Ingestion Pipeline (`ingest.py`)

### 5.1 Modes

```bash
python ingest.py              # incremental — only new files not yet in SQLite
python ingest.py --wipe       # full rebuild — clear SQLite + ChromaDB, re-ingest all
python ingest.py --year 2024  # re-ingest a specific year only
```

### 5.2 Step-by-Step

**Step 1 — Classify**
`src/ingestion/file_classifier.py` classifies each file in `staging/` as `NG | PS | handout | unknown` based on filename patterns. Handout and unknown are skipped.

**Step 2 — Group**
`src/ingestion/sermon_grouper.py` pairs NG + PS files into sermon units using:
- Date proximity (≤ 3 days between dates in filenames)
- Topic-word Jaccard similarity (≥ 0.5) as fallback
- Standalone PS (no matching NG) become a sermon unit with `ng_file = NULL`

**Step 3 — Extract NG metadata**
`src/ingestion/ng_extractor.py` (new):
- PyMuPDF text extraction
- **Regex** on labeled fields: `TOPIC`, `SPEAKER`, `THEME`, `DATE` (explicit in all 2024+ NGs)
- Fallback: `filename_parser.py` for older files without labeled structure
- If neither regex nor filename parsing yields a speaker, `speaker = NULL` and status = `partial`
- Body text = everything after the `INTRODUCTION` label (or full text if no label found)

**Step 4 — Extract PS verses**
`src/ingestion/ps_extractor.py` (new):
- **Filename regex** — parse verse references like `LUKE-9V23`, `JOHN-3V16`, `HEBREWS` from filename
- **Text extraction** — PyMuPDF on PS files that have readable text (~50%)
- **LLM extraction** — `llama3.1:8b` on extracted text for structured verse list when text is available
- First verse found = `key_verse`; all verses stored in `verses` table
- If no verse is found from any source, `key_verse = NULL` and the sermon is still ingested (verses table has no rows for it)

**Step 5 — Generate summary**
`llama3.1:8b` prompt combining:
- NG: topic, theme, body text (first 2000 chars)
- PS: key verse, all verses, any extracted slide text

Output: 3–5 sentence summary stored in `sermons.summary` and embedded as a standalone ChromaDB chunk.

**Step 6 — Embed**
`src/storage/chroma_store.py`:
- Chunk NG body text (800/150) with BGE-M3
- Embed LLM summary as single chunk
- Store with full sermon metadata as ChromaDB filter fields

---

## 6. Scraper Changes (`src/scraper/bbtc_scraper.py`)

- Before downloading any file, classify the URL filename
- If `handout` or `unknown` → skip, do not download
- Log skipped files at DEBUG level
- Keeps `staging/` clean — only NG and PS files ever downloaded

---

## 7. Dagster Pipeline (`dagster_pipeline.py`)

- Thin wrapper around `ingest.py` logic
- Weekly schedule: Saturday night (so new Sunday files are available)
- Incremental mode only — calls `ingest.py` equivalent with `--incremental`
- Dagster UI available via `dagster dev -m dagster_pipeline` for monitoring

---

## 8. Agent Tools

### `sql_tool`
Executes SQL against `data/sermons.db`. System prompt guides it to use the `verses` table for verse/book queries and `sermons` for speaker/date queries.

Example queries it must handle:
- `SELECT DISTINCT speaker FROM sermons ORDER BY speaker`
- `SELECT speaker, COUNT(*) as count FROM sermons WHERE year = 2023 GROUP BY speaker ORDER BY count DESC`
- `SELECT book, COUNT(*) as count FROM verses GROUP BY book ORDER BY count DESC LIMIT 10`
- `SELECT v.verse_ref, COUNT(*) FROM verses v JOIN sermons s USING(sermon_id) WHERE s.speaker LIKE '%Chua%' GROUP BY v.verse_ref ORDER BY COUNT(*) DESC`

### `search_tool`
Semantic search over `sermon_collection` using BGE-M3. Supports metadata filters (speaker, year, language). Returns top-k chunks with source sermon metadata. Uses CrossEncoder reranker for final ranking.

### `chart_tool`
Plotly interactive charts rendered in Gradio via `gr.Plot()`:
- `sermons_per_speaker` — horizontal bar chart
- `sermons_per_year` — line or bar chart
- `verses_per_book` — bar chart (from `verses` table)
- `top_speakers_by_year` — grouped bar or heatmap
- 3D scatter — deferred (future enhancement)

---

## 9. Files to Keep / Delete / Create

### Keep & revise
- `app.py` — rewrite agent tools + swap matplotlib → Plotly
- `dagster_pipeline.py` — simplify to thin wrapper
- `src/scraper/bbtc_scraper.py` — add classify-before-download
- `src/ingestion/file_classifier.py` — keep, may refine patterns
- `src/ingestion/filename_parser.py` — keep as fallback for older files
- `src/ingestion/sermon_grouper.py` — revise grouping logic
- `src/storage/sqlite_store.py` — rewrite for new schema
- `src/storage/chroma_store.py` — update embedding model + doc structure
- `src/storage/normalize_speaker.py` — keep
- `src/storage/reranker.py` — keep
- `src/tools/sql_tool.py` — revise system prompt for verses table
- `src/tools/vector_tool.py` — update for BGE-M3 + new metadata fields
- `src/tools/viz_tool.py` — replace matplotlib with Plotly
- `src/llm.py` — keep

### Create new
- `ingest.py` — main ingestion script (replaces dagster_pipeline as runnable)
- `src/ingestion/ng_extractor.py` — regex + fallback LLM NG metadata extraction
- `src/ingestion/ps_extractor.py` — filename regex + LLM verse extraction

### Delete
- `quick_ingest.py`
- `backfill_metadata.py`
- `backfill_text.py`
- `normalize_speakers.py` (root level)
- `wipe_and_restart.sh`
- `scratch/` (entire directory)
- `src/ingestion/metadata_extractor.py`
- `src/ingestion/speaker_from_filename.py`
- `src/ingestion/speaker_from_pdf.py`
- `src/ingestion/bible/` (entire directory)
- `src/agents/sermon_structure_agent.py`
- `src/graph/state.py`
- `render.yaml`
- `Dockerfile`

---

## 10. CLAUDE.md Updates

- Update project overview: sermon-unit ingestion model
- Update architecture data flow diagram
- Update SQLite schema to include `verses` table
- Update component table with new files
- Update running instructions (`ingest.py` replaces Dagster for manual runs)
- Add Ollama model recommendations (BGE-M3 for embeddings, llama3.1:8b for extraction)

---

## 11. Success Criteria

The redesign is complete when:
1. `python ingest.py --wipe` runs without errors and populates SQLite + ChromaDB from `staging/`
2. SQL queries for speakers, years, verse counts, and book aggregations return correct results
3. Semantic search returns relevant sermon chunks for open-ended questions
4. Gradio UI answers: "list all speakers", "who preached most in 2023", "what verses did SP Chua preach", "what is the most preached book" — all correctly
5. Dagster weekly schedule runs incremental ingest successfully
6. Interactive Plotly charts render correctly in Gradio
