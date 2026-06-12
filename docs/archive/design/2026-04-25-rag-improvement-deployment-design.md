# RAG Improvement & Cloud Deployment Design

**Date:** 2026-04-25  
**Status:** Approved

---

## Problem Statement

The existing RAG pipeline returns empty results or shallow answers because:

1. **Embedding mismatch** — documents are indexed with Ollama `nomic-embed-text`, but if Ollama is unavailable at query time, ChromaDB falls back to its own default embedder. The vector spaces are incompatible.
2. **Stub reranker** — `Reranker` is a pass-through that does not actually rerank candidates.
3. **Poor metadata coverage** — 43% of sermons are missing speaker, 39% missing date, 37% missing verse.
4. **Gradio UI** — was a testing harness, not a production interface.
5. **No cloud deployment path** — Ollama cannot run on Render or HuggingFace Spaces.

---

## Goals

- Fix RAG quality: consistent embeddings, real reranking, metadata-aware retrieval, cited answers.
- Replace Gradio with a professional React dashboard with interactive Plotly charts.
- Deploy backend to Render, frontend to HuggingFace Spaces.
- Preserve the existing Dagster weekly ingestion pipeline.

---

## Architecture

```
HuggingFace Spaces (free, Docker)     Render (web service + cron + persistent disk)
┌──────────────────────────┐          ┌─────────────────────────────────────────┐
│  React + Vite            │ REST API │  FastAPI (uvicorn)                      │
│  - Analytics dashboard   │◄────────►│  - /api/stats                           │
│  - Plotly charts         │          │  - /api/charts/*                        │
│  - RAG chat panel        │          │  - /api/sermons                         │
└──────────────────────────┘          │  - /api/chat  (RAG)                     │
                                      │                                         │
                                      │  Sentence-Transformers                  │
                                      │  bge-small-en-v1.5 (embed)             │
                                      │  ms-marco-MiniLM-L-6-v2 (rerank)       │
                                      │                                         │
                                      │  Render Cron Job (Sunday 00:00)        │
                                      │  → dagster_pipeline.py                 │
                                      │                                         │
                                      │  Persistent Disk (/data)               │
                                      │  - sermons.db                          │
                                      │  - chroma_db/                          │
                                      │  - sermons/ (.txt files)               │
                                      │  - models/ (cached HF weights)        │
                                      └─────────────────────────────────────────┘
```

---

## Folder Structure

```
deploy/
├── backend/                    → Render web service
│   ├── api/
│   │   ├── main.py             FastAPI app, CORS, lifespan startup
│   │   ├── rag.py              RAG pipeline (embed → search → rerank → agent)
│   │   ├── charts.py           SQL-backed chart data endpoints
│   │   └── models.py           Pydantic request/response schemas
│   ├── scripts/
│   │   ├── reindex.py          One-time re-embed all sermons with bge-small
│   │   └── run_ingestion.py    Standalone ingestion script (called by Render cron)
│   ├── Dockerfile
│   ├── render.yaml
│   └── requirements.txt
└── frontend/                   → HuggingFace Spaces (Docker SDK)
    ├── src/
    │   ├── App.tsx
    │   ├── components/
    │   │   ├── StatCards.tsx
    │   │   ├── BarChart.tsx
    │   │   ├── BubbleChart.tsx
    │   │   ├── TopVersesChart.tsx
    │   │   └── ChatPanel.tsx
    │   ├── hooks/
    │   │   └── useApi.ts
    │   └── main.tsx
    ├── public/
    ├── Dockerfile
    ├── nginx.conf
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.js
    └── README.md               HuggingFace Space card (sdk: docker, app_port: 3000)
```

The existing `src/`, `data/`, `dagster_pipeline.py`, `quick_ingest.py` remain unchanged in the repo root. The `deploy/backend/Dockerfile` uses the **repo root as build context** and copies `src/` into the image, so all existing modules are available without modification.

---

## RAG Pipeline (Improved)

### Embedding

- Model: `BAAI/bge-small-en-v1.5` via `sentence-transformers`
- Used consistently for **both** indexing (`reindex.py`) and querying (`rag.py`)
- Model weights cached to `/data/models` on Render persistent disk

### Retrieval

```
POST /api/chat  { query, year_filter?, speaker_filter? }
   ↓
1. Build ChromaDB `where` filter from explicit year/speaker params (if provided)
2. Embed query with bge-small-en-v1.5
3. ChromaDB search: fetch top 20 candidates (with filter if set)
4. CrossEncoder rerank: cross-encoder/ms-marco-MiniLM-L-6-v2 → top 5
5. LangGraph ReAct agent:
     - sql_query_tool   → stats, counts, date lookups
     - search_sermons_tool → pre-reranked chunks (passed directly)
     - charts are handled client-side, not via tool
6. Response: { answer, citations: [{filename, speaker, date, verse}] }
```

### LLM Selection (unchanged logic)

- Default: Groq (`llama3-70b-8192`)
- Fallback: Gemini (`gemini-1.5-flash`) on rate limit
- No Ollama in production

### System Prompt (improved)

```
You are the BBTC Sermon Intelligence Assistant.
Answer ONLY from the provided sermon excerpts. Never invent facts.
For every claim, cite the sermon filename and speaker.
Use 'sql_query_tool' for counts/statistics (column: primary_verse, not verse).
The user may filter by year or speaker — respect those constraints.
Be concise. If the answer is not in the excerpts, say so explicitly.
```

---

## Frontend Dashboard

### Layout

```
┌──────────────────────────────────────────────────────────┐
│  BBTC Sermon Intelligence          [Year ▼] [Speaker ▼]  │
├──────────────┬──────────────┬───────────────────────────┤
│ 1,162 Sermons│  12 Speakers │  2015 – 2026              │
├──────────────┴──────────────┴───────────────────────────┤
│  Sermons per Year (bar)  │  Top Speakers (horizontal bar)│
├──────────────────────────┴───────────────────────────────┤
│  Bubble Chart: Year (x) × Speaker (y) × Count (size)    │
│  Click bubble → pre-fills year+speaker filter in chat   │
├──────────────────────────────────────────────────────────┤
│  Top Bible Books / Verses (horizontal bar)              │
├──────────────────────────────────────────────────────────┤
│  💬 Ask about the sermons...                [Send]       │
│  (chat panel, shows citations below each answer)        │
└──────────────────────────────────────────────────────────┘
```

### Tech Stack

- React 18 + Vite + TypeScript
- Tailwind CSS (dark theme, professional)
- Plotly.js (`react-plotly.js`) — bar, horizontal bar, bubble/scatter, treemap
- Fetch API for backend calls (no Redux, keep it simple)

### Chart → Chat Integration

Clicking a bubble in the scatter chart sets `year` and `speaker` state which:
1. Highlights the selected filter in the header dropdowns
2. Passes `year_filter` and `speaker_filter` to the next `/api/chat` call

---

## API Endpoints

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/stats` | `{ total_sermons, total_speakers, year_min, year_max }` |
| GET | `/api/charts/by-year` | `[{ year, count }]` |
| GET | `/api/charts/by-speaker` | `[{ speaker, count }]` |
| GET | `/api/charts/by-verse` | `[{ bible_book, count }]` |
| GET | `/api/charts/scatter` | `[{ year, speaker, count }]` |
| GET | `/api/sermons` | paginated list with `?year=&speaker=&page=` |
| POST | `/api/chat` | `{ query, year_filter?, speaker_filter? }` → `{ answer, citations }` |

All endpoints return JSON. CORS is configured to allow the HF Spaces origin.

---

## Deployment Configuration

### Render (`render.yaml`)

```yaml
services:
  - type: web
    name: bbtc-sermon-api
    runtime: docker
    dockerfilePath: deploy/backend/Dockerfile
    disk:
      name: sermon-data
      mountPath: /data
      sizeGB: 10
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: GEMINI_API_KEY
        sync: false
      - key: FRONTEND_URL
        sync: false
      - key: MODEL_CACHE_DIR
        value: /data/models
      - key: DATA_DIR
        value: /data

  - type: cron
    name: bbtc-sermon-ingestion
    runtime: docker
    dockerfilePath: deploy/backend/Dockerfile
    schedule: "0 0 * * 0"
    dockerCommand: python deploy/backend/scripts/run_ingestion.py
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: DATA_DIR
        value: /data
```

### HuggingFace Spaces (`README.md` card)

```yaml
---
title: BBTC Sermon Intelligence
emoji: 📖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 3000
pinned: true
---
```

### Environment Variables

| Variable | Service | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | Render | Primary LLM |
| `GEMINI_API_KEY` | Render | LLM fallback |
| `FRONTEND_URL` | Render | CORS allowlist |
| `DATA_DIR` | Render | Root for DB + ChromaDB + sermons |
| `MODEL_CACHE_DIR` | Render | HuggingFace model cache |
| `VITE_API_URL` | HF Spaces | Render API base URL (build arg) |

---

## One-Time Re-index

Run once after first Render deploy to rebuild ChromaDB with consistent embeddings:

```bash
# On Render shell (or as a one-off job)
python deploy/backend/scripts/reindex.py
```

This script:
1. Loads `bge-small-en-v1.5` (cached to `MODEL_CACHE_DIR`)
2. Reads all `indexed` sermons from SQLite
3. Deletes the existing `sermon_collection` in ChromaDB
4. Re-chunks and re-embeds all `.txt` files (chunk_size=512, overlap=64)
5. Upserts to ChromaDB with full metadata
6. Estimated time: ~15 min on Render starter

---

## What Is Not Changing

- `src/scraper/bbtc_scraper.py` — unchanged
- `src/storage/sqlite_store.py` — unchanged
- `src/ingestion/metadata_extractor.py` — unchanged (Groq + Ollama fallback)
- `dagster_pipeline.py` — unchanged; `run_ingestion.py` calls its core logic directly without the Dagster scheduler overhead
- `data/sermons.db` and `.txt` files — migrated to `/data/` on Render disk
