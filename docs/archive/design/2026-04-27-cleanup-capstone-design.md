# Cleanup, Simplification & Capstone Preparation Design

**Date:** 2026-04-27  
**Status:** Approved  
**Approach:** Option A — big-bang single commit

---

## Goal

Clean up the repository for capstone submission by removing abandoned code, simplifying the LLM stack to Ollama-only, adding a scatterplot chart, and updating documentation.

---

## Section 1 — Dead Code Removal

The following paths are confirmed abandoned and will be deleted entirely:

| Path | Reason |
|---|---|
| `app/` | Old Flask app branded "AlphaPulse AI" — not related to BBTC |
| `deploy/` | Partial FastAPI backend — never completed or wired up |
| `templates/` | Flask HTML templates for old `app/` |
| `static/` | Flask CSS/JS for old `app/` |
| `flask_session/` | Flask session files from old `app/` |
| `run.py` | Flask runner that imports from dead `app/` |
| `scratch/` | 8 one-off debug/exploration scripts |
| `vectorstore/` | Duplicate/stale Chroma SQLite at repo root |

Files retained: `app.py`, `src/`, `dagster_pipeline.py`, `quick_ingest.py`, `requirements.txt`, `Dockerfile`, `render.yaml`, `tests/`, `docs/`, `CLAUDE.md`, `README.md`, `.gitignore`.

---

## Section 2 — Ollama-Only LLM Simplification

Remove all Groq and Gemini/Google cloud provider code. The app will use only local Ollama.

### `src/llm.py`
- Strip `get_llm()` down to Ollama only.
- Remove `provider_type` parameter or make it a no-op stub.
- Remove imports for `langchain-groq` and `langchain-google-genai`.

### `src/ingestion/metadata_extractor.py`
- Remove Groq primary + rate-limit (HTTP 429) fallback logic.
- Use Ollama (`llama3.2:3b`) directly as the single provider.

### `app.py`
- Remove the `provider_radio` dropdown widget (⚙️ Engine Settings sidebar).
- Remove `if provider == "Gemini"` / `elif provider == "Groq"` branching in `respond()`.
- Agent always initialises with `get_llm()` (Ollama).
- Sidebar ⚙️ section becomes a simple system status display (no provider selector).
- Remove the `GEMINI_API_KEY → GOOGLE_API_KEY` remap block at the top of `app.py`.

### `requirements.txt`
- Remove `langchain-groq`, `langchain-google-genai`, and `groq` packages.

### `render.yaml`
- Remove `GROQ_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY` env var entries.

### `.env` / `.env.example`
- Remove cloud API key entries if `.env.example` exists.

---

## Section 3 — Scatterplot Chart (`sermons_scatter`)

### `src/tools/matplotlib_tool.py`
Add a new `sermons_scatter` chart type to the existing `make_matplotlib_tool` function:

- **SQL query:** `SELECT year, speaker, COUNT(*) FROM sermons WHERE year IS NOT NULL AND speaker IS NOT NULL AND speaker != '' GROUP BY year, speaker ORDER BY year`
- **Rendering:** Bubble/scatter plot — x-axis = year, y-axis = speaker name, bubble size proportional to sermon count.
- **Output:** PNG saved to `/tmp/sermons_scatter.png`, path returned via the existing `CHART_PATH:` injection mechanism.

### `app.py`
- Update the agent system prompt: add `'sermons_scatter'` to the list of valid `chart_name` values.
- Add a new Quick Inquiry example: `"Show a scatter plot of sermon counts by speaker and year"`.

---

## Section 4 — README Rewrite

Rewrite `README.md` for capstone presentation:

1. **Project title and motivation** — what the system does and why it was built.
2. **Architecture overview** — text-based data-flow diagram (matches CLAUDE.md diagram).
3. **Key components table** — maps each module to its purpose.
4. **Local setup** — Ollama-only instructions (no cloud API keys required).
5. **Running the app** — `python app.py`, Dagster pipeline, `quick_ingest.py`.
6. **Deployment** — Render via Docker, persistent volume for `data/`.
7. **Agent tools** — brief description of SQL, vector search, chart, and Bible tools.

---

## Change Summary

| Area | Change |
|---|---|
| Dead code (8 paths) | Deleted |
| `src/llm.py` | Ollama-only, remove cloud branches |
| `src/ingestion/metadata_extractor.py` | Ollama-only, remove Groq fallback |
| `app.py` | Remove provider radio; add scatterplot example; remove API key remap |
| `src/tools/matplotlib_tool.py` | Add `sermons_scatter` chart type |
| `requirements.txt` | Remove `langchain-groq`, `langchain-google-genai`, `groq` |
| `render.yaml` | Remove 3 cloud API key env vars |
| `README.md` | Rewritten for capstone presentation |

---

## What Does NOT Change

- `src/storage/` (chroma_store, sqlite_store, reranker) — untouched
- `src/tools/sql_tool.py`, `vector_tool.py`, `bible_tool.py` — untouched
- `src/scraper/bbtc_scraper.py` — untouched
- `dagster_pipeline.py`, `quick_ingest.py` — untouched
- `Dockerfile` — untouched
- `tests/` — untouched
