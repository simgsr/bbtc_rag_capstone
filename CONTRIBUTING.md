# Contributing & Maintainer Guide

This guide is for developers taking over or maintaining **BBTC Sermon
Intelligence**. It complements the two reference docs:

- [`README.md`](README.md) — what the project is and how to run it.
- [`CLAUDE.md`](CLAUDE.md) — full architecture, component map, DB schema, and the
  "notable quirks" you will eventually hit.

## 1. Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # versions are pinned for reproducibility
cp .env.example .env                 # then fill in optional API keys
```

Or one-click: `make install` (creates the venv, installs deps, seeds `.env`).

**Platform:** built for Apple Silicon (macOS). The default inference/embedding
backends use MLX; cloud (Groq/Gemini) and Ollama backends are available as
fallbacks — see `.env.example` and `CLAUDE.md` → "Environment Setup".

## 2. Everyday commands

| Task | Make target | Direct command |
|---|---|---|
| Run the chat UI | `make run` | `python app.py` (→ http://127.0.0.1:7860) |
| Run tests | `make test` | `python -m pytest` |
| Scrape one year | `make scrape YEAR=2024` | `python src/scraper/bbtc_scraper.py 2024` |
| Incremental ingest | `make ingest` | `python ingest.py` |
| Full rebuild | — | `python ingest.py --wipe` |
| Weekly scheduler | `make dagster` | `DAGSTER_HOME=$(pwd)/.dagster dagster dev -m dagster_pipeline` |

## 3. Project layout

```
app.py                 # Gradio UI + LangGraph ReAct agent (entry point)
ingest.py              # classify → group → extract → summarize → embed
dagster_pipeline.py    # weekly Saturday schedule wrapping ingest.py
src/
  scraper/             # BBTC website scraper (classify-before-download)
  ingestion/           # file classifier, grouper, NG/PS extractors, Bible EPUB parser
  storage/             # SQLite registry, ChromaDB store, name normalizers
  tools/               # the 5 agent tools (sql, vector, bible, viz)
  llm.py               # LLM factory (ingest + chat) — MLX/Ollama/Groq/Gemini
  ui_helpers.py        # pure, unit-tested UI helpers
tests/                 # pytest suite (hermetic — no services required)
scripts/               # one-time migrations (see scripts/README.md)
docs/                  # design notes & plans (see docs/README.md)
data/                  # generated artifacts — gitignored, never committed
```

Every source module now carries a module docstring explaining its role; start
there when navigating an unfamiliar file.

## 4. Testing

The suite is hermetic — it spins up no LLM, Ollama, or MLX server, so it runs in
~1.5s and is safe to run on every change:

```bash
python -m pytest            # all 107 tests
python -m pytest tests/test_sql_tool.py -v
```

**Any change to ingestion, storage, or tools must keep the suite green.** Add a
test alongside behavioral changes.

## 5. Data & the golden rule of embeddings

Everything under `data/` is generated and gitignored (the DB, ChromaDB, staged
files). Do not commit it.

⚠️ **Switching `EMBED_BACKEND` changes the vector space.** Stored document vectors
and query vectors must come from the same backend, so after changing it you MUST
wipe and re-ingest both collections:

```bash
python ingest.py --wipe
python -m src.ingestion.bible.bible_ingest --wipe
```

## 6. Operational gotchas

- **Orphan model servers.** `SIGKILL` / hard crashes bypass the MLX / Ollama
  cleanup handlers, leaving `mlx_lm.server` or a self-spawned `ollama serve`
  running. Detect and reclaim:
  ```bash
  pgrep -fl "mlx_lm|ollama serve"
  kill <pid>
  ```
  (Never kill a pre-existing *system* `ollama serve` you didn't spawn.)
- **First MLX run is slow.** Weights download from HuggingFace on first use; the
  80B option is ~85 GB — raise `MLX_SERVER_STARTUP_TIMEOUT` if it times out.
- **"Client has been closed" (MLX chat).** Handled by a rebuild-and-retry loop in
  `app.py:respond()`; see `CLAUDE.md` → "Notable Quirks" if you touch that path.

## 7. Making changes

1. Branch off `main`.
2. Keep behavior changes covered by tests; run `python -m pytest` before pushing.
3. Update `README.md` / `CLAUDE.md` when you change architecture, tools, schema,
   or agent behavior — the SQL/tool docstrings are a contract the LLM reads, so
   keep them accurate.
4. Open a PR against `main` with a clear description of the change and how it was
   verified.
