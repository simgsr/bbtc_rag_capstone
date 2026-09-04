# CLAUDE.md

Guidance for Claude Code when working in this repo. The full architecture, DB schema, and component map are derivable from the code (every module has a docstring; README.md has the overview) — this file keeps only what you can't reconstruct by reading. Everyday commands live in the Makefile and README.

## Project Overview

**Hybrid Agentic RAG pipeline** for the BBTC (Bethesda Bedok-Tampines Church) sermon archive: scrapes sermon PDFs, groups them into **sermon units** (one Notes/Guide + one Slides/PPT per Sunday), extracts metadata, stores in SQLite + ChromaDB, and serves a Gradio chat UI backed by a LangGraph ReAct agent.

## Environment Setup

- Ollama must be running: `ollama serve`. Chat + ingest LLMs run on Ollama; embeddings run on Apple Silicon (MPS) without Ollama.
- **Chat LLM** — picked at runtime via the "Inference Engine" dropdown in `app.py`. The list is built at startup: local Ollama models are auto-discovered (`_discover_ollama_models()`, queries `/api/tags`), filtered to tool-capable models ≥30B params (`_OLLAMA_PARAM_MIN_B`), with `_PINNED_MODEL` (default `qwen3.8:latest`) pinned to the front as the default engine; static cloud engines (Gemini, Groq) are appended. To change the default/pin/min-params/cloud engines, edit `_PINNED_MODEL` / `_OLLAMA_PARAM_MIN_B` / `_CLOUD_OPTIONS` in `app.py` — the local list is discovered, not hardcoded. Agents/LLMs are cached per `(provider, model)`.
- **Ingest LLM** — `INGEST_PROVIDER` in `.env`: `ollama_local` (default; uses `OLLAMA_INGEST_MODEL` = `qwen3:4b` for summarisation — small + fast, distinct from the chat model) · `mlx` (faster but less stable — kept as an option, not the default) · `groq`/`gemini` (cloud fallbacks). Textless/image-only PDFs are read by the multimodal `OLLAMA_VISION_MODEL` (default `gemma4:e4b`) — see the vision-fallback quirk below.
- **Embeddings** — `EMBED_BACKEND` in `.env` (default `st` = BAAI/bge-m3 via sentence-transformers on MPS, 1024-dim). **Switching backends changes the vector space — you must wipe + re-ingest both collections** (`ingest.py --wipe` and `bible_ingest.py --wipe`) so stored and query vectors match.

## Architecture

### Sermon Unit Model

Every weekend BBTC posts two files that form one **sermon unit** (the atomic unit of ingestion):
- **NG** (Notes/Guide): PDF with labeled `TOPIC`, `SPEAKER`, `THEME`, `DATE` fields + body text
- **PS** (Slides/PPT): PDF whose filename encodes the key verse

### Agent Tools

- **`sql_query_tool`** — SQL against `data/sermons.db`. For **gap/coverage analysis** ("books never preached") use an anti-join against the `bible_books` reference table (`... WHERE book_name NOT IN (SELECT DISTINCT book FROM verses)`) — don't recall the 66-book list. Returns up to 200 rows; a result hitting exactly 200 appends a truncation notice.
- **`search_sermons_tool`** — BGE-M3 semantic search over `sermon_collection`.
- **`viz_tool`** — Plotly charts; optional `top_n` (default 15) for ranked charts.
- **`get_bible_versions_tool`** — all stored translations of a verse (KJV, ASV, YLT, BBE, ChiUn, NIV, ESV).
- **`search_bible_tool`** — semantic Bible search; pass an English `version` (e.g. `NIV`) for English topic queries so Chinese `ChiUn` verses don't surface.

## Notable Quirks

- **Vision fallback for textless PDFs** (`ingest.py` + `src/ingestion/vision_extractor.py`): ~50% of PS (slides) PDFs are image-only with no extractable text, and some groups have no NG (notes) file at all. When NG text is missing/empty or a group is PS-only, `process_group` renders the PDF pages (PyMuPDF `get_pixmap`) and asks the multimodal `OLLAMA_VISION_MODEL` (default `gemma4:e4b`, via Ollama — no MLX runtime) to recover topic/speaker/theme/key-verse/summary. Vision **fills gaps only** — it never overwrites text-derived values. Textless PS slides get a separate verse-only vision call. Set `OLLAMA_VISION_MODEL` empty to disable.
- **Metadata chunks for every sermon** (`ingest.py`): every sermon gets a `doc_type="metadata"` chunk embedding `"Topic | Theme | Speaker | Key verse | Date"` so topical/title queries retrieve the right sermon directly. `search_sermons_tool` keeps these retrievable but doesn't surface their content as excerpts (the header already shows it); for textless sermons whose only chunk is metadata it falls back to that content. Title text is built by the shared `build_sermon_title_text()` in `src/ingestion/title_chunk.py` so ingest and backfill can't drift. Backfill for pre-change sermons: `python backfill_title_chunks.py` (idempotent).
- **`search_sermons_tool` speaker filter is partial-match** (`src/tools/vector_tool.py`): Chroma `where` is exact-match only but speakers carry titles ("SP Chua Seng Lee"), so the filter fetches the **whole collection** then case-insensitive substring post-filters — guaranteeing a prolific speaker on a rare topic still fills `k`. Year/min_year/max_year still use `where`.
- **`.env` is loaded in `chroma_store.py`** (not just `app.py`/`src/llm.py`) so every entry point resolves `EMBED_BACKEND` identically — otherwise a standalone script would silently mix embedding backends within a collection.
- **Embedding-dim guard** (`SermonVectorStore._check_vector_dim_alignment`): raises `RuntimeError` if the embedder's output dim mismatches a non-empty collection (the `EMBED_BACKEND`-switched-without-wipe footgun). Catches dim changes (1024→4096); does NOT catch same-dim swaps (`st`↔`mlx_bge`) — those are prevented by the `.env` rule above.
- **Distance-space guard** (`SermonVectorStore._check_distance_space`): both collections use **cosine** (`hnsw:space="cosine"`); a pre-existing non-empty collection still on Chroma's default `l2` raises `RuntimeError` pointing to `scripts/migrate_chroma_cosine.py` (in-place, preserves stored vectors, filesystem backup). Chroma silently ignores the cosine metadata on `get_or_create` for an existing l2 collection — hence the guard.
- **Ollama sampling overrides** (`src/llm.py:get_llm`): `ChatOllama` gets explicit `seed`/`top_k`/`top_p`/`repeat_penalty` (env-overridable: `OLLAMA_SEED`/`OLLAMA_TOP_K`/`OLLAMA_TOP_P`/`OLLAMA_REPEAT_PENALTY`) so per-model Modelfile defaults don't silently control generation. `presence_penalty`/`min_p` are NOT exposed by `ChatOllama` — to neutralise those, edit the Modelfile directly (`ollama show --modelfile X > MF; ollama create -q`).

## Notable Quirks (legacy)

- NG labeled fields are reliable for 2022+ files; older files fall back to `filename_parser.py`.
- Pre-2020 pages sometimes posted only slides → PS-only groups with no topic/speaker (expected, not a bug); genuinely PS-only groups are skipped correctly in incremental mode via `ps_file_indexed`.
- `Member27s` in filenames is a URL-decoded apostrophe (`%27s` → `27s`); the classifier regex handles all encoded forms.
- ~50% of PS files are image-based PDFs with no extractable text — verse extraction relies on filename regex.
- The scraper skips handouts before downloading (classify-before-download).
- `create_react_agent` from `langgraph.prebuilt` is used — NOT `langchain.agents.create_agent`.
- MLX chat: `MLXChatModel` does NOT implement `bind_tools`, so it can't drive the ReAct agent; the chat path uses `mlx_lm.server` (OpenAI-compat) + `ChatOpenAI` instead (`_ensure_mlx_server()` in `src/llm.py`). Subprocess cleanup via `atexit`/`SIGTERM`/`SIGINT`/`SIGHUP` (registered at module load, main thread); `SIGKILL`/crashes bypass it — orphan processes: `pgrep -fl "mlx_lm|ollama serve"`.
- Agent caching (`_agent_cache`/`_llm_cache` in `app.py`): keyed by `(provider, model)`; `_llm_cache` holds a strong reference to the LLM to prevent GC from closing the httpx client under LangGraph.
- "Client closed" retry: `respond()` retries `agent.invoke()` 3× on `"client has been closed"`, evicting both caches and rebuilding; other exceptions bubble.
- Chat history window: `respond()` passes the last 6 history entries (3 user + 3 assistant exchanges).
- `OLLAMA_CHAT_MODEL`/`OLLAMA_INGEST_MODEL` auto-detect via `_auto_detect_ollama_model()` in `src/llm.py` if unset (queries `/api/tags`, raises if none found).
- Embedding init in `SermonVectorStore` is lazy — deferred to first `_upsert_in_batches`/`_search` (reads `EMBED_BACKEND` then), so importing doesn't load the model.
- `bible_ingest.py` treats `status="skipped"` as `"indexed"` in `_is_indexed()`, so missing EPUBs aren't retried every run.
- `_run_archive_update` (app.py "Update Archive" button): scrapes current year + incremental ingest as isolated subprocesses. **Freshness caveat:** the header stats bar reads SQLite fresh, but the chat agent's semantic search holds ChromaDB loaded in memory at startup and won't see new chunks until the app restarts. Chroma has no first-class concurrent-multi-process support — treat as an occasional maintenance action.

## RAG Evaluation Harness

See `evals/CLAUDE.md` — retrieval + groundedness harness driven by `evals/golden_set.json` (`python -m evals.run_eval`).
