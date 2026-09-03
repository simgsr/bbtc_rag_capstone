# REPO_AUDIT.md — repo-onboard audit artifact

**Repo:** `simgsr/bbtc_rag_capstone` (https://github.com/simgsr/bbtc_rag_capstone)
**Audited:** 2026-09-03 · **Machine:** Apple M5 Max, 18 cores, 128 GB RAM, arm64
**Method:** `/repo-onboard` skill — five gates, nothing executed before approval.

---

## Gate 0 — Scope & isolation

- Repo confirmed: the user's own project, already on this Mac.
- User chose **full onboard** and **audit of the current working tree** (no isolated clone — the isolated-clone rule exists for untrusted external repos; this is the owner's own checkout).
- Path: `/Users/simgsr/Documents/git_project/bbtc_rag_capstone`

## Gate 1 — Static security audit

**VERDICT: PASS**

Findings (all read-only; nothing executed):
- **Dependencies** — all 25 direct deps pinned `==`; no typosquats, no `--index-url`/`--extra-index-url` overrides, no VCS deps, no CUDA pins. `mlx-lm`/`mlx-embeddings` are Apple-native.
- **`pyproject.toml`** — metadata only; no build hooks, no entry points, no install-time scripts.
- **Supply chain** — annotated tag `v1.0.0` dereferences exactly to HEAD (`e99e6c1`); all 163 commits authored by the repo owner; MIT LICENSE present.
- **No obfuscation** — no `eval`/`exec` on variables (all hits are `re.compile`), no base64 blobs, no shipped `.pyc`, no `.pth`, no post-install execution surface.
- **Network surface is app-intended** — `bbtc.com.sg` scraping, local Ollama (`127.0.0.1:11434`), Scrollmapper public-domain Bible JSON (fixed GitHub URL), Google Fonts + BBTC logo (UI assets). No `requests`/`httpx`/`.post`; no download-and-execute (PDFs/JSON data only).
- **No exfiltration flow** — API keys read from env and passed to the cloud clients they belong to.
- **Path traversal hardened** — scraper `_staging_filename` decodes-then-basenames, rejects `.`/`..`, plus realpath containment; regression tests in `tests/test_scraper_paths.py`.
- **Destructive ops scoped** — `Makefile clean` targets only `.venv` and `data/`; migration `rmtree` on a temp work dir.
- **`.env.example`** — placeholders only; app binds `127.0.0.1` by default.

Observations (non-blocking):
- No lockfile existed (added in Gate 4).
- `cloudscraper` is a Cloudflare-bypass library (dual-use) — legitimate here (scraping the owner's own church's public archive).
- `.envrc` did not exist (added in Gate 2); `.gitignore` did not list it — fine, Template A contains no secrets.

## Gate 2 — direnv + auto-activating venv

- `direnv` 2.37.1 already installed; shell hook already present in `~/.zprofile` (login shells — what macOS terminals use). No duplicate added to `~/.zshrc` (would double-register the precmd hook).
- Wrote `.envrc` (Template A — repo-root `.venv`, matching the Makefile's `VENV_DIR`).
- `direnv allow` run; verified in a fresh login shell: `cd` into the repo → `direnv: loading …/.envrc` → `export +VIRTUAL_ENV` → `VIRTUAL_ENV=/Users/simgsr/Documents/git_project/bbtc_rag_capstone/.venv`.
- `.gitignore` already covered `.venv/` and `.env`; no change needed. `.envrc` left untracked (safe to commit — no secrets).

## Gate 3 — Apple Silicon dependency audit

Hardware detected at runtime: **Apple M5 Max, arm64, 18 cores, 128 GB RAM**. Venv Python 3.14.7 (3.12 also available).

- No CUDA index URLs, no `+cu` pins, no `--no-binary`, no platform markers — the file was already tuned for Apple Silicon.
- `torch` was a **floating transitive** (via `sentence-transformers`/`chromadb`) → pinned explicitly (Gate 4).
- No lockfile existed → generated (Gate 4).
- `mlx-lm`/`mlx-embeddings` already Apple-native; `chromadb`/`PyMuPDF` have arm64 wheels; transitive `numpy`/`scipy`/`pandas` link Apple Accelerate.
- Python 3.14 wheel-compatibility was the open question → resolved in Gate 4 (all pins have 3.14 wheels).

## Gate 4 — Controlled install + verification

**Dependency changes applied (user-approved):**
1. `requirements.txt` — added `torch==2.14.0` (pinned the floating transitive; arm64 PyPI wheel is MPS-capable).
2. `requirements.lock` — generated via `uv pip compile requirements.txt -o requirements.lock` (732 lines, full transitive pin set).

**Install:** `UV_COMPILE_BYTECODE=1 uv pip install -r requirements.lock --python .venv/bin/python` — succeeded.

**Verification:**
- `uname -m` → `arm64` ✓
- `torch 2.14.0` → `torch.backends.mps.is_available()` → **True** ✓
- `numpy 2.5.2` → BLAS/LAPACK `name: accelerate` (Apple Accelerate) ✓
- Smoke import of `chroma_store`, `sqlite_store`, all four tools, scraper, ingestion modules → OK ✓
- Ollama daemon running (user's own) ✓
- **Test suite: 125 passed in 4.26s** (pytest 9.1.1, Python 3.14.7) ✓

## How the environment activates

`cd /Users/simgsr/Documents/git_project/bbtc_rag_capstone` → direnv loads `.envrc` → `.venv` activates automatically (`$VIRTUAL_ENV` set, `python` resolves into the venv). First `cd` after a fresh clone creates the venv (slow); subsequent `cd`s just activate it.

## Risks / items left open

- **No data in the repo** — `data/` is all `.gitkeep` sentinels. The SQLite DB, ChromaDB, and staging files are gitignored and must be built by running the pipeline (`make setup` or `python ingest.py` + `bible_ingest`). The eval harness's golden set was verified against a DB that isn't present here.
- **`cloudscraper`** — Cloudflare-bypass library; legitimate for this use, but it is dual-use tooling. Keep it scoped to the BBTC archive.
- **Gradio auth** — the app binds `127.0.0.1` by default (secure). If the user later exposes it on a LAN (`GRADIO_SERVER_NAME=0.0.0.0`), they should set `GRADIO_USERNAME`/`GRADIO_PASSWORD` first.
- **`.envrc` untracked** — safe to commit (no secrets), but the user should decide; if it ever gains secrets, add `.envrc` to `.gitignore`.
- **Python 3.14** — all current pins have 3.14 wheels, but future pin bumps should be re-verified; 3.12 is available as a fallback.
- **Lockfile drift** — `requirements.lock` reflects the resolved set at audit time; re-run `uv pip compile` after any `requirements.txt` change.
