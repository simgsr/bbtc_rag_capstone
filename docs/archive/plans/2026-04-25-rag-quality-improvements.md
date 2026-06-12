# RAG Quality Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stub reranker and weak tooling with real CrossEncoder reranking, filtered semantic search, a grounded system prompt, and a matplotlib tool that queries live data — fixing the main causes of poor RAG response quality.

**Architecture:** The ChromaDB collection already has 8,171 chunks indexed at 768-dim (nomic-embed-text embeddings stored explicitly). All fixes operate at the retrieval and prompt layers — no re-indexing required. CrossEncoder reranking re-scores (query, document) text pairs *after* retrieval, so it is embedding-model-agnostic. The `app.py` Gradio agent is the user-facing entry point; improvements to tools and the system prompt flow directly through it.

**Tech Stack:** Python 3.14, LangChain tools, LangGraph ReAct, ChromaDB, `sentence-transformers` (`cross-encoder/ms-marco-MiniLM-L-6-v2`, already installed), Groq/Gemini/Ollama via `get_llm()`.

---

## Key findings (read before touching code)

- `src/storage/reranker.py` — stub; `rerank()` just slices `candidates[:top_k]`, no model involved.
- `src/tools/vector_tool.py` — no year/speaker filters; returns raw text blocks with no sermon metadata in the output string.
- `app.py` SYSTEM_PROMPT — too short; does not instruct the model to cite sources or explain when to use SQL vs. vector search.
- `src/tools/sql_tool.py` — schema hint missing `date`, `series`, `language`, `file_type`, `url`, `date_scraped` columns; LLM writes broken SQL.
- `src/tools/matplotlib_tool.py` — hardcoded dummy data; bar/pie charts never reflect real DB contents.
- `deploy/backend/api/rag.py` — `RAGPipeline` uses `BAAI/bge-small-en-v1.5` (384-dim) to query a collection stored at 768-dim. It will raise a dimension-mismatch error against real data. Tests pass only because ChromaDB is fully mocked. **Do not wire this class into `app.py`** — it needs a separate re-indexing step first (out of scope here).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/storage/reranker.py` | Modify | Real CrossEncoder scoring |
| `src/tools/vector_tool.py` | Modify | Filters + citation-formatted output |
| `src/tools/sql_tool.py` | Modify | Complete schema hint |
| `src/tools/matplotlib_tool.py` | Modify | Live DB queries for chart data |
| `app.py` | Modify | Better SYSTEM_PROMPT |
| `tests/test_reranker.py` | Create | Unit tests for Reranker |
| `tests/test_vector_tool.py` | Create | Unit tests for vector tool |

---

## Task 1: Real CrossEncoder reranking

**Files:**
- Modify: `src/storage/reranker.py`
- Create: `tests/test_reranker.py`

- [ ] **Step 1.1 — Write the failing tests**

```python
# tests/test_reranker.py
from unittest.mock import patch, MagicMock
from src.storage.reranker import Reranker


def _candidates():
    return [
        {"content": "Jesus is Lord", "metadata": {}, "distance": 0.9},
        {"content": "Grace is unmerited favour", "metadata": {}, "distance": 0.8},
        {"content": "Faith without works is dead", "metadata": {}, "distance": 0.7},
    ]


def test_reranker_returns_top_k():
    with patch("src.storage.reranker.CrossEncoder") as mock_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9, 0.3]
        mock_cls.return_value = mock_model

        r = Reranker()
        results = r.rerank("grace", _candidates(), top_k=2)

    assert len(results) == 2


def test_reranker_orders_by_score_descending():
    with patch("src.storage.reranker.CrossEncoder") as mock_cls:
        mock_model = MagicMock()
        # second candidate should score highest
        mock_model.predict.return_value = [0.2, 0.95, 0.4]
        mock_cls.return_value = mock_model

        r = Reranker()
        results = r.rerank("grace", _candidates(), top_k=3)

    assert results[0]["content"] == "Grace is unmerited favour"


def test_reranker_handles_empty_candidates():
    with patch("src.storage.reranker.CrossEncoder"):
        r = Reranker()
        assert r.rerank("grace", [], top_k=5) == []


def test_reranker_top_k_larger_than_candidates():
    with patch("src.storage.reranker.CrossEncoder") as mock_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9]
        mock_cls.return_value = mock_model

        r = Reranker()
        results = r.rerank("grace", _candidates()[:2], top_k=10)

    assert len(results) == 2
```

- [ ] **Step 1.2 — Run tests to verify they fail**

```bash
cd /Users/simgsr/Documents/structure_db_rag
source .venv/bin/activate
pytest tests/test_reranker.py -v
```

Expected: 4 FAILED (AttributeError or assertion errors — stub reranker has no CrossEncoder import).

- [ ] **Step 1.3 — Implement real CrossEncoder in Reranker**

Replace the entire content of `src/storage/reranker.py`:

```python
from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = _MODEL_NAME):
        self._model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates:
            return []
        pairs = [[query, c["content"]] for c in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]
```

- [ ] **Step 1.4 — Run tests to verify they pass**

```bash
pytest tests/test_reranker.py -v
```

Expected: 4 PASSED.

- [ ] **Step 1.5 — Commit**

```bash
git add src/storage/reranker.py tests/test_reranker.py
git commit -m "feat: add real CrossEncoder reranking to Reranker"
```

---

## Task 2: Vector tool — filters and citation output

**Files:**
- Modify: `src/tools/vector_tool.py`
- Create: `tests/test_vector_tool.py`

- [ ] **Step 2.1 — Write the failing tests**

```python
# tests/test_vector_tool.py
from unittest.mock import MagicMock
from src.tools.vector_tool import make_vector_tool


def _make_store(results):
    store = MagicMock()
    store.search_sermons.return_value = results
    return store


def _sample_results():
    return [
        {
            "content": "God so loved the world.",
            "metadata": {
                "filename": "english_2024_grace.pdf",
                "speaker": "Pastor John",
                "date": "2024-03-10",
                "primary_verse": "John 3:16",
            },
            "distance": 0.1,
        },
        {
            "content": "Faith is the substance of things hoped for.",
            "metadata": {
                "filename": "english_2023_faith.pdf",
                "speaker": "Pastor Mary",
                "date": "2023-06-01",
                "primary_verse": "Hebrews 11:1",
            },
            "distance": 0.2,
        },
    ]


def test_tool_returns_string():
    tool = make_vector_tool(_make_store(_sample_results()))
    result = tool.invoke({"query": "grace"})
    assert isinstance(result, str)


def test_tool_includes_speaker_in_output():
    tool = make_vector_tool(_make_store(_sample_results()))
    result = tool.invoke({"query": "grace"})
    assert "Pastor John" in result


def test_tool_includes_filename_in_output():
    tool = make_vector_tool(_make_store(_sample_results()))
    result = tool.invoke({"query": "grace"})
    assert "english_2024_grace.pdf" in result


def test_tool_passes_year_filter():
    store = _make_store(_sample_results())
    tool = make_vector_tool(store)
    tool.invoke({"query": "grace", "year": 2024})
    store.search_sermons.assert_called_once()
    _, kwargs = store.search_sermons.call_args
    assert kwargs.get("where") == {"year": {"$eq": 2024}}


def test_tool_passes_speaker_filter():
    store = _make_store(_sample_results())
    tool = make_vector_tool(store)
    tool.invoke({"query": "grace", "speaker": "Pastor John"})
    _, kwargs = store.search_sermons.call_args
    assert kwargs.get("where") == {"speaker": {"$eq": "Pastor John"}}


def test_tool_no_results():
    tool = make_vector_tool(_make_store([]))
    result = tool.invoke({"query": "something obscure"})
    assert "No relevant" in result


def test_tool_default_no_filters():
    store = _make_store(_sample_results())
    tool = make_vector_tool(store)
    tool.invoke({"query": "grace"})
    _, kwargs = store.search_sermons.call_args
    assert kwargs.get("where") is None
```

- [ ] **Step 2.2 — Run tests to verify they fail**

```bash
pytest tests/test_vector_tool.py -v
```

Expected: most tests FAILED (tool signature doesn't accept `year`/`speaker`, output doesn't include speaker/filename citations).

- [ ] **Step 2.3 — Implement improved vector tool**

Replace the entire content of `src/tools/vector_tool.py`:

```python
from langchain_core.tools import tool
from src.storage.chroma_store import SermonVectorStore


def make_vector_tool(vector_store: SermonVectorStore):
    @tool
    def search_sermons_tool(query: str, year: int | None = None, speaker: str | None = None) -> str:
        """Searches sermon text for relevant excerpts using semantic similarity.
        Use for 'What did the pastor say about X?' or 'Find sermons about Y'.
        Optionally filter by year (integer, e.g. 2024) or speaker (exact name string).
        Returns excerpts with filename, speaker, date, and verse citations."""

        where: dict | None = None
        if year is not None and speaker:
            where = {"$and": [{"year": {"$eq": year}}, {"speaker": {"$eq": speaker}}]}
        elif year is not None:
            where = {"year": {"$eq": year}}
        elif speaker:
            where = {"speaker": {"$eq": speaker}}

        results = vector_store.search_sermons(query, k=5, where=where)
        if not results:
            return "No relevant sermon excerpts found."

        parts = []
        for res in results:
            m = res["metadata"]
            header = (
                f"[{m.get('filename', 'unknown')} | {m.get('speaker', 'Unknown')} "
                f"| {m.get('date', '')} | {m.get('primary_verse', '')}]"
            )
            parts.append(f"{header}\n{res['content']}")

        return "\n\n---\n\n".join(parts)

    return search_sermons_tool
```

- [ ] **Step 2.4 — Run tests to verify they pass**

```bash
pytest tests/test_vector_tool.py -v
```

Expected: 8 PASSED.

- [ ] **Step 2.5 — Commit**

```bash
git add src/tools/vector_tool.py tests/test_vector_tool.py
git commit -m "feat: add year/speaker filters and citation output to vector tool"
```

---

## Task 3: Better system prompt in app.py

**Files:**
- Modify: `app.py` (only the `SYSTEM_PROMPT` constant, lines ~28–35)

No tests needed for a prompt string — quality is verified by running the app manually.

- [ ] **Step 3.1 — Replace the SYSTEM_PROMPT**

In `app.py`, find the block:

```python
    SYSTEM_PROMPT = (
        "You are the BBTC Sermon & Bible Research Assistant.\n"
        "Use 'sql_query_tool' for stats, 'search_sermons_tool' for sermon content,\n"
        "and 'compare_bible_versions' to compare scripture across translations.\n"
        "CRITICAL: If a user asks for a bible verse, use 'compare_bible_versions' to show NIV/ESV side-by-side.\n"
        "GROUNDING: Answer ONLY using tool data. Never invent facts.\n"
    )
```

Replace with:

```python
    SYSTEM_PROMPT = (
        "You are the BBTC Sermon Intelligence Assistant for Bethesda Bedok-Tampines Church.\n\n"
        "## Tool routing\n"
        "- Use 'sql_query_tool' for: counts, statistics, lists of speakers/years, date lookups, "
        "questions that need numbers (e.g. 'how many sermons', 'top 5 speakers').\n"
        "- Use 'search_sermons_tool' for: questions about sermon *content*, topics, theology, "
        "what a pastor said, summaries of specific sermons. Pass 'year' or 'speaker' filters "
        "when the user specifies them.\n"
        "- Use 'compare_bible_versions' only when the user explicitly asks to compare Bible translations.\n"
        "- Use 'matplotlib_tool' only when the user asks for a chart or visualization.\n\n"
        "## Grounding rules\n"
        "- Answer ONLY from data returned by the tools. Never invent sermon content, speaker names, "
        "dates, or verses.\n"
        "- Every factual claim must cite its source: include the sermon filename and speaker name.\n"
        "- If the tools return no relevant data, say so explicitly — do not guess or fill gaps.\n"
        "- If you need more information to answer precisely, call the relevant tool again with "
        "a refined query before responding.\n"
    )
```

- [ ] **Step 3.2 — Smoke-test manually**

```bash
source .venv/bin/activate
python app.py
```

Open the Gradio UI and test these queries:
1. "How many sermons are in the database?" — should use `sql_query_tool`, return a number.
2. "What did the pastor say about forgiveness?" — should use `search_sermons_tool`, return excerpts with filename/speaker citations.
3. "Who preached the most in 2024?" — should use `sql_query_tool`.

Verify the assistant cites filenames and speakers. If it hallucinates or ignores tools, re-check the SYSTEM_PROMPT for typos.

- [ ] **Step 3.3 — Commit**

```bash
git add app.py
git commit -m "feat: improve system prompt with tool routing and grounding rules"
```

---

## Task 4: Fix SQL tool — complete schema hint

**Files:**
- Modify: `src/tools/sql_tool.py`

- [ ] **Step 4.1 — Replace the schema hint in sql_tool.py**

Find the error-return line:

```python
                schema = "sermons(sermon_id, filename, speaker, bible_book, primary_verse, year)"
                return f"Error: {str(e)}. Please check your column names. The schema is: {schema}"
```

Replace with:

```python
                schema = (
                    "sermons("
                    "sermon_id TEXT PRIMARY KEY, "
                    "filename TEXT, "
                    "url TEXT, "
                    "speaker TEXT, "
                    "date TEXT (YYYY-MM-DD), "
                    "series TEXT, "
                    "bible_book TEXT, "
                    "primary_verse TEXT, "
                    "language TEXT ('English'|'Mandarin'), "
                    "file_type TEXT (pdf|pptx|docx), "
                    "year INTEGER, "
                    "status TEXT (extracted|indexed|failed), "
                    "date_scraped TEXT"
                    ")"
                )
                return f"SQL Error: {str(e)}. Full schema: {schema}"
```

Also update the tool docstring so the LLM sees the schema before making mistakes. Find:

```python
        """Executes a SQL query against the sermons database and returns the result."""
```

Replace with:

```python
        """Executes a SQL query against the sermons SQLite database.
        Schema: sermons(sermon_id TEXT, filename TEXT, url TEXT, speaker TEXT,
        date TEXT YYYY-MM-DD, series TEXT, bible_book TEXT, primary_verse TEXT,
        language TEXT, file_type TEXT, year INTEGER, status TEXT, date_scraped TEXT).
        Returns up to 50 rows. Use COUNT(), GROUP BY, ORDER BY as needed."""
```

- [ ] **Step 4.2 — Quick sanity check**

```bash
source .venv/bin/activate
python -c "
from src.storage.sqlite_store import SermonRegistry
from src.tools.sql_tool import make_sql_tool
r = SermonRegistry()
t = make_sql_tool(r)
print(t.invoke({'query': 'SELECT speaker, COUNT(*) as n FROM sermons GROUP BY speaker ORDER BY n DESC LIMIT 5'}))
"
```

Expected: a list of the top 5 speakers by sermon count.

- [ ] **Step 4.3 — Commit**

```bash
git add src/tools/sql_tool.py
git commit -m "fix: complete SQL tool schema hint and docstring"
```

---

## Task 5: Fix matplotlib tool — live DB data

**Files:**
- Modify: `src/tools/matplotlib_tool.py`

- [ ] **Step 5.1 — Replace matplotlib_tool with real DB queries**

Replace the entire content of `src/tools/matplotlib_tool.py`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sqlite3
import os
import uuid
from langchain_core.tools import tool


def make_matplotlib_tool(registry):
    db_path = registry.db_path

    @tool
    def matplotlib_tool(chart_name: str) -> str:
        """Generates a chart from live sermon data and returns the PNG file path.
        Supported chart_name values:
        - 'sermons_per_speaker' — bar chart of sermon count per speaker (top 10)
        - 'sermons_per_year' — bar chart of sermon count per year
        - 'top_bible_books' — bar chart of most-preached Bible books (top 10)
        Returns the file path to the saved PNG."""
        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            with sqlite3.connect(db_path) as conn:
                if chart_name == "sermons_per_speaker":
                    rows = conn.execute(
                        "SELECT speaker, COUNT(*) as n FROM sermons "
                        "WHERE speaker IS NOT NULL AND speaker != '' "
                        "GROUP BY speaker ORDER BY n DESC LIMIT 10"
                    ).fetchall()
                    if not rows:
                        plt.close(fig)
                        return "No sermon data found."
                    labels, values = zip(*rows)
                    ax.barh(labels, values, color="#3b82f6")
                    ax.set_xlabel("Number of Sermons")
                    ax.set_title("Top 10 Speakers by Sermon Count")
                    ax.invert_yaxis()

                elif chart_name == "sermons_per_year":
                    rows = conn.execute(
                        "SELECT year, COUNT(*) as n FROM sermons "
                        "WHERE year IS NOT NULL "
                        "GROUP BY year ORDER BY year"
                    ).fetchall()
                    if not rows:
                        plt.close(fig)
                        return "No sermon data found."
                    labels, values = zip(*rows)
                    ax.bar([str(y) for y in labels], values, color="#6366f1")
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Number of Sermons")
                    ax.set_title("Sermons per Year")

                elif chart_name == "top_bible_books":
                    rows = conn.execute(
                        "SELECT bible_book, COUNT(*) as n FROM sermons "
                        "WHERE bible_book IS NOT NULL AND bible_book != '' "
                        "GROUP BY bible_book ORDER BY n DESC LIMIT 10"
                    ).fetchall()
                    if not rows:
                        plt.close(fig)
                        return "No sermon data found."
                    labels, values = zip(*rows)
                    ax.barh(labels, values, color="#10b981")
                    ax.set_xlabel("Number of Sermons")
                    ax.set_title("Top 10 Preached Bible Books")
                    ax.invert_yaxis()

                else:
                    plt.close(fig)
                    return (
                        f"Unknown chart '{chart_name}'. "
                        "Valid options: sermons_per_speaker, sermons_per_year, top_bible_books."
                    )

        except Exception as e:
            plt.close(fig)
            return f"Chart generation error: {e}"

        plt.tight_layout()
        file_path = os.path.join("/tmp", f"bbtc_chart_{uuid.uuid4().hex[:8]}.png")
        fig.savefig(file_path)
        plt.close(fig)
        return file_path

    return matplotlib_tool
```

- [ ] **Step 5.2 — Smoke-test the charts**

```bash
source .venv/bin/activate
python -c "
from src.storage.sqlite_store import SermonRegistry
from src.tools.matplotlib_tool import make_matplotlib_tool
r = SermonRegistry()
t = make_matplotlib_tool(r)
for chart in ['sermons_per_speaker', 'sermons_per_year', 'top_bible_books']:
    result = t.invoke({'chart_name': chart})
    print(chart, '->', result)
"
```

Expected: three `/tmp/bbtc_chart_*.png` file paths printed. Open one to verify it shows real data.

- [ ] **Step 5.3 — Commit**

```bash
git add src/tools/matplotlib_tool.py
git commit -m "fix: matplotlib_tool queries live DB instead of hardcoded dummy data"
```

---

## Self-Review

**Spec coverage:**
- Stub reranker → Task 1 ✓
- RAGPipeline not integrated → covered by note in Key Findings (out of scope; bge/nomic dimension mismatch) ✓
- No year/speaker filters → Task 2 ✓
- Weak system prompt → Task 3 ✓
- Incomplete SQL schema → Task 4 ✓
- Hardcoded matplotlib → Task 5 ✓

**Placeholder scan:** None found — all steps include full code.

**Type consistency:**
- `Reranker.rerank(query, candidates, top_k)` — consistent across Task 1 implementation and tests ✓
- `vector_store.search_sermons(query, k=5, where=where)` — matches `SermonVectorStore.search_sermons` signature ✓
- `tool.invoke({"query": ..., "year": ..., "speaker": ...})` — matches new `search_sermons_tool` signature ✓

**Known limitation not in scope:** `deploy/backend/api/rag.py` has a 384-dim vs 768-dim embedding mismatch. It is not wired into `app.py` and should not be until a deliberate re-indexing step is planned separately.
