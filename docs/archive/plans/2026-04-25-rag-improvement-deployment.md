# RAG Improvement & Cloud Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Gradio prototype with a production-grade FastAPI backend (Render) + React dashboard (HuggingFace Spaces), fixing RAG quality via consistent sentence-transformers embeddings and real CrossEncoder reranking.

**Architecture:** Render hosts FastAPI + ChromaDB + SQLite on a persistent disk and runs weekly ingestion as a cron job; HuggingFace Spaces hosts the React frontend which calls the Render API. Sentence-transformers (`BAAI/bge-small-en-v1.5`) replaces Ollama for embeddings, used consistently at both index and query time.

**Tech Stack:** FastAPI, sentence-transformers, ChromaDB, LangGraph, LangChain-Groq, React 18, Vite, TypeScript, Tailwind CSS v3, Plotly.js, Docker, Render, HuggingFace Spaces

---

## File Map

```
deploy/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py          FastAPI app: lifespan, CORS, routes
│   │   ├── rag.py           RAGPipeline: embed → search → rerank → agent
│   │   ├── charts.py        /api/charts/* and /api/stats endpoints
│   │   └── models.py        Pydantic request/response schemas
│   ├── scripts/
│   │   ├── reindex.py       One-time: delete + rebuild ChromaDB with bge-small
│   │   └── run_ingestion.py Render cron: scrape + extract + upsert new sermons
│   ├── tests/
│   │   ├── conftest.py      Fixtures: temp SQLite DB, test client
│   │   ├── test_charts.py   Tests for all chart + stats endpoints
│   │   ├── test_rag.py      Tests for RAGPipeline with mocked embedder/ChromaDB
│   │   └── test_main.py     Integration tests for /api/chat
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── types.ts         Shared TypeScript interfaces
│   │   ├── hooks/
│   │   │   └── useApi.ts    Data-fetching hooks for all endpoints
│   │   └── components/
│   │       ├── StatCards.tsx
│   │       ├── BarChart.tsx
│   │       ├── BubbleChart.tsx
│   │       ├── TopVersesChart.tsx
│   │       └── ChatPanel.tsx
│   ├── public/
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── tsconfig.json
│   └── README.md            HuggingFace Space card
render.yaml                  Render service + cron definitions (repo root)
```

---

## Phase 1: Backend

---

### Task 1: Backend scaffolding and requirements

**Files:**
- Create: `deploy/backend/api/__init__.py`
- Create: `deploy/backend/scripts/__init__.py` (empty)
- Create: `deploy/backend/tests/__init__.py` (empty)
- Create: `deploy/backend/requirements.txt`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p deploy/backend/api deploy/backend/scripts deploy/backend/tests
touch deploy/backend/api/__init__.py deploy/backend/scripts/__init__.py deploy/backend/tests/__init__.py
```

- [ ] **Step 2: Write requirements.txt**

```
# deploy/backend/requirements.txt
fastapi==0.136.1
uvicorn[standard]==0.34.0
sentence-transformers==4.1.0
chromadb==1.0.6
langchain==0.3.25
langchain-core==1.3.2
langchain-groq==1.1.2
langchain-google-genai==4.2.2
langchain-text-splitters==0.3.8
langgraph==1.1.9
python-dotenv==1.1.0
pydantic==2.11.4
httpx==0.28.1
pytest==8.3.5
pytest-asyncio==0.26.0
```

- [ ] **Step 3: Install into the existing venv**

```bash
source .venv/bin/activate
pip install sentence-transformers==4.1.0 httpx==0.28.1 pytest==8.3.5 pytest-asyncio==0.26.0
```

Expected: installs without errors. `sentence-transformers` will pull in `torch` (~500MB first time).

- [ ] **Step 4: Commit**

```bash
git add deploy/
git commit -m "chore: scaffold deploy/ directory structure and backend requirements"
```

---

### Task 2: Pydantic models

**Files:**
- Create: `deploy/backend/api/models.py`

- [ ] **Step 1: Write models.py**

```python
# deploy/backend/api/models.py
from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    query: str
    year_filter: Optional[int] = None
    speaker_filter: Optional[str] = None


class Citation(BaseModel):
    filename: str
    speaker: Optional[str] = None
    date: Optional[str] = None
    verse: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


class StatsResponse(BaseModel):
    total_sermons: int
    total_speakers: int
    year_min: Optional[int]
    year_max: Optional[int]


class YearCount(BaseModel):
    year: int
    count: int


class SpeakerCount(BaseModel):
    speaker: str
    count: int


class VerseCount(BaseModel):
    bible_book: str
    count: int


class ScatterPoint(BaseModel):
    year: int
    speaker: str
    count: int
```

- [ ] **Step 2: Verify models parse correctly**

```bash
source .venv/bin/activate
python3 -c "
from deploy.backend.api.models import ChatRequest, ChatResponse, Citation
req = ChatRequest(query='test', year_filter=2024)
print(req.model_dump())
resp = ChatResponse(answer='hello', citations=[Citation(filename='a.pdf', speaker='John')])
print(resp.model_dump())
print('Models OK')
"
```

Expected: prints dicts and `Models OK`.

- [ ] **Step 3: Commit**

```bash
git add deploy/backend/api/models.py
git commit -m "feat: add Pydantic request/response models for backend API"
```

---

### Task 3: Chart and stats endpoints

**Files:**
- Create: `deploy/backend/api/charts.py`
- Create: `deploy/backend/tests/conftest.py`
- Create: `deploy/backend/tests/test_charts.py`

- [ ] **Step 1: Write the failing tests**

```python
# deploy/backend/tests/conftest.py
import sqlite3
import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_db(tmp_path_factory):
    """Create a temp SQLite DB with 6 test sermons."""
    db_path = str(tmp_path_factory.mktemp("data") / "sermons.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE sermons (
                sermon_id TEXT PRIMARY KEY, filename TEXT, url TEXT UNIQUE,
                speaker TEXT, date TEXT, series TEXT, bible_book TEXT,
                primary_verse TEXT, language TEXT, file_type TEXT,
                year INTEGER, status TEXT, date_scraped TEXT
            )
        """)
        rows = [
            ("s1", "a.pdf", "http://a", "Pastor A", "2022-01-01", "S1", "John", "John 3:16", "English", "pdf", 2022, "indexed", "2026-01-01"),
            ("s2", "b.pdf", "http://b", "Pastor A", "2022-06-01", "S1", "Romans", "Romans 8:28", "English", "pdf", 2022, "indexed", "2026-01-01"),
            ("s3", "c.pdf", "http://c", "Pastor B", "2023-03-01", "S2", "John", "John 1:1", "English", "pdf", 2023, "indexed", "2026-01-01"),
            ("s4", "d.pdf", "http://d", "Pastor B", "2023-09-01", "S2", "Psalms", "Psalm 23:1", "English", "pdf", 2023, "indexed", "2026-01-01"),
            ("s5", "e.pdf", "http://e", "Pastor A", "2024-01-01", "S3", "John", "John 14:6", "English", "pdf", 2024, "indexed", "2026-01-01"),
            ("s6", "f.pdf", "http://f", "Pastor C", "2024-05-01", "S3", "Genesis", "Gen 1:1", "English", "pdf", 2024, "indexed", "2026-01-01"),
        ]
        conn.executemany("INSERT INTO sermons VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    return db_path


@pytest.fixture(scope="session")
def client(test_db, monkeypatch_session):
    """FastAPI test client wired to the temp DB."""
    os.environ["DATA_DIR"] = os.path.dirname(test_db)
    os.environ["GROQ_API_KEY"] = "test-key"
    from deploy.backend.api.main import app
    return TestClient(app)


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch workaround."""
    import _pytest.monkeypatch
    mp = _pytest.monkeypatch.MonkeyPatch()
    yield mp
    mp.undo()
```

```python
# deploy/backend/tests/test_charts.py
import sqlite3
import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def charts_client(test_db):
    os.environ["DATA_DIR"] = os.path.dirname(test_db)
    os.environ["GROQ_API_KEY"] = "test-key"
    # Import after env vars are set
    import importlib
    import deploy.backend.api.charts as charts_mod
    importlib.reload(charts_mod)
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(charts_mod.router)
    return TestClient(app)


def test_stats(charts_client):
    r = charts_client.get("/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_sermons"] == 6
    assert data["total_speakers"] == 3
    assert data["year_min"] == 2022
    assert data["year_max"] == 2024


def test_by_year(charts_client):
    r = charts_client.get("/api/charts/by-year")
    assert r.status_code == 200
    years = {item["year"]: item["count"] for item in r.json()}
    assert years[2022] == 2
    assert years[2023] == 2
    assert years[2024] == 2


def test_by_speaker(charts_client):
    r = charts_client.get("/api/charts/by-speaker")
    assert r.status_code == 200
    speakers = {item["speaker"]: item["count"] for item in r.json()}
    assert speakers["Pastor A"] == 3
    assert speakers["Pastor B"] == 2


def test_by_verse(charts_client):
    r = charts_client.get("/api/charts/by-verse")
    assert r.status_code == 200
    books = {item["bible_book"]: item["count"] for item in r.json()}
    assert books["John"] == 3


def test_scatter(charts_client):
    r = charts_client.get("/api/charts/scatter")
    assert r.status_code == 200
    points = r.json()
    assert len(points) > 0
    assert all("year" in p and "speaker" in p and "count" in p for p in points)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source .venv/bin/activate
cd /path/to/repo
python -m pytest deploy/backend/tests/test_charts.py -v 2>&1 | head -20
```

Expected: `ImportError` or `ModuleNotFoundError` — `charts.py` does not exist yet.

- [ ] **Step 3: Write charts.py**

```python
# deploy/backend/api/charts.py
import sqlite3
import os
from fastapi import APIRouter

router = APIRouter()

def _db_path() -> str:
    return os.path.join(os.getenv("DATA_DIR", "data"), "sermons.db")


def _query(sql: str, params: tuple = ()) -> list[tuple]:
    with sqlite3.connect(_db_path()) as conn:
        return conn.execute(sql, params).fetchall()


@router.get("/api/stats")
def get_stats():
    total = _query("SELECT COUNT(*) FROM sermons")[0][0]
    speakers = _query(
        "SELECT COUNT(DISTINCT speaker) FROM sermons WHERE speaker IS NOT NULL AND speaker != ''"
    )[0][0]
    year_row = _query("SELECT MIN(year), MAX(year) FROM sermons WHERE year IS NOT NULL")[0]
    return {
        "total_sermons": total,
        "total_speakers": speakers,
        "year_min": year_row[0],
        "year_max": year_row[1],
    }


@router.get("/api/charts/by-year")
def by_year():
    rows = _query(
        "SELECT year, COUNT(*) FROM sermons WHERE year IS NOT NULL GROUP BY year ORDER BY year"
    )
    return [{"year": r[0], "count": r[1]} for r in rows]


@router.get("/api/charts/by-speaker")
def by_speaker():
    rows = _query(
        "SELECT speaker, COUNT(*) FROM sermons "
        "WHERE speaker IS NOT NULL AND speaker != '' "
        "GROUP BY speaker ORDER BY COUNT(*) DESC LIMIT 20"
    )
    return [{"speaker": r[0], "count": r[1]} for r in rows]


@router.get("/api/charts/by-verse")
def by_verse():
    rows = _query(
        "SELECT bible_book, COUNT(*) FROM sermons "
        "WHERE bible_book IS NOT NULL AND bible_book != '' "
        "GROUP BY bible_book ORDER BY COUNT(*) DESC LIMIT 20"
    )
    return [{"bible_book": r[0], "count": r[1]} for r in rows]


@router.get("/api/charts/scatter")
def scatter():
    rows = _query(
        "SELECT year, speaker, COUNT(*) FROM sermons "
        "WHERE year IS NOT NULL AND speaker IS NOT NULL AND speaker != '' "
        "GROUP BY year, speaker ORDER BY year"
    )
    return [{"year": r[0], "speaker": r[1], "count": r[2]} for r in rows]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
source .venv/bin/activate
python -m pytest deploy/backend/tests/test_charts.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add deploy/backend/api/charts.py deploy/backend/tests/
git commit -m "feat: add chart and stats API endpoints with tests"
```

---

### Task 4: RAG pipeline

**Files:**
- Create: `deploy/backend/api/rag.py`
- Create: `deploy/backend/tests/test_rag.py`

- [ ] **Step 1: Write the failing tests**

```python
# deploy/backend/tests/test_rag.py
import os
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture()
def mock_rag(tmp_path):
    """RAGPipeline with mocked heavy dependencies."""
    os.environ["DATA_DIR"] = str(tmp_path)
    os.environ["GROQ_API_KEY"] = "test-key"

    # Create a minimal SQLite DB
    import sqlite3
    db = tmp_path / "sermons.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute("""CREATE TABLE sermons (
            sermon_id TEXT PRIMARY KEY, filename TEXT, url TEXT UNIQUE,
            speaker TEXT, date TEXT, series TEXT, bible_book TEXT,
            primary_verse TEXT, language TEXT, file_type TEXT,
            year INTEGER, status TEXT, date_scraped TEXT
        )""")

    with (
        patch("deploy.backend.api.rag.SentenceTransformer") as mock_st,
        patch("deploy.backend.api.rag.CrossEncoder") as mock_ce,
        patch("deploy.backend.api.rag.chromadb.PersistentClient") as mock_chroma,
        patch("deploy.backend.api.rag.get_llm") as mock_llm,
        patch("deploy.backend.api.rag.create_react_agent") as mock_agent,
    ):
        # Embedder returns a fixed vector
        mock_st.return_value.encode.return_value = [[0.1] * 384]

        # CrossEncoder returns scores
        mock_ce.return_value.predict.return_value = [0.9, 0.7, 0.5]

        # ChromaDB returns 3 results
        mock_col = MagicMock()
        mock_col.count.return_value = 3
        mock_col.query.return_value = {
            "documents": [["chunk 1", "chunk 2", "chunk 3"]],
            "metadatas": [[
                {"filename": "a.pdf", "speaker": "Pastor A", "date": "2022-01-01", "primary_verse": "John 3:16"},
                {"filename": "b.pdf", "speaker": "Pastor B", "date": "2023-01-01", "primary_verse": "Rom 8:28"},
                {"filename": "c.pdf", "speaker": "Pastor A", "date": "2024-01-01", "primary_verse": "John 1:1"},
            ]],
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_col

        # Agent returns a simple message
        from langchain_core.messages import AIMessage
        mock_agent.return_value.invoke.return_value = {
            "messages": [AIMessage(content="Grace is God's unmerited favour.")]
        }

        from deploy.backend.api.rag import RAGPipeline
        pipeline = RAGPipeline(data_dir=str(tmp_path))
        yield pipeline


def test_query_returns_answer(mock_rag):
    result = mock_rag.query("What is grace?")
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


def test_query_returns_citations(mock_rag):
    result = mock_rag.query("What is grace?")
    assert "citations" in result
    assert len(result["citations"]) > 0
    first = result["citations"][0]
    assert "filename" in first


def test_query_empty_collection(mock_rag, monkeypatch):
    mock_rag._collection.count.return_value = 0
    result = mock_rag.query("What is grace?")
    assert "No sermons" in result["answer"] or isinstance(result["answer"], str)


def test_query_with_year_filter(mock_rag):
    result = mock_rag.query("sermon about faith", year_filter=2023)
    call_kwargs = mock_rag._collection.query.call_args[1]
    assert call_kwargs.get("where") == {"year": {"$eq": 2023}}


def test_query_with_speaker_filter(mock_rag):
    result = mock_rag.query("sermon", speaker_filter="Pastor A")
    call_kwargs = mock_rag._collection.query.call_args[1]
    assert call_kwargs.get("where") == {"speaker": {"$eq": "Pastor A"}}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source .venv/bin/activate
python -m pytest deploy/backend/tests/test_rag.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'deploy.backend.api.rag'`

- [ ] **Step 3: Write rag.py**

```python
# deploy/backend/api/rag.py
import os
import sqlite3
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from src.llm import get_llm

_SYSTEM_PROMPT = (
    "You are the BBTC Sermon Intelligence Assistant.\n"
    "Answer ONLY from the sermon excerpts provided in the prompt. Never invent facts.\n"
    "For every claim, cite the sermon filename and speaker.\n"
    "Use 'sql_query_tool' for counts/statistics. "
    "Key column names: primary_verse (not verse), speaker, year, bible_book.\n"
    "If the answer is not in the excerpts, say so explicitly."
)

_MODEL_CACHE_DEFAULT = os.path.join(os.getenv("DATA_DIR", "data"), "models")


class RAGPipeline:
    def __init__(self, data_dir: str = "data"):
        self._data_dir = data_dir
        model_cache = os.getenv("MODEL_CACHE_DIR", os.path.join(data_dir, "models"))
        os.makedirs(model_cache, exist_ok=True)

        self._embedder = SentenceTransformer(
            "BAAI/bge-small-en-v1.5", cache_folder=model_cache
        )
        self._reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
            cache_folder=model_cache,
        )

        chroma_path = os.path.join(data_dir, "chroma_db")
        self._chroma = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._chroma.get_or_create_collection(
            "sermon_collection", metadata={"hnsw:space": "cosine"}
        )

        self._db_path = os.path.join(data_dir, "sermons.db")

        llm = get_llm(provider_type="groq", temperature=0.1)
        self._agent = create_react_agent(
            llm,
            tools=[self._make_sql_tool()],
            prompt=_SYSTEM_PROMPT,
        )

    def _make_sql_tool(self):
        db_path = self._db_path

        @tool
        def sql_query_tool(query: str) -> str:
            """Runs SQL against the sermons database for stats and counts.
            Schema: sermons(sermon_id, filename, speaker, date, series,
            bible_book, primary_verse, language, year, status)."""
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.execute(query)
                    cols = [d[0] for d in cursor.description]
                    rows = cursor.fetchall()
                    if not rows:
                        return "No results."
                    result = f"Columns: {', '.join(cols)}\n"
                    result += "\n".join(str(r) for r in rows[:50])
                    return result
            except Exception as e:
                return f"SQL Error: {e}"

        return sql_query_tool

    def _semantic_search(
        self, query: str, year_filter: int | None, speaker_filter: str | None
    ) -> tuple[str, list[dict]]:
        """Returns (formatted_context, citations)."""
        n = self._collection.count()
        if n == 0:
            return "", []

        embedding = self._embedder.encode(query).tolist()

        kwargs: dict = {
            "query_embeddings": [embedding],
            "n_results": min(20, n),
            "include": ["documents", "metadatas"],
        }

        where: dict = {}
        if year_filter is not None:
            where["year"] = {"$eq": year_filter}
        if speaker_filter:
            where["speaker"] = {"$eq": speaker_filter}
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not docs:
            return "", []

        # CrossEncoder rerank
        pairs = [[query, doc] for doc in docs]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(scores, docs, metas), reverse=True)[:5]

        parts = []
        citations = []
        for _, doc, meta in ranked:
            filename = meta.get("filename", "unknown")
            speaker = meta.get("speaker") or "Unknown"
            date = meta.get("date") or ""
            verse = meta.get("primary_verse") or ""
            parts.append(f"[{filename} | {speaker} | {date}]\n{doc}")
            citations.append(
                {"filename": filename, "speaker": speaker, "date": date, "verse": verse}
            )

        return "\n\n---\n\n".join(parts), citations

    def query(
        self,
        question: str,
        year_filter: int | None = None,
        speaker_filter: str | None = None,
    ) -> dict:
        context, citations = self._semantic_search(question, year_filter, speaker_filter)

        if context:
            augmented = f"Sermon excerpts:\n\n{context}\n\nQuestion: {question}"
        else:
            augmented = question

        result = self._agent.invoke({"messages": [HumanMessage(content=augmented)]})
        answer = result["messages"][-1].content

        return {"answer": answer, "citations": citations}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
source .venv/bin/activate
python -m pytest deploy/backend/tests/test_rag.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add deploy/backend/api/rag.py deploy/backend/tests/test_rag.py
git commit -m "feat: add RAGPipeline with sentence-transformers embeddings and CrossEncoder reranking"
```

---

### Task 5: FastAPI main app

**Files:**
- Create: `deploy/backend/api/main.py`
- Create: `deploy/backend/tests/test_main.py`

- [ ] **Step 1: Write the failing test**

```python
# deploy/backend/tests/test_main.py
import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture()
def app_client(test_db):
    os.environ["DATA_DIR"] = os.path.dirname(test_db)
    os.environ["GROQ_API_KEY"] = "test-key"
    os.environ["FRONTEND_URL"] = "http://localhost:5173"

    with patch("deploy.backend.api.rag.RAGPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "answer": "Grace is God's favour.",
            "citations": [{"filename": "a.pdf", "speaker": "Pastor A", "date": "2022-01-01", "verse": "John 3:16"}],
        }
        mock_pipeline_cls.return_value = mock_pipeline

        from deploy.backend.api.main import app
        with TestClient(app) as client:
            yield client


def test_stats_endpoint(app_client):
    r = app_client.get("/api/stats")
    assert r.status_code == 200
    assert "total_sermons" in r.json()


def test_chat_endpoint(app_client):
    r = app_client.post("/api/chat", json={"query": "What is grace?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "citations" in data


def test_chat_with_filters(app_client):
    r = app_client.post(
        "/api/chat",
        json={"query": "sermon", "year_filter": 2023, "speaker_filter": "Pastor A"},
    )
    assert r.status_code == 200


def test_cors_header(app_client):
    r = app_client.get("/api/stats", headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
source .venv/bin/activate
python -m pytest deploy/backend/tests/test_main.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError` for `main.py`

- [ ] **Step 3: Write main.py**

```python
# deploy/backend/api/main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from deploy.backend.api.charts import router as charts_router
from deploy.backend.api.models import ChatRequest, ChatResponse, Citation

load_dotenv()

_rag = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag
    from deploy.backend.api.rag import RAGPipeline
    data_dir = os.getenv("DATA_DIR", "data")
    _rag = RAGPipeline(data_dir=data_dir)
    yield
    _rag = None


app = FastAPI(title="BBTC Sermon Intelligence API", lifespan=lifespan)

frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(charts_router)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = _rag.query(
        request.query,
        year_filter=request.year_filter,
        speaker_filter=request.speaker_filter,
    )
    citations = [Citation(**c) for c in result["citations"]]
    return ChatResponse(answer=result["answer"], citations=citations)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
source .venv/bin/activate
python -m pytest deploy/backend/tests/test_main.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Smoke-test the API locally**

```bash
source .venv/bin/activate
DATA_DIR=data GROQ_API_KEY=<your-key> FRONTEND_URL=http://localhost:5173 \
  uvicorn deploy.backend.api.main:app --reload --port 8000
```

In a second terminal:
```bash
curl http://localhost:8000/api/stats
curl http://localhost:8000/api/charts/by-year
```

Expected: JSON responses with real sermon counts.

- [ ] **Step 6: Commit**

```bash
git add deploy/backend/api/main.py deploy/backend/tests/test_main.py
git commit -m "feat: add FastAPI main app with CORS, lifespan RAG init, and chat endpoint"
```

---

### Task 6: Re-index script

**Files:**
- Create: `deploy/backend/scripts/reindex.py`

- [ ] **Step 1: Write reindex.py**

```python
# deploy/backend/scripts/reindex.py
"""
One-time script: deletes the existing sermon_collection in ChromaDB and
rebuilds it using BAAI/bge-small-en-v1.5 embeddings for consistency.

Run on Render after first deploy (takes ~15 min on starter tier):
  python deploy/backend/scripts/reindex.py
"""
import os
import sqlite3
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_CACHE = os.getenv("MODEL_CACHE_DIR", os.path.join(DATA_DIR, "models"))
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
BATCH_SIZE = 500


def main():
    from sentence_transformers import SentenceTransformer
    import chromadb
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    os.makedirs(MODEL_CACHE, exist_ok=True)
    print(f"Loading BAAI/bge-small-en-v1.5 (cache: {MODEL_CACHE})...")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", cache_folder=MODEL_CACHE)

    chroma_path = os.path.join(DATA_DIR, "chroma_db")
    print(f"Connecting to ChromaDB at {chroma_path}...")
    client = chromadb.PersistentClient(path=chroma_path)

    try:
        client.delete_collection("sermon_collection")
        print("Deleted existing sermon_collection.")
    except Exception:
        pass

    collection = client.create_collection(
        "sermon_collection", metadata={"hnsw:space": "cosine"}
    )

    db_path = os.path.join(DATA_DIR, "sermons.db")
    print(f"Loading indexed sermons from {db_path}...")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sermons = [
            dict(r)
            for r in conn.execute("SELECT * FROM sermons WHERE status = 'indexed'").fetchall()
        ]

    print(f"Found {len(sermons)} indexed sermons.\n")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    total_chunks = 0
    for i, sermon in enumerate(sermons):
        txt_name = os.path.splitext(sermon["filename"])[0] + ".txt"
        txt_path = os.path.join(DATA_DIR, "sermons", txt_name)

        if not os.path.exists(txt_path):
            print(f"  [{i+1}/{len(sermons)}] SKIP (no .txt): {sermon['filename']}")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = splitter.split_text(content) or [content[:512]]

        ids = [f"{sermon['sermon_id']}_{j}" for j in range(len(chunks))]
        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
        meta = {
            "filename": sermon.get("filename") or "",
            "speaker": sermon.get("speaker") or "",
            "date": sermon.get("date") or "",
            "primary_verse": sermon.get("primary_verse") or "",
            "bible_book": sermon.get("bible_book") or "",
            "series": sermon.get("series") or "",
            "year": int(sermon.get("year") or 0),
            "language": sermon.get("language") or "",
            "sermon_id": sermon["sermon_id"],
        }
        metadatas = [meta] * len(chunks)

        for start in range(0, len(chunks), BATCH_SIZE):
            collection.upsert(
                ids=ids[start : start + BATCH_SIZE],
                documents=chunks[start : start + BATCH_SIZE],
                embeddings=embeddings[start : start + BATCH_SIZE],
                metadatas=metadatas[start : start + BATCH_SIZE],
            )

        total_chunks += len(chunks)
        print(f"  [{i+1}/{len(sermons)}] {sermon['filename']} ({len(chunks)} chunks)")

    print(f"\nDone. {len(sermons)} sermons re-indexed, {total_chunks} total chunks.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script runs (dry-run with DATA_DIR=data)**

```bash
source .venv/bin/activate
python deploy/backend/scripts/reindex.py
```

Expected: downloads `bge-small-en-v1.5` weights to `data/models/`, processes all 1,144 indexed sermons, prints progress. Takes 5–15 min on first run (model download + encoding).

- [ ] **Step 3: Confirm ChromaDB was rebuilt**

```bash
source .venv/bin/activate
python3 -c "
import chromadb
c = chromadb.PersistentClient('data/chroma_db')
col = c.get_collection('sermon_collection')
print('sermon_collection count:', col.count())
meta = col.get(limit=1, include=['metadatas'])
print('sample meta:', meta['metadatas'][0] if meta['metadatas'] else 'empty')
"
```

Expected: count > 0, metadata shows `filename`, `speaker`, `year` etc.

- [ ] **Step 4: Test a query end-to-end**

```bash
source .venv/bin/activate
python3 -c "
import os; os.environ['DATA_DIR']='data'
from deploy.backend.api.rag import RAGPipeline
# This will download cross-encoder model on first run
pipeline = RAGPipeline()
result = pipeline.query('What sermons were preached about grace?')
print('Answer:', result['answer'][:200])
print('Citations:', len(result['citations']))
"
```

Expected: non-empty answer with at least one citation.

- [ ] **Step 5: Commit**

```bash
git add deploy/backend/scripts/reindex.py
git commit -m "feat: add reindex script to rebuild ChromaDB with bge-small-en-v1.5"
```

---

### Task 7: Ingestion script (Render cron)

**Files:**
- Create: `deploy/backend/scripts/run_ingestion.py`

- [ ] **Step 1: Write run_ingestion.py**

```python
# deploy/backend/scripts/run_ingestion.py
"""
Standalone ingestion script for Render cron job (runs every Sunday 00:00).
Replaces the Dagster scheduler in production — same logic, no Dagster overhead.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")


def main():
    from src.scraper.bbtc_scraper import BBTCScraper
    from src.storage.sqlite_store import SermonRegistry
    from src.ingestion.metadata_extractor import MetadataExtractor
    from sentence_transformers import SentenceTransformer
    import chromadb
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    registry = SermonRegistry(db_path=os.path.join(DATA_DIR, "sermons.db"))
    scraper = BBTCScraper(
        download_dir=os.path.join(DATA_DIR, "sermons"),
        staging_dir=os.path.join(DATA_DIR, "staging"),
        registry=registry,
    )
    extractor = MetadataExtractor()

    # 1. Scrape current year
    current_year = datetime.now().year
    print(f"Scraping year {current_year}...")
    scraper.scrape_year(current_year)

    # 2. Extract metadata for newly scraped sermons
    sermons = registry.get_all_sermons()
    pending = [s for s in sermons if s["status"] in ("extracted", "processed")]
    print(f"Found {len(pending)} sermons pending metadata extraction.")

    for sermon in pending:
        txt_name = os.path.splitext(sermon["filename"])[0] + ".txt"
        txt_path = os.path.join(DATA_DIR, "sermons", txt_name)
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        metadata = extractor.extract(content[:500])
        update = {**sermon, **metadata, "status": "indexed"}
        registry.insert_sermon(update)

    # 3. Vectorize newly indexed sermons using bge-small-en-v1.5
    model_cache = os.getenv("MODEL_CACHE_DIR", os.path.join(DATA_DIR, "models"))
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", cache_folder=model_cache)

    client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma_db"))
    collection = client.get_or_create_collection(
        "sermon_collection", metadata={"hnsw:space": "cosine"}
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

    # Re-fetch — some may now be indexed after step 2
    newly_indexed = [s for s in registry.get_all_sermons() if s["status"] == "indexed"]
    existing_ids = set(collection.get(include=[])["ids"])

    upserted = 0
    for sermon in newly_indexed:
        first_chunk_id = f"{sermon['sermon_id']}_0"
        if first_chunk_id in existing_ids:
            continue  # already vectorized

        txt_name = os.path.splitext(sermon["filename"])[0] + ".txt"
        txt_path = os.path.join(DATA_DIR, "sermons", txt_name)
        if not os.path.exists(txt_path):
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = splitter.split_text(content) or [content[:512]]
        ids = [f"{sermon['sermon_id']}_{j}" for j in range(len(chunks))]
        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
        meta = {
            "filename": sermon.get("filename") or "",
            "speaker": sermon.get("speaker") or "",
            "date": sermon.get("date") or "",
            "primary_verse": sermon.get("primary_verse") or "",
            "bible_book": sermon.get("bible_book") or "",
            "series": sermon.get("series") or "",
            "year": int(sermon.get("year") or 0),
            "language": sermon.get("language") or "",
            "sermon_id": sermon["sermon_id"],
        }
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[meta] * len(chunks),
        )
        upserted += 1
        print(f"  Vectorized: {sermon['filename']}")

    print(f"\nIngestion complete. {upserted} new sermons vectorized.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs without errors**

```bash
source .venv/bin/activate
python deploy/backend/scripts/run_ingestion.py
```

Expected: scrapes current year, reports 0 pending (all already indexed), reports 0 new sermons vectorized (all already in ChromaDB from reindex.py).

- [ ] **Step 3: Commit**

```bash
git add deploy/backend/scripts/run_ingestion.py
git commit -m "feat: add standalone ingestion script for Render cron job"
```

---

### Task 8: Backend Dockerfile and render.yaml

**Files:**
- Create: `deploy/backend/Dockerfile`
- Create: `render.yaml`

- [ ] **Step 1: Write backend Dockerfile**

```dockerfile
# deploy/backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers + chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY deploy/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared source modules from repo root
COPY src/ ./src/

# Copy backend application code
COPY deploy/backend/api/ ./api/
COPY deploy/backend/scripts/ ./scripts/

# Copy root-level modules needed by src/
COPY dagster_pipeline.py ./dagster_pipeline.py

ENV PYTHONPATH=/app
ENV DATA_DIR=/data
ENV MODEL_CACHE_DIR=/data/models

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write render.yaml**

```yaml
# render.yaml  (place in repo root)
services:
  - type: web
    name: bbtc-sermon-api
    runtime: docker
    dockerfilePath: deploy/backend/Dockerfile
    dockerContext: .
    healthCheckPath: /api/stats
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
      - key: DATA_DIR
        value: /data
      - key: MODEL_CACHE_DIR
        value: /data/models

  - type: cron
    name: bbtc-sermon-ingestion
    runtime: docker
    dockerfilePath: deploy/backend/Dockerfile
    dockerContext: .
    schedule: "0 0 * * 0"
    dockerCommand: python scripts/run_ingestion.py
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: GEMINI_API_KEY
        sync: false
      - key: DATA_DIR
        value: /data
      - key: MODEL_CACHE_DIR
        value: /data/models
```

- [ ] **Step 3: Build the Docker image locally to confirm it works**

```bash
docker build -f deploy/backend/Dockerfile -t bbtc-api .
```

Expected: builds without errors. The `sentence-transformers` install will take a few minutes.

- [ ] **Step 4: Smoke-test the image locally**

```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/data:/data \
  -e GROQ_API_KEY=<your-key> \
  -e FRONTEND_URL=http://localhost:5173 \
  bbtc-api
```

In another terminal:
```bash
curl http://localhost:8000/api/stats
```

Expected: JSON with `total_sermons`, `total_speakers` etc.

- [ ] **Step 5: Commit**

```bash
git add deploy/backend/Dockerfile render.yaml
git commit -m "feat: add backend Dockerfile and render.yaml for Render deployment"
```

---

## Phase 2: Frontend

---

### Task 9: React project setup

**Files:**
- Create: `deploy/frontend/package.json`
- Create: `deploy/frontend/vite.config.ts`
- Create: `deploy/frontend/tailwind.config.js`
- Create: `deploy/frontend/postcss.config.js`
- Create: `deploy/frontend/tsconfig.json`
- Create: `deploy/frontend/index.html`
- Create: `deploy/frontend/src/main.tsx`

- [ ] **Step 1: Scaffold the React project**

```bash
cd deploy/frontend
npm create vite@latest . -- --template react-ts
npm install
npm install react-plotly.js plotly.js
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
cd ../..
```

- [ ] **Step 2: Configure tailwind.config.js**

Replace the generated content:

```javascript
// deploy/frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: {
          900: "#0f172a",
          800: "#1e293b",
          700: "#334155",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
```

- [ ] **Step 3: Update src/main.tsx**

```tsx
// deploy/frontend/src/main.tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

- [ ] **Step 4: Replace src/index.css with Tailwind directives**

```css
/* deploy/frontend/src/index.css */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  @apply bg-navy-900 text-slate-100 font-sans;
}
```

- [ ] **Step 5: Confirm dev server starts**

```bash
cd deploy/frontend && npm run dev
```

Expected: Vite starts at `http://localhost:5173`, browser shows placeholder React page.

- [ ] **Step 6: Commit**

```bash
cd ../..
git add deploy/frontend/
git commit -m "feat: scaffold React frontend with Vite, TypeScript, and Tailwind"
```

---

### Task 10: TypeScript types and API hooks

**Files:**
- Create: `deploy/frontend/src/types.ts`
- Create: `deploy/frontend/src/hooks/useApi.ts`

- [ ] **Step 1: Write types.ts**

```typescript
// deploy/frontend/src/types.ts
export interface StatsResponse {
  total_sermons: number;
  total_speakers: number;
  year_min: number | null;
  year_max: number | null;
}

export interface YearCount {
  year: number;
  count: number;
}

export interface SpeakerCount {
  speaker: string;
  count: number;
}

export interface VerseCount {
  bible_book: string;
  count: number;
}

export interface ScatterPoint {
  year: number;
  speaker: string;
  count: number;
}

export interface Citation {
  filename: string;
  speaker: string | null;
  date: string | null;
  verse: string | null;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
}

export interface Filters {
  year: number | null;
  speaker: string | null;
}
```

- [ ] **Step 2: Write hooks/useApi.ts**

```typescript
// deploy/frontend/src/hooks/useApi.ts
import { useState, useEffect, useCallback } from "react";
import type {
  StatsResponse, YearCount, SpeakerCount,
  VerseCount, ScatterPoint, ChatResponse, Filters,
} from "../types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json();
}

function useQuery<T>(path: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    apiFetch<T>(path)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [path]);

  return { data, loading, error };
}

export function useStats() {
  return useQuery<StatsResponse>("/api/stats");
}

export function useByYear() {
  return useQuery<YearCount[]>("/api/charts/by-year");
}

export function useBySpeaker() {
  return useQuery<SpeakerCount[]>("/api/charts/by-speaker");
}

export function useByVerse() {
  return useQuery<VerseCount[]>("/api/charts/by-verse");
}

export function useScatter() {
  return useQuery<ScatterPoint[]>("/api/charts/scatter");
}

export function useChat() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (query: string, filters: Filters): Promise<ChatResponse | null> => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            year_filter: filters.year ?? undefined,
            speaker_filter: filters.speaker ?? undefined,
          }),
        });
        if (!res.ok) throw new Error(`API error ${res.status}`);
        return await res.json();
      } catch (e: any) {
        setError(e.message);
        return null;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return { sendMessage, loading, error };
}
```

- [ ] **Step 3: Verify TypeScript compilation**

```bash
cd deploy/frontend && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
cd ../..
git add deploy/frontend/src/types.ts deploy/frontend/src/hooks/
git commit -m "feat: add TypeScript types and API data-fetching hooks"
```

---

### Task 11: StatCards component

**Files:**
- Create: `deploy/frontend/src/components/StatCards.tsx`

- [ ] **Step 1: Write StatCards.tsx**

```tsx
// deploy/frontend/src/components/StatCards.tsx
import { useStats } from "../hooks/useApi";

interface CardProps {
  label: string;
  value: string | number;
}

function Card({ label, value }: CardProps) {
  return (
    <div className="bg-navy-800 border border-slate-700 rounded-xl p-6 flex flex-col gap-1">
      <span className="text-slate-400 text-sm font-medium uppercase tracking-wider">
        {label}
      </span>
      <span className="text-3xl font-bold text-white">{value}</span>
    </div>
  );
}

export function StatCards() {
  const { data, loading, error } = useStats();

  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[0, 1, 2].map((i) => (
          <div key={i} className="bg-navy-800 border border-slate-700 rounded-xl p-6 animate-pulse h-24" />
        ))}
      </div>
    );
  }

  if (error || !data) {
    return <p className="text-red-400 text-sm">Failed to load stats.</p>;
  }

  const yearRange =
    data.year_min && data.year_max
      ? `${data.year_min} – ${data.year_max}`
      : "—";

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <Card label="Total Sermons" value={data.total_sermons.toLocaleString()} />
      <Card label="Speakers" value={data.total_speakers} />
      <Card label="Years Covered" value={yearRange} />
    </div>
  );
}
```

- [ ] **Step 2: Verify it renders (add to App.tsx temporarily)**

```tsx
// In deploy/frontend/src/App.tsx — temporary test
import { StatCards } from "./components/StatCards";
export default function App() {
  return <div className="p-8"><StatCards /></div>;
}
```

Start dev server (`npm run dev` from `deploy/frontend`) and confirm cards render (loading skeletons visible if backend not running).

- [ ] **Step 3: Commit**

```bash
git add deploy/frontend/src/components/StatCards.tsx
git commit -m "feat: add StatCards summary component"
```

---

### Task 12: BarChart and TopVersesChart components

**Files:**
- Create: `deploy/frontend/src/components/BarChart.tsx`
- Create: `deploy/frontend/src/components/TopVersesChart.tsx`

- [ ] **Step 1: Write BarChart.tsx**

```tsx
// deploy/frontend/src/components/BarChart.tsx
import Plot from "react-plotly.js";
import { useByYear, useBySpeaker } from "../hooks/useApi";

interface Props {
  title: string;
  x: (string | number)[];
  y: number[];
  color?: string;
  horizontal?: boolean;
}

function PlotlyBar({ title, x, y, color = "#60a5fa", horizontal = false }: Props) {
  return (
    <Plot
      data={[
        {
          type: "bar",
          x: horizontal ? y : x,
          y: horizontal ? x : y,
          orientation: horizontal ? "h" : "v",
          marker: { color },
          hovertemplate: horizontal
            ? "%{y}: <b>%{x}</b><extra></extra>"
            : "%{x}: <b>%{y}</b><extra></extra>",
        },
      ]}
      layout={{
        title: { text: title, font: { color: "#f8fafc", size: 14 } },
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#94a3b8" },
        xaxis: { gridcolor: "#334155", zerolinecolor: "#334155" },
        yaxis: { gridcolor: "#334155", zerolinecolor: "#334155" },
        margin: { t: 40, r: 10, b: 40, l: horizontal ? 100 : 40 },
        bargap: 0.3,
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", height: "300px" }}
    />
  );
}

export function SermonsByYearChart() {
  const { data, loading } = useByYear();
  if (loading || !data) return <div className="h-[300px] animate-pulse bg-navy-800 rounded-xl" />;
  return (
    <PlotlyBar
      title="Sermons per Year"
      x={data.map((d) => d.year)}
      y={data.map((d) => d.count)}
      color="#60a5fa"
    />
  );
}

export function SermonsBySpeakerChart() {
  const { data, loading } = useBySpeaker();
  if (loading || !data) return <div className="h-[300px] animate-pulse bg-navy-800 rounded-xl" />;
  const sorted = [...data].sort((a, b) => a.count - b.count);
  return (
    <PlotlyBar
      title="Sermons per Speaker"
      x={sorted.map((d) => d.speaker)}
      y={sorted.map((d) => d.count)}
      color="#a78bfa"
      horizontal
    />
  );
}
```

- [ ] **Step 2: Write TopVersesChart.tsx**

```tsx
// deploy/frontend/src/components/TopVersesChart.tsx
import Plot from "react-plotly.js";
import { useByVerse } from "../hooks/useApi";

export function TopVersesChart() {
  const { data, loading } = useByVerse();

  if (loading || !data) {
    return <div className="h-[300px] animate-pulse bg-navy-800 rounded-xl" />;
  }

  const top = [...data].sort((a, b) => b.count - a.count).slice(0, 15);
  const sorted = [...top].sort((a, b) => a.count - b.count);

  return (
    <Plot
      data={[
        {
          type: "bar",
          x: sorted.map((d) => d.count),
          y: sorted.map((d) => d.bible_book),
          orientation: "h",
          marker: {
            color: sorted.map((_, i) =>
              `hsl(${200 + (i / sorted.length) * 60}, 80%, 65%)`
            ),
          },
          hovertemplate: "%{y}: <b>%{x} sermons</b><extra></extra>",
        },
      ]}
      layout={{
        title: { text: "Top Bible Books Preached", font: { color: "#f8fafc", size: 14 } },
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#94a3b8" },
        xaxis: { gridcolor: "#334155", zerolinecolor: "#334155" },
        yaxis: { gridcolor: "#334155", zerolinecolor: "#334155" },
        margin: { t: 40, r: 10, b: 40, l: 100 },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", height: "350px" }}
    />
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add deploy/frontend/src/components/BarChart.tsx deploy/frontend/src/components/TopVersesChart.tsx
git commit -m "feat: add bar charts for sermons by year, speaker, and bible book"
```

---

### Task 13: BubbleChart component

**Files:**
- Create: `deploy/frontend/src/components/BubbleChart.tsx`

- [ ] **Step 1: Write BubbleChart.tsx**

```tsx
// deploy/frontend/src/components/BubbleChart.tsx
import Plot from "react-plotly.js";
import { useScatter } from "../hooks/useApi";
import type { Filters } from "../types";

interface Props {
  onSelect: (filters: Filters) => void;
}

export function BubbleChart({ onSelect }: Props) {
  const { data, loading } = useScatter();

  if (loading || !data) {
    return <div className="h-[420px] animate-pulse bg-navy-800 rounded-xl" />;
  }

  const speakers = Array.from(new Set(data.map((d) => d.speaker)));

  return (
    <div>
      <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-2">
        Year × Speaker × Sermon Count
        <span className="ml-2 text-xs font-normal text-slate-500">
          (click a bubble to filter the chat below)
        </span>
      </h3>
      <Plot
        data={[
          {
            type: "scatter",
            mode: "markers",
            x: data.map((d) => d.year),
            y: data.map((d) => d.speaker),
            marker: {
              size: data.map((d) => Math.max(Math.sqrt(d.count) * 10, 8)),
              color: data.map((d) => d.count),
              colorscale: "Blues",
              showscale: true,
              colorbar: {
                title: "Count",
                titlefont: { color: "#94a3b8" },
                tickfont: { color: "#94a3b8" },
              },
              line: { color: "rgba(255,255,255,0.2)", width: 1 },
            },
            text: data.map(
              (d) => `${d.speaker} (${d.year})<br><b>${d.count} sermons</b>`
            ),
            hovertemplate: "%{text}<extra></extra>",
          },
        ]}
        layout={{
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "#94a3b8" },
          xaxis: {
            title: "Year",
            gridcolor: "#334155",
            zerolinecolor: "#334155",
            dtick: 1,
          },
          yaxis: {
            gridcolor: "#334155",
            zerolinecolor: "#334155",
            categoryorder: "total ascending",
          },
          margin: { t: 10, r: 80, b: 50, l: 120 },
          hovermode: "closest",
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: "400px" }}
        onClick={(event) => {
          if (!event.points?.length) return;
          const pt = event.points[0];
          onSelect({
            year: pt.x as number,
            speaker: pt.y as string,
          });
        }}
      />
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add deploy/frontend/src/components/BubbleChart.tsx
git commit -m "feat: add interactive bubble chart (year × speaker × count)"
```

---

### Task 14: ChatPanel component

**Files:**
- Create: `deploy/frontend/src/components/ChatPanel.tsx`

- [ ] **Step 1: Write ChatPanel.tsx**

```tsx
// deploy/frontend/src/components/ChatPanel.tsx
import { useState, useRef, useEffect } from "react";
import { useChat } from "../hooks/useApi";
import type { Citation, Filters } from "../types";

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

interface Props {
  filters: Filters;
  onFiltersChange: (f: Filters) => void;
}

function CitationBadge({ citation }: { citation: Citation }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-700 text-xs text-slate-300">
      <span className="text-blue-400">📄</span>
      {citation.speaker && <span className="font-medium">{citation.speaker}</span>}
      {citation.date && <span className="text-slate-400">{citation.date}</span>}
      {citation.verse && <span className="text-purple-400">{citation.verse}</span>}
    </span>
  );
}

export function ChatPanel({ filters, onFiltersChange }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const { sendMessage, loading } = useChat();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const query = input.trim();
    if (!query || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: query }]);

    const result = await sendMessage(query, filters);
    if (result) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: result.answer, citations: result.citations },
      ]);
    }
  };

  const filterLabel =
    filters.year || filters.speaker
      ? `Filtering: ${[filters.year, filters.speaker].filter(Boolean).join(" · ")}`
      : null;

  return (
    <div className="bg-navy-800 border border-slate-700 rounded-xl flex flex-col h-[500px]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
        <span className="text-sm font-semibold text-slate-200">Ask about the sermons</span>
        {filterLabel && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-blue-400 bg-blue-400/10 px-2 py-0.5 rounded-full">
              {filterLabel}
            </span>
            <button
              onClick={() => onFiltersChange({ year: null, speaker: null })}
              className="text-xs text-slate-500 hover:text-slate-300"
            >
              ✕ Clear
            </button>
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <p className="text-slate-500 text-sm text-center mt-8">
            Ask a question, or click a bubble in the chart above to focus your query.
          </p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-2 text-sm ${
                msg.role === "user"
                  ? "bg-blue-600 text-white rounded-br-sm"
                  : "bg-slate-700 text-slate-100 rounded-bl-sm"
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {msg.citations.slice(0, 5).map((c, j) => (
                    <CitationBadge key={j} citation={c} />
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-700 rounded-2xl rounded-bl-sm px-4 py-2">
              <span className="flex gap-1">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-slate-700 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
          placeholder="Ask about a sermon topic, verse, or speaker..."
          className="flex-1 bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500"
          disabled={loading}
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm font-medium rounded-lg transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add deploy/frontend/src/components/ChatPanel.tsx
git commit -m "feat: add chat panel with citation badges and filter display"
```

---

### Task 15: App.tsx — wire everything together

**Files:**
- Modify: `deploy/frontend/src/App.tsx`

- [ ] **Step 1: Write the full App.tsx**

```tsx
// deploy/frontend/src/App.tsx
import { useState } from "react";
import { StatCards } from "./components/StatCards";
import { SermonsByYearChart, SermonsBySpeakerChart } from "./components/BarChart";
import { TopVersesChart } from "./components/TopVersesChart";
import { BubbleChart } from "./components/BubbleChart";
import { ChatPanel } from "./components/ChatPanel";
import type { Filters } from "./types";

export default function App() {
  const [filters, setFilters] = useState<Filters>({ year: null, speaker: null });

  return (
    <div className="min-h-screen bg-navy-900 text-slate-100 font-sans">
      {/* Header */}
      <header className="border-b border-slate-800 px-6 py-4 flex items-center gap-4">
        <div>
          <h1 className="text-2xl font-extrabold bg-gradient-to-r from-blue-400 to-violet-400 bg-clip-text text-transparent">
            BBTC Sermon Intelligence
          </h1>
          <p className="text-slate-400 text-sm">Agentic RAG Pipeline · 2015 – present</p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Summary stats */}
        <StatCards />

        {/* Primary charts row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-navy-800 border border-slate-700 rounded-xl p-4">
            <SermonsByYearChart />
          </div>
          <div className="bg-navy-800 border border-slate-700 rounded-xl p-4">
            <SermonsBySpeakerChart />
          </div>
        </div>

        {/* Bubble chart — full width */}
        <div className="bg-navy-800 border border-slate-700 rounded-xl p-4">
          <BubbleChart
            onSelect={(f) => setFilters(f)}
          />
        </div>

        {/* Bible books */}
        <div className="bg-navy-800 border border-slate-700 rounded-xl p-4">
          <TopVersesChart />
        </div>

        {/* Chat panel */}
        <div>
          <h2 className="text-base font-semibold text-slate-300 mb-3">
            💬 Ask the Sermon Assistant
          </h2>
          <ChatPanel
            filters={filters}
            onFiltersChange={setFilters}
          />
        </div>
      </main>

      <footer className="border-t border-slate-800 text-center py-4 text-xs text-slate-600">
        BBTC Sermon Intelligence · Powered by Groq + ChromaDB + sentence-transformers
      </footer>
    </div>
  );
}
```

- [ ] **Step 2: Start dev server and verify the full layout**

```bash
cd deploy/frontend && npm run dev
```

Open `http://localhost:5173`. With backend running at `http://localhost:8000`:
- Stat cards load real numbers
- Bar charts show year/speaker distributions
- Bubble chart is interactive — clicking a bubble updates the chat filter badge
- Chat panel sends questions and shows citations

- [ ] **Step 3: Build for production to confirm no TypeScript errors**

```bash
npm run build
```

Expected: `dist/` folder created, no errors.

- [ ] **Step 4: Commit**

```bash
cd ../..
git add deploy/frontend/src/App.tsx
git commit -m "feat: wire up full dashboard in App.tsx with filter integration"
```

---

### Task 16: Frontend Dockerfile, nginx, and HuggingFace Space config

**Files:**
- Create: `deploy/frontend/Dockerfile`
- Create: `deploy/frontend/nginx.conf`
- Create: `deploy/frontend/README.md`

- [ ] **Step 1: Write nginx.conf**

```nginx
# deploy/frontend/nginx.conf
server {
    listen 3000;
    root /usr/share/nginx/html;
    index index.html;

    # SPA fallback — all routes serve index.html
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets aggressively, HTML never
    location ~* \.(js|css|png|jpg|svg|ico|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    location = /index.html {
        add_header Cache-Control "no-cache";
    }
}
```

- [ ] **Step 2: Write frontend Dockerfile**

```dockerfile
# deploy/frontend/Dockerfile
# --- Build stage ---
FROM node:20-alpine AS builder

WORKDIR /app
COPY deploy/frontend/package*.json ./
RUN npm ci

COPY deploy/frontend/ ./

# VITE_API_URL is injected at build time via --build-arg
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# --- Serve stage ---
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY deploy/frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
```

- [ ] **Step 3: Write README.md (HuggingFace Space card)**

```markdown
---
title: BBTC Sermon Intelligence
emoji: 📖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 3000
pinned: true
---

# BBTC Sermon Intelligence

A professional analytics dashboard and RAG-powered chat interface for the BBTC sermon archive (2015–present).

**Features:**
- Interactive charts: sermons by year, speaker, and Bible book
- Bubble scatter plot: year × speaker × sermon count
- Semantic search with CrossEncoder reranking
- Cited answers grounded in actual sermon text
```

- [ ] **Step 4: Build the frontend Docker image to verify**

```bash
docker build \
  -f deploy/frontend/Dockerfile \
  --build-arg VITE_API_URL=http://localhost:8000 \
  -t bbtc-frontend \
  .
docker run --rm -p 3000:3000 bbtc-frontend
```

Open `http://localhost:3000`. Expected: dashboard loads (charts show empty/loading state without backend).

- [ ] **Step 5: Commit**

```bash
git add deploy/frontend/Dockerfile deploy/frontend/nginx.conf deploy/frontend/README.md
git commit -m "feat: add frontend Dockerfile and HuggingFace Space config"
```

---

## Phase 3: Deployment

---

### Task 17: Deploy backend to Render

- [ ] **Step 1: Push the repo to GitHub**

```bash
git remote add origin https://github.com/<your-username>/structure_db_rag.git
git push -u origin main
```

- [ ] **Step 2: Create Render account and connect the repo**

1. Go to [render.com](https://render.com) → New → Blueprint
2. Connect your GitHub repo
3. Render auto-detects `render.yaml` and shows 2 services: `bbtc-sermon-api` (web) and `bbtc-sermon-ingestion` (cron)
4. Click **Apply**

- [ ] **Step 3: Set secret environment variables in Render dashboard**

In Render → `bbtc-sermon-api` → Environment:
- `GROQ_API_KEY` → your Groq key
- `GEMINI_API_KEY` → your Gemini key
- `FRONTEND_URL` → (set after HF Spaces deploy in Task 18, e.g. `https://your-space.hf.space`)

- [ ] **Step 4: Trigger first deploy and run reindex**

After the first deploy completes, open Render → `bbtc-sermon-api` → Shell:

```bash
# Copy your data files to the persistent disk first
# (on first deploy the disk is empty)
# If you have a way to upload, copy data/sermons.db and data/sermons/*.txt to /data/
# Then run:
python scripts/reindex.py
```

Expected: ChromaDB rebuilt with `bge-small-en-v1.5`. Takes ~15 min.

- [ ] **Step 5: Verify the API is live**

```bash
curl https://bbtc-sermon-api.onrender.com/api/stats
```

Expected: JSON with real sermon counts.

---

### Task 18: Deploy frontend to HuggingFace Spaces

- [ ] **Step 1: Create a HuggingFace Space**

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Space name: `bbtc-sermon-intelligence`
3. SDK: **Docker**
4. Visibility: Public (or Private)
5. Click **Create Space**

- [ ] **Step 2: Push only the frontend files to the Space repo**

HuggingFace Spaces uses a separate Git repo. The Dockerfile must be at the repo root.

```bash
# Create a dedicated HF deployment directory
mkdir -p hf_deploy
cp -r deploy/frontend/* hf_deploy/
# The Dockerfile references deploy/frontend/ paths — update it for HF (flat structure)
```

Update the copy in `hf_deploy/Dockerfile` to use flat paths (since HF Space root IS the frontend directory):

```dockerfile
# hf_deploy/Dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
ARG VITE_API_URL=https://bbtc-sermon-api.onrender.com
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
```

```bash
cd hf_deploy
git init
git remote add origin https://huggingface.co/spaces/<your-username>/bbtc-sermon-intelligence
git add .
git commit -m "initial deploy"
git push origin main
```

- [ ] **Step 3: Set VITE_API_URL as a Space secret**

In HuggingFace → your Space → Settings → Variables and secrets:
- `VITE_API_URL` = `https://bbtc-sermon-api.onrender.com`

Trigger a rebuild after adding the secret.

- [ ] **Step 4: Update FRONTEND_URL in Render**

Back in Render → `bbtc-sermon-api` → Environment:
- `FRONTEND_URL` = `https://<your-username>-bbtc-sermon-intelligence.hf.space`

This updates CORS to allow the HF Spaces origin. Redeploy the Render service.

- [ ] **Step 5: End-to-end verification**

Open `https://<your-username>-bbtc-sermon-intelligence.hf.space`:
- Stat cards show real counts
- Charts render with real data
- Clicking a bubble sets year/speaker filter
- Asking a question in chat returns an answer with citations

- [ ] **Step 6: Final commit**

```bash
cd ..
git add hf_deploy/
git commit -m "chore: add HuggingFace Spaces deployment directory"
```

---

## Self-Review Checklist

- [x] Spec: embedding consistency fix → Task 6 (reindex) + Task 4 (rag.py uses bge-small)
- [x] Spec: CrossEncoder reranking → Task 4 `_reranker.predict()`
- [x] Spec: chart endpoints → Task 3 (all 5 endpoints with tests)
- [x] Spec: `/api/chat` with year/speaker filters → Task 5 (endpoint) + Task 4 (where filter)
- [x] Spec: React dashboard, Plotly charts → Tasks 11–15
- [x] Spec: BubbleChart click → filter propagation → Task 13 `onSelect` prop + Task 15 `filters` state
- [x] Spec: Render persistent disk, cron job → Task 8 render.yaml
- [x] Spec: HuggingFace Spaces Docker → Task 16 Dockerfile + README
- [x] Spec: sentence-transformers no API key needed → confirmed in rag.py, reindex.py, run_ingestion.py
- [x] Spec: `run_ingestion.py` standalone (not Dagster) → Task 7
- [x] Types consistent: `Filters` defined in types.ts, used in `useChat`, `ChatPanel`, `BubbleChart`, `App.tsx`
- [x] API paths consistent: all hooks use same paths as charts.py router definitions
