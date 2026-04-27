# Cleanup, Simplification & Capstone Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all dead code, switch to Ollama-only LLM, add a scatterplot chart, and rewrite the README for capstone submission — all in one clean pass.

**Architecture:** The live app is `app.py` (Gradio + LangGraph ReAct agent) backed by `src/` (storage, tools, ingestion). Everything else (Flask app, FastAPI deploy folder, scratch scripts) is dead weight being deleted. The LLM stack is simplified to a single `ChatOllama` call throughout.

**Tech Stack:** Python 3.11, Gradio, LangGraph, LangChain, ChromaDB, SQLite, Matplotlib, Ollama (`llama3.1:8b` for chat, `llama3.2:3b` for metadata extraction)

---

## File Map

| File | Action | What changes |
|---|---|---|
| `app/` | **Delete** | Entire folder — old Flask app |
| `deploy/` | **Delete** | Entire folder — abandoned FastAPI backend |
| `templates/` | **Delete** | Entire folder — Flask HTML |
| `static/` | **Delete** | Entire folder — Flask CSS/JS |
| `flask_session/` | **Delete** | Entire folder — Flask session files |
| `run.py` | **Delete** | Flask runner |
| `scratch/` | **Delete** | Entire folder — debug scripts |
| `vectorstore/` | **Delete** | Entire folder — stale root-level Chroma DB |
| `requirements.txt` | **Modify** | Remove `langchain-groq`, `langchain-google-genai`, `groq`, `flask`, `flask-session`, `gunicorn` |
| `render.yaml` | **Modify** | Remove `GROQ_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY` env vars |
| `src/llm.py` | **Modify** | Ollama-only — strip all Groq/Gemini branches |
| `src/ingestion/metadata_extractor.py` | **Modify** | Ollama-only — remove Groq primary + fallback |
| `src/tools/matplotlib_tool.py` | **Modify** | Add `sermons_scatter` chart type |
| `app.py` | **Modify** | Remove provider radio, API key remap, simplify `respond()` |
| `tests/test_matplotlib_tool.py` | **Create** | Tests for `sermons_scatter` |
| `tests/test_llm.py` | **Create** | Test that `get_llm()` returns `ChatOllama` |
| `README.md` | **Modify** | Rewrite for capstone presentation |

---

## Task 1: Delete dead code

**Files:**
- Delete: `app/`, `deploy/`, `templates/`, `static/`, `flask_session/`, `run.py`, `scratch/`, `vectorstore/`

- [ ] **Step 1: Delete all abandoned paths**

```bash
cd /Users/simgsr/Documents/structure_db_rag
rm -rf app/ deploy/ templates/ static/ flask_session/ run.py scratch/ vectorstore/
```

- [ ] **Step 2: Verify deletions**

```bash
ls app/ deploy/ templates/ static/ flask_session/ run.py scratch/ vectorstore/ 2>&1
```

Expected: `ls: cannot access ...` errors for all paths — none should exist.

- [ ] **Step 3: Run existing tests to confirm nothing broke**

```bash
python -m pytest tests/ -v
```

Expected: All existing tests pass (test_reranker, test_ui_helpers, test_vector_tool).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove all abandoned code (Flask app, FastAPI deploy, scratch scripts)"
```

---

## Task 2: Trim requirements.txt and render.yaml

**Files:**
- Modify: `requirements.txt`
- Modify: `render.yaml`

- [ ] **Step 1: Remove cloud packages from requirements.txt**

Replace the entire `requirements.txt` with:

```
dagster
dagster-webserver
cloudscraper
beautifulsoup4
PyMuPDF
python-docx
python-pptx
chromadb
langchain
langchain-chroma
langchain-ollama
langchain-community
sentence-transformers
ollama
pandas
python-dotenv
gradio
matplotlib
langgraph
```

- [ ] **Step 2: Remove cloud API keys from render.yaml**

Replace `render.yaml` with:

```yaml
services:
  - type: web
    name: sermon-intelligence
    runtime: docker
    plan: starter
    region: singapore
    envVars:
      - key: PORT
        value: 7860
    disk:
      name: sermon-data
      mountPath: /app/data
      sizeGB: 1
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt render.yaml
git commit -m "chore: remove cloud LLM packages and API key env vars"
```

---

## Task 3: Simplify src/llm.py to Ollama-only

**Files:**
- Modify: `src/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm.py`:

```python
from unittest.mock import patch, MagicMock
from langchain_ollama import ChatOllama


def test_get_llm_returns_chat_ollama():
    with patch("src.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm
        result = get_llm()
        mock_cls.assert_called_once()


def test_get_llm_passes_temperature():
    with patch("src.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm
        get_llm(temperature=0.5)
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("temperature") == 0.5


def test_get_llm_uses_custom_model():
    with patch("src.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm
        get_llm(ollama_model="llama3.2:3b")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("model") == "llama3.2:3b"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_llm.py -v
```

Expected: FAIL — `get_llm` currently has Groq/Gemini imports that no longer exist after requirements.txt trimmed, or wrong return type.

- [ ] **Step 3: Rewrite src/llm.py**

```python
from langchain_ollama import ChatOllama


def get_llm(temperature=0, ollama_model="llama3.1:8b"):
    return ChatOllama(model=ollama_model, temperature=temperature)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_llm.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/llm.py tests/test_llm.py
git commit -m "refactor: simplify get_llm() to Ollama-only"
```

---

## Task 4: Simplify MetadataExtractor to Ollama-only

**Files:**
- Modify: `src/ingestion/metadata_extractor.py`

- [ ] **Step 1: Rewrite src/ingestion/metadata_extractor.py**

```python
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

_SYSTEM = """You are a metadata extraction assistant for church sermon files.
Given the first 500 characters of a sermon document, extract:
- speaker: the pastor or preacher's name (string or null)
- date: the sermon date as YYYY-MM-DD (string or null)
- series: the sermon series name (string or null)
- bible_book: the primary Bible book referenced (string or null)
- primary_verse: the key verse e.g. "Romans 8:28" (string or null)

Respond ONLY with a valid JSON object. No explanation. No markdown fences."""

_EMPTY = {"speaker": None, "date": None, "series": None, "bible_book": None, "primary_verse": None}


class MetadataExtractor:
    def __init__(self):
        self._llm = ChatOllama(model="llama3.2:3b", temperature=0)

    def extract(self, text_preview: str) -> dict:
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=text_preview[:500]),
        ]
        try:
            raw = self._llm.invoke(messages).content.strip()
            return json.loads(raw)
        except Exception:
            return _EMPTY.copy()
```

- [ ] **Step 2: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass (MetadataExtractor has no direct unit tests but the import chain is verified).

- [ ] **Step 3: Commit**

```bash
git add src/ingestion/metadata_extractor.py
git commit -m "refactor: simplify MetadataExtractor to Ollama-only"
```

---

## Task 5: Add sermons_scatter chart

**Files:**
- Modify: `src/tools/matplotlib_tool.py`
- Create: `tests/test_matplotlib_tool.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_matplotlib_tool.py`:

```python
import os
import sqlite3
import pytest
from unittest.mock import MagicMock
from src.tools.matplotlib_tool import make_matplotlib_tool


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "sermons.db")
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE sermons (year INTEGER, speaker TEXT, bible_book TEXT)"
        )
        conn.executemany(
            "INSERT INTO sermons VALUES (?, ?, ?)",
            [
                (2022, "Pastor A", "Romans"),
                (2022, "Pastor A", "John"),
                (2023, "Pastor B", "Psalms"),
                (2023, "Pastor A", "Genesis"),
                (2024, "Pastor B", "Romans"),
            ],
        )
    return path


@pytest.fixture
def chart_tool(db_path):
    registry = MagicMock()
    registry.db_path = db_path
    return make_matplotlib_tool(registry)


def test_sermons_scatter_returns_png_path(chart_tool):
    result = chart_tool.invoke({"chart_name": "sermons_scatter"})
    assert result.endswith(".png"), f"Expected a PNG path, got: {result}"
    assert os.path.exists(result)


def test_sermons_scatter_file_is_nonempty(chart_tool):
    result = chart_tool.invoke({"chart_name": "sermons_scatter"})
    assert os.path.getsize(result) > 0


def test_sermons_scatter_empty_db_returns_message(tmp_path):
    path = str(tmp_path / "empty.db")
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE sermons (year INTEGER, speaker TEXT, bible_book TEXT)"
        )
    registry = MagicMock()
    registry.db_path = path
    tool = make_matplotlib_tool(registry)
    result = tool.invoke({"chart_name": "sermons_scatter"})
    assert "No sermon data" in result


def test_unknown_chart_name_returns_error(chart_tool):
    result = chart_tool.invoke({"chart_name": "unknown_chart"})
    assert "Unknown chart" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_matplotlib_tool.py::test_sermons_scatter_returns_png_path -v
```

Expected: FAIL — `"sermons_scatter"` hits the `else` branch returning `"Unknown chart ..."`.

- [ ] **Step 3: Add sermons_scatter case to matplotlib_tool.py**

In `src/tools/matplotlib_tool.py`, replace the `else` block (lines 70–75) with:

```python
                elif chart_name == "sermons_scatter":
                    rows = conn.execute(
                        "SELECT year, speaker, COUNT(*) as n FROM sermons "
                        "WHERE year IS NOT NULL AND speaker IS NOT NULL AND speaker != '' "
                        "GROUP BY year, speaker ORDER BY year"
                    ).fetchall()
                    if not rows:
                        plt.close(fig)
                        return "No sermon data found."
                    years = [r[0] for r in rows]
                    speakers = [r[1] for r in rows]
                    counts = [r[2] for r in rows]
                    unique_speakers = sorted(set(speakers))
                    speaker_idx = {s: i for i, s in enumerate(unique_speakers)}
                    y_vals = [speaker_idx[s] for s in speakers]
                    n_speakers = len(unique_speakers)
                    fig.set_figheight(max(6, n_speakers * 0.5))
                    ax.scatter(years, y_vals, s=[c * 40 for c in counts], alpha=0.6, color="#f59e0b")
                    ax.set_yticks(range(n_speakers))
                    ax.set_yticklabels(unique_speakers, fontsize=8)
                    ax.set_xlabel("Year")
                    ax.set_title("Sermon Count by Speaker and Year")
                    ax.annotate(
                        "Bubble size = sermon count",
                        xy=(0.01, 0.01), xycoords="axes fraction", fontsize=8, color="gray"
                    )

                else:
                    plt.close(fig)
                    return (
                        f"Unknown chart '{chart_name}'. "
                        "Valid options: sermons_per_speaker, sermons_per_year, "
                        "top_bible_books, sermons_scatter."
                    )
```

Also update the tool docstring at the top of `matplotlib_tool` to add the new entry:

```python
        """Generates a chart from live sermon data and returns the PNG file path.
        Supported chart_name values:
        - 'sermons_per_speaker' — bar chart of sermon count per speaker (top 10)
        - 'sermons_per_year' — bar chart of sermon count per year
        - 'top_bible_books' — bar chart of most-preached Bible books (top 10)
        - 'sermons_scatter' — bubble chart of sermon count by speaker and year
        Returns the file path to the saved PNG."""
```

- [ ] **Step 4: Run all matplotlib tool tests**

```bash
python -m pytest tests/test_matplotlib_tool.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/tools/matplotlib_tool.py tests/test_matplotlib_tool.py
git commit -m "feat: add sermons_scatter bubble chart to matplotlib_tool"
```

---

## Task 6: Simplify app.py

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace app.py with the simplified version**

Replace the entire `app.py` with:

```python
import gradio as gr
import os
from dotenv import load_dotenv
from src.storage.chroma_store import SermonVectorStore
from src.llm import get_llm
from src.ui_helpers import extract_chart_path, fetch_archive_stats, render_stats_bar
from src.storage.sqlite_store import SermonRegistry
from src.tools.sql_tool import make_sql_tool
from src.tools.vector_tool import make_vector_tool
from src.tools.bible_tool import make_bible_tool
from src.tools.matplotlib_tool import make_matplotlib_tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

try:
    registry = SermonRegistry()
    vector_store = SermonVectorStore()
    llm = get_llm(temperature=0.1)

    sql_tool = make_sql_tool(registry)
    vector_tool = make_vector_tool(vector_store)
    bible_tool = make_bible_tool(vector_store)
    viz_tool = make_matplotlib_tool(registry)

    SYSTEM_PROMPT = (
        "You are the BBTC Sermon Intelligence Assistant for Bethesda Bedok-Tampines Church.\n\n"
        "## Tool routing\n"
        "- Use 'sql_query_tool' for: counts, statistics, lists of speakers/years, date lookups, "
        "questions that need numbers (e.g. 'how many sermons', 'top 5 speakers').\n"
        "- Use 'search_sermons_tool' for: questions about sermon *content*, topics, theology, "
        "what a pastor said, summaries of specific sermons. Pass 'year' or 'speaker' filters "
        "when the user specifies them.\n"
        "- For 'what was said about X in year Y' or 'what did speaker Z say about X', use search_sermons_tool "
        "with the year/speaker filter directly — do not run sql_query_tool first.\n"
        "- Use 'compare_bible_versions' only when the user explicitly asks to compare Bible translations.\n"
        "- Use 'matplotlib_tool' only when the user asks for a chart or visualization. "
        "Valid chart_name values: 'sermons_per_speaker', 'sermons_per_year', 'top_bible_books', 'sermons_scatter'.\n\n"
        "## Grounding rules\n"
        "- Answer ONLY from data returned by the tools. Never invent sermon content, speaker names, "
        "dates, or verses.\n"
        "- When answering from search_sermons_tool results, cite the sermon filename and speaker name for every excerpt quoted.\n"
        "- If the tools return no relevant data, say so explicitly — do not guess or fill gaps.\n"
        "- If you need more information to answer precisely, call the relevant tool again with "
        "a refined query before responding.\n"
    )

    agent = create_agent(llm, tools=[sql_tool, vector_tool, bible_tool, viz_tool], system_prompt=SYSTEM_PROMPT)

except Exception as e:
    print(f"⚠️ Initialization warning: {e}")
    agent = None
    registry = None
    vector_store = None

_stats_bar_html = (
    render_stats_bar(fetch_archive_stats(registry.db_path))
    if registry is not None
    else render_stats_bar(None)
)


def respond(message, history):
    if agent is None:
        return "⚠️ Agent not initialized. Check that Ollama is running."

    truncated_history = history[-6:] if len(history) > 6 else history
    messages = []
    for turn in truncated_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            content = turn["content"]
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") for block in content
                    if block.get("type") == "text"
                )
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=message))

    try:
        result = agent.invoke({"messages": messages})
        return result["messages"][-1].content
    except Exception as e:
        return f"⚠️ An error occurred while processing your request: {e}"


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

footer {visibility: hidden}
.gradio-container {
    background-color: #0f172a !important;
    color: #f8fafc;
    font-family: 'Inter', sans-serif !important;
    max-width: 1200px !important;
}

.sidebar {
    background: rgba(30, 41, 59, 0.7) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 20px !important;
    border-radius: 12px;
}

.chatbot-container {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background: rgba(30, 41, 59, 0.4) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.message-user {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 16px !important;
    box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.2);
}
.message-assistant {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 12px 16px !important;
}

.input-container {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 5px !important;
}

.btn-primary {
    background: linear-gradient(to right, #38bdf8, #818cf8) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}

#title-container {
    margin-bottom: 30px;
    text-align: left;
    display: flex;
    align-items: center;
    gap: 20px;
}
#title-container img {
    height: 60px;
}
#title-text h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(to right, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#title-text p {
    color: #94a3b8;
    margin: 0;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-online { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
.status-offline { background: rgba(239, 68, 68, 0.2); color: #f87171; }

.stats-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    background: #1e293b;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 16px;
    color: #94a3b8;
    font-size: 0.875rem;
}

@media (max-width: 768px) {
    .gradio-container { max-width: 100% !important; }
    .sidebar {
        border-right: none !important;
        border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    .stats-bar { flex-direction: column; gap: 6px; align-items: flex-start; }
    #title-text h1 { font-size: 1.8rem; }
    .message-user, .message-assistant { font-size: 0.9rem; padding: 10px 12px !important; }
}
"""

with gr.Blocks() as demo:
    with gr.Row(elem_id="header"):
        with gr.Column(scale=4):
            gr.HTML("""
                <div id='title-container'>
                    <img src='https://www.bbtc.com.sg/wp-content/uploads/2021/04/BBTC-Logo-Header.png' alt='Logo'>
                    <div id='title-text'>
                        <h1>Sermon Intelligence</h1>
                        <p>Agentic RAG Pipeline for BBTC Sermon History</p>
                    </div>
                </div>
            """)

    gr.HTML(_stats_bar_html)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                min_height=400,
                show_label=False,
                elem_classes="chatbot-container",
                avatar_images=(None, "https://www.bbtc.com.sg/wp-content/uploads/2021/04/BBTC-Logo-Header.png")
            )
            with gr.Row(elem_classes="input-container"):
                msg = gr.Textbox(
                    placeholder="Ask about a sermon topic, verse, or summary...",
                    container=False,
                    scale=7,
                )
                submit = gr.Button("🚀 Send", variant="primary", scale=1, elem_classes="btn-primary")

            with gr.Row():
                gr.Examples(
                    examples=[
                        ["How many sermons are in the archive and who are the top 5 speakers?"],
                        ["Show a bar chart of how many sermons were preached each year"],
                        ["Show a bar chart of the top 10 most-preached Bible books"],
                        ["Create a bar chart of sermon count per speaker"],
                        ["Show a scatter plot of sermon counts by speaker and year"],
                        ["What sermons have been preached on the book of Romans?"],
                        ["Find sermons about forgiveness, grace, and redemption"],
                        ["What have our pastors said about faith during trials and suffering?"],
                        ["Find sermons that cover John 3:16 or the topic of eternal life"],
                        ["What was the most recent sermon and what were its key points?"],
                    ],
                    inputs=msg,
                    label="⚡ Quick Inquiries"
                )

        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("### ⚙️ System Status")

            vec_status = "online" if vector_store else "offline"
            gr.HTML(f"""
                <div style='display: flex; flex-direction: column; gap: 10px;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span>Vector Store</span>
                        <span class='status-badge status-{vec_status}'>{vec_status.upper()}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between;'>
                        <span>Database</span>
                        <span class='status-badge status-online'>CONNECTED</span>
                    </div>
                    <div style='display: flex; justify-content: space-between;'>
                        <span>LLM Engine</span>
                        <span class='status-badge status-online'>OLLAMA</span>
                    </div>
                </div>
            """)

            gr.Markdown("---")
            gr.Markdown("### 📖 About")
            gr.Markdown(
                "This assistant uses a hybrid Agentic RAG pipeline. "
                "It routes queries between SQL metadata search and semantic vector search, "
                "and can generate data visualisations on demand."
            )

            clear = gr.Button("🗑️ Reset Conversation", variant="secondary")

    def user_msg(user_message, history: list):
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]

    def bot_msg(history: list):
        if not history or history[-1]["role"] != "user":
            return history

        user_message = history[-1]["content"]
        if isinstance(user_message, list):
            user_message = " ".join(
                [m["text"] for m in user_message if isinstance(m, dict) and m.get("type") == "text"]
            )

        chat_history = history[:-1]
        bot_message = respond(user_message, chat_history)

        if not isinstance(bot_message, str):
            bot_message = str(bot_message)

        text, chart_path = extract_chart_path(bot_message)
        if chart_path:
            content = [
                {"type": "text", "text": text},
                {"path": chart_path},
            ]
        else:
            content = bot_message

        history.append({"role": "assistant", "content": content})
        return history

    disable_submit = lambda: gr.update(value="⏳ Thinking...", interactive=False)
    enable_submit = lambda: gr.update(value="🚀 Send", interactive=True)

    msg.submit(user_msg, [msg, chatbot], [msg, chatbot], queue=True).then(
        disable_submit, None, submit
    ).then(
        bot_msg, [chatbot], chatbot
    ).then(
        enable_submit, None, submit
    )
    submit.click(user_msg, [msg, chatbot], [msg, chatbot], queue=True).then(
        disable_submit, None, submit
    ).then(
        bot_msg, [chatbot], chatbot
    ).then(
        enable_submit, None, submit
    )
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=custom_css,
        theme=gr.themes.Default(),
        allowed_paths=["/tmp"]
    )
```

- [ ] **Step 2: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "refactor: simplify app.py — Ollama-only, remove provider radio, add scatter quick query"
```

---

## Task 7: Rewrite README for capstone

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README.md**

```markdown
# BBTC Sermon Intelligence

A **Hybrid Agentic RAG pipeline** for the Bethesda Bedok-Tampines Church (BBTC) sermon archive. It combines structured SQL metadata querying with semantic vector search, backed by a LangGraph ReAct agent that intelligently routes each question to the right data source.

Built as a capstone project demonstrating end-to-end ML engineering: data ingestion, dual-layer storage, LLM-powered metadata extraction, agent orchestration, and a production-ready chat interface.

---

## Features

- **Agentic RAG** — LangGraph ReAct agent dynamically routes queries between SQL, vector search, and chart generation
- **Hybrid storage** — SQLite for structured metadata (speaker, date, series, verse) + ChromaDB for semantic content search
- **CrossEncoder reranking** — improves retrieval quality by reranking top-20 candidates with a cross-encoder model
- **Data visualisations** — bar charts and scatter plots generated on demand from live SQLite data
- **Automated ingestion** — Dagster pipeline scrapes and indexes new sermons weekly

---

## Architecture

```
BBTC Website → BBTCScraper → data/staging/ (raw files)
                           → data/sermons/ (.txt extracts)
                           → SQLite (data/sermons.db) [status: extracted]
                               ↓ MetadataExtractor (Ollama)
                           → SQLite [status: indexed, with speaker/date/verse]
                           → ChromaDB (data/chroma_db/) [vector chunks]
                               ↓
                        Gradio UI → LangGraph ReAct Agent → Tools → Response
```

### Key Components

| Component | File | Purpose |
|---|---|---|
| `SermonRegistry` | `src/storage/sqlite_store.py` | SQLite CRUD; tracks ingestion status |
| `SermonVectorStore` | `src/storage/chroma_store.py` | ChromaDB with Ollama embeddings + CrossEncoder reranker |
| `BBTCScraper` | `src/scraper/bbtc_scraper.py` | Cloudflare-bypass scraper; downloads + text-extracts PDFs/PPTX/DOCX |
| `MetadataExtractor` | `src/ingestion/metadata_extractor.py` | Ollama LLM extracts speaker/date/series/verse from sermon text |
| `get_llm()` | `src/llm.py` | Returns a `ChatOllama` instance |
| `dagster_pipeline.py` | root | Dagster asset — weekly scrape + ingest schedule |
| `app.py` | root | Gradio UI + LangGraph ReAct agent |

### Agent Tools

| Tool | When used |
|---|---|
| `sql_query_tool` | Counts, stats, date lookups, top-N queries |
| `search_sermons_tool` | Semantic content search with optional year/speaker filter |
| `matplotlib_tool` | On-demand charts: `sermons_per_speaker`, `sermons_per_year`, `top_bible_books`, `sermons_scatter` |
| `compare_bible_versions` | Bible translation comparisons |

---

## Local Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally with these models pulled:
  ```bash
  ollama pull llama3.1:8b        # chat agent
  ollama pull llama3.2:3b        # metadata extraction
  ollama pull nomic-embed-text   # embeddings
  ```

### Install and run

```bash
git clone <repo-url>
cd structure_db_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open [http://localhost:7860](http://localhost:7860).

### SQLite Schema

```sql
sermons(
  sermon_id TEXT PRIMARY KEY,
  filename TEXT,
  url TEXT UNIQUE,
  speaker TEXT,
  date TEXT,          -- YYYY-MM-DD
  series TEXT,
  bible_book TEXT,
  primary_verse TEXT, -- e.g. "Romans 8:28"
  language TEXT,      -- "English" | "Mandarin"
  file_type TEXT,     -- pdf | pptx | docx
  year INTEGER,
  status TEXT,        -- extracted → indexed | failed
  date_scraped TEXT
)
```

---

## Data Pipeline

```bash
# Full pipeline: scrape + extract + vectorise (via Dagster)
dagster asset materialize --select sermon_ingestion_summary -m dagster_pipeline

# Dagster web UI (to trigger/monitor)
DAGSTER_HOME=$(mktemp -d) dagster dev -m dagster_pipeline

# Vectorise already-extracted sermons without re-scraping
python quick_ingest.py

# Scrape a single year only
python src/scraper/bbtc_scraper.py 2024
```

---

## Deployment

The `Dockerfile` runs the Gradio interface on port `7860`. Deploy on [Render](https://render.com) using `render.yaml`:

```bash
# Build and test locally
docker build -t sermon-intelligence .
docker run -p 7860:7860 -v $(pwd)/data:/app/data sermon-intelligence
```

Mount `data/` to a persistent volume to preserve the SQLite database and ChromaDB vector store across restarts.

---

## Running Tests

```bash
python -m pytest tests/ -v
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for capstone submission"
```

---

## Task 8: Final verification

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass with no errors.

- [ ] **Step 2: Verify import chain is clean**

```bash
python -c "import app" 2>&1 | head -20
```

Expected: No `ModuleNotFoundError` for `langchain_groq`, `langchain_google_genai`, or any deleted module.

- [ ] **Step 3: Check git log**

```bash
git log --oneline -8
```

Expected: 7 clean commits visible (Tasks 1–7).

- [ ] **Step 4: Confirm deleted paths are gone**

```bash
ls app/ deploy/ templates/ static/ flask_session/ run.py scratch/ vectorstore/ 2>&1 | grep -c "cannot access"
```

Expected: `8` (all 8 paths absent).

- [ ] **Step 5: Final commit if any stray files remain**

```bash
git status
```

If any untracked files remain that should be ignored, add them to `.gitignore` and commit.
