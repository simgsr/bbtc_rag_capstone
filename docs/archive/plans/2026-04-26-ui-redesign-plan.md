# UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 5 UI enhancements to `app.py`: inline chart rendering in chat, live stats bar, loading indicator, 10 updated quick queries, and mobile-responsive CSS.

**Architecture:** All UI wiring stays in `app.py`. Two pure helper functions (`extract_chart_path`, `fetch_archive_stats`, `render_stats_bar`) live in `src/ui_helpers.py` for independent testability — they have no Gradio or LLM dependencies. The `bot_msg` function detects chart file paths in agent responses and reconstructs the message as a Gradio multimodal content block. Stats are fetched from SQLite at module load time and rendered into a `gr.HTML` pill bar.

**Tech Stack:** Python, Gradio 4.x, SQLite (stdlib `sqlite3`), `re` (stdlib), pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/ui_helpers.py` | Create | `extract_chart_path`, `fetch_archive_stats`, `render_stats_bar` — pure/DB helpers, no Gradio |
| `tests/test_ui_helpers.py` | Create | Unit tests for all three helpers |
| `app.py` | Modify | Import helpers, update `bot_msg`, layout, CSS, event chains, launch flags |

---

## Task 1: Create `src/ui_helpers.py` with unit tests (TDD)

**Files:**
- Create: `src/ui_helpers.py`
- Create: `tests/test_ui_helpers.py`

### Step 1.1 — Write the failing tests

Create `tests/test_ui_helpers.py`:

```python
import sqlite3
import pytest
from src.ui_helpers import extract_chart_path, fetch_archive_stats, render_stats_bar


# ── extract_chart_path ────────────────────────────────────────────────────────

class TestExtractChartPath:
    def test_detects_path_in_response(self):
        response = "Here is the chart: /tmp/bbtc_chart_abc12345.png"
        _, path = extract_chart_path(response)
        assert path == "/tmp/bbtc_chart_abc12345.png"

    def test_strips_path_from_text(self):
        response = "Here is the chart: /tmp/bbtc_chart_abc12345.png"
        text, _ = extract_chart_path(response)
        assert "/tmp/bbtc_chart" not in text

    def test_strips_trailing_colon_artifact(self):
        response = "Here is the chart: /tmp/bbtc_chart_abc12345.png"
        text, _ = extract_chart_path(response)
        assert text == "Here is the chart"

    def test_returns_none_when_no_path(self):
        response = "No chart here, just text."
        text, path = extract_chart_path(response)
        assert path is None
        assert text == "No chart here, just text."

    def test_default_label_when_only_path(self):
        response = "/tmp/bbtc_chart_abc12345.png"
        text, path = extract_chart_path(response)
        assert path == "/tmp/bbtc_chart_abc12345.png"
        assert text == "Here is the chart:"

    def test_non_hex_filename_not_matched(self):
        # 'x' is not in [a-f0-9] — should not match
        response = "See /tmp/bbtc_chart_xyz99999.png for details"
        _, path = extract_chart_path(response)
        assert path is None

    def test_preserves_text_before_and_after_path(self):
        response = "Preface text. /tmp/bbtc_chart_aabb1234.png More text."
        text, path = extract_chart_path(response)
        assert path == "/tmp/bbtc_chart_aabb1234.png"
        assert "Preface text." in text
        assert "More text." in text


# ── fetch_archive_stats ───────────────────────────────────────────────────────

class TestFetchArchiveStats:
    @pytest.fixture
    def db_path(self, tmp_path):
        path = str(tmp_path / "test.db")
        with sqlite3.connect(path) as conn:
            conn.execute("""
                CREATE TABLE sermons (
                    sermon_id TEXT PRIMARY KEY,
                    speaker TEXT,
                    year INTEGER,
                    language TEXT
                )
            """)
            conn.executemany(
                "INSERT INTO sermons VALUES (?, ?, ?, ?)",
                [
                    ("s1", "Pastor A", 2022, "English"),
                    ("s2", "Pastor A", 2023, "English"),
                    ("s3", "Pastor B", 2024, "Mandarin"),
                    ("s4", None, None, None),
                ],
            )
        return path

    def test_sermon_count_includes_all_rows(self, db_path):
        assert fetch_archive_stats(db_path)["sermons"] == 4

    def test_speaker_count_excludes_null(self, db_path):
        assert fetch_archive_stats(db_path)["speakers"] == 2

    def test_year_range(self, db_path):
        stats = fetch_archive_stats(db_path)
        assert stats["year_min"] == 2022
        assert stats["year_max"] == 2024

    def test_language_count_excludes_null(self, db_path):
        assert fetch_archive_stats(db_path)["languages"] == 2

    def test_returns_none_when_db_missing(self):
        assert fetch_archive_stats("/nonexistent/db.sqlite") is None


# ── render_stats_bar ──────────────────────────────────────────────────────────

class TestRenderStatsBar:
    def test_renders_all_stat_fields(self):
        stats = {
            "sermons": 847, "speakers": 14,
            "year_min": 2018, "year_max": 2024, "languages": 2,
        }
        html = render_stats_bar(stats)
        assert "847 sermons" in html
        assert "14 speakers" in html
        assert "2018" in html
        assert "2024" in html
        assert "2 languages" in html

    def test_fallback_html_when_stats_none(self):
        html = render_stats_bar(None)
        assert "unavailable" in html.lower()
        assert "stats-bar" in html

    def test_renders_na_when_year_is_none(self):
        stats = {
            "sermons": 10, "speakers": 3,
            "year_min": None, "year_max": None, "languages": 1,
        }
        html = render_stats_bar(stats)
        assert "N/A" in html
```

- [ ] **Step 1.2 — Run tests, confirm they all fail**

```bash
cd /Users/simgsr/Documents/structure_db_rag
source .venv/bin/activate
pytest tests/test_ui_helpers.py -v
```

Expected: `ImportError` — `src.ui_helpers` does not exist yet.

- [ ] **Step 1.3 — Implement `src/ui_helpers.py`**

Create `src/ui_helpers.py`:

```python
import re
import sqlite3


def extract_chart_path(response: str) -> tuple[str, str | None]:
    """Extract a chart file path from agent response text.
    Returns (cleaned_text, chart_path) or (original_text, None) if no path found.
    """
    match = re.search(r'/tmp/bbtc_chart_[a-f0-9]+\.png', response)
    if match is None:
        return response, None
    chart_path = match.group(0)
    cleaned = (response[:match.start()] + response[match.end():]).strip().rstrip(':').strip()
    if not cleaned:
        cleaned = "Here is the chart:"
    return cleaned, chart_path


def fetch_archive_stats(db_path: str) -> dict | None:
    """Fetch live archive counts from SQLite.
    Returns dict with keys: sermons, speakers, year_min, year_max, languages.
    Returns None if the DB is unavailable.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            sermon_count = conn.execute("SELECT COUNT(*) FROM sermons").fetchone()[0]
            speaker_count = conn.execute(
                "SELECT COUNT(DISTINCT speaker) FROM sermons WHERE speaker IS NOT NULL"
            ).fetchone()[0]
            year_row = conn.execute(
                "SELECT MIN(year), MAX(year) FROM sermons WHERE year IS NOT NULL"
            ).fetchone()
            lang_count = conn.execute(
                "SELECT COUNT(DISTINCT language) FROM sermons "
                "WHERE language IS NOT NULL AND language != ''"
            ).fetchone()[0]
            return {
                "sermons": sermon_count,
                "speakers": speaker_count,
                "year_min": year_row[0],
                "year_max": year_row[1],
                "languages": lang_count,
            }
    except Exception:
        return None


def render_stats_bar(stats: dict | None) -> str:
    """Render archive stats as an HTML pill bar for use in gr.HTML."""
    if stats is None:
        return "<div class='stats-bar'>📚 Archive stats unavailable</div>"
    year_range = (
        f"{stats['year_min']} – {stats['year_max']}"
        if stats["year_min"] is not None
        else "N/A"
    )
    return (
        f"<div class='stats-bar'>"
        f"📚 {stats['sermons']} sermons &nbsp;·&nbsp; "
        f"👤 {stats['speakers']} speakers &nbsp;·&nbsp; "
        f"📅 {year_range} &nbsp;·&nbsp; "
        f"🌐 {stats['languages']} languages"
        f"</div>"
    )
```

- [ ] **Step 1.4 — Run tests, confirm they all pass**

```bash
pytest tests/test_ui_helpers.py -v
```

Expected: all 15 tests PASS.

- [ ] **Step 1.5 — Commit**

```bash
git add src/ui_helpers.py tests/test_ui_helpers.py
git commit -m "feat: add ui_helpers with chart path extraction and stats bar rendering"
```

---

## Task 2: Inline chart rendering in `bot_msg`

**Files:**
- Modify: `app.py` (imports, `bot_msg`, `gr.Chatbot`, `demo.launch`)

- [ ] **Step 2.1 — Add import for `extract_chart_path` in `app.py`**

Find the existing import block at the top of `app.py`. After line 5 (`from src.llm import get_llm`), add:

```python
from src.ui_helpers import extract_chart_path
```

The top of `app.py` should look like:

```python
import gradio as gr
import os
from dotenv import load_dotenv
from src.storage.chroma_store import SermonVectorStore
from src.llm import get_llm
from src.ui_helpers import extract_chart_path
```

- [ ] **Step 2.2 — Update `bot_msg` to detect chart paths and build multimodal content**

Find `bot_msg` (around line 281). Replace its body with:

```python
def bot_msg(history: list, provider):
    if not history or history[-1]["role"] != "user":
        return history

    user_message = history[-1]["content"]
    if isinstance(user_message, list):
        user_message = " ".join(
            [m["text"] for m in user_message if isinstance(m, dict) and m.get("type") == "text"]
        )

    chat_history = history[:-1]
    bot_message = respond(user_message, chat_history, provider)

    text, chart_path = extract_chart_path(bot_message)
    if chart_path:
        content = [
            {"type": "text", "text": text},
            {"type": "image", "url": chart_path},
        ]
    else:
        content = bot_message

    history.append({"role": "assistant", "content": content})
    return history
```

- [ ] **Step 2.3 — Update `gr.Chatbot` to use messages type and fluid height**

Find the `gr.Chatbot(...)` block (around line 206). Replace it with:

```python
            chatbot = gr.Chatbot(
                type="messages",
                min_height=400,
                show_label=False,
                elem_classes="chatbot-container",
                avatar_images=(None, "https://www.bbtc.com.sg/wp-content/uploads/2021/04/BBTC-Logo-Header.png")
            )
```

- [ ] **Step 2.4 — Allow Gradio to serve `/tmp` files**

Find the last line of `app.py`:

```python
    demo.launch(css=custom_css, theme=gr.themes.Default())
```

Replace with:

```python
    demo.launch(css=custom_css, theme=gr.themes.Default(), allowed_paths=["/tmp"])
```

- [ ] **Step 2.5 — Smoke test chart rendering**

```bash
python app.py
```

Open `http://127.0.0.1:7860`. Type: `"Show a bar chart of how many sermons were preached each year"`. Confirm the chart renders as an image inside the assistant bubble, not as a file path string.

- [ ] **Step 2.6 — Commit**

```bash
git add app.py
git commit -m "feat: render matplotlib charts as inline images in chatbot"
```

---

## Task 3: Live stats bar

**Files:**
- Modify: `app.py` (imports, module-level stats fetch, CSS, layout)

- [ ] **Step 3.1 — Expand the import from `src.ui_helpers`**

Update the import added in Task 2 to include the two additional helpers:

```python
from src.ui_helpers import extract_chart_path, fetch_archive_stats, render_stats_bar
```

- [ ] **Step 3.2 — Fetch stats at module load**

Find the `except Exception as e:` block that wraps the initialization (around line 57):

```python
except Exception as e:
    print(f"⚠️ Initialization warning: {e}")
    agent = None
```

Add two lines after it (outside the try/except):

```python
except Exception as e:
    print(f"⚠️ Initialization warning: {e}")
    agent = None

try:
    _stats_bar_html = render_stats_bar(fetch_archive_stats(registry.db_path))
except Exception:
    _stats_bar_html = render_stats_bar(None)
```

- [ ] **Step 3.3 — Add `.stats-bar` CSS to `custom_css`**

Find the closing `"""` of `custom_css` (around line 188). Insert before it:

```css
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
```

- [ ] **Step 3.4 — Add `gr.HTML` stats bar to layout**

Find the line inside `with gr.Blocks() as demo:` that starts the main `gr.Row()` (the row that contains the chat area and sidebar, around line 203):

```python
    with gr.Row():
        # Main Chat Area
```

Insert the stats bar component just before this row:

```python
    gr.HTML(_stats_bar_html)

    with gr.Row():
        # Main Chat Area
```

- [ ] **Step 3.5 — Smoke test stats bar**

```bash
python app.py
```

Open `http://127.0.0.1:7860`. Confirm the stats bar is visible below the title, showing sermon count, speaker count, year range, and language count.

- [ ] **Step 3.6 — Commit**

```bash
git add app.py
git commit -m "feat: add live archive stats bar to app header"
```

---

## Task 4: Loading indicator for Submit button

**Files:**
- Modify: `app.py` (event chains)

- [ ] **Step 4.1 — Replace event chains with loading indicator wiring**

Find the event-handling block near the bottom of the `with gr.Blocks()` section (around lines 294–300):

```python
    msg.submit(user_msg, [msg, chatbot], [msg, chatbot], queue=True).then(
        bot_msg, [chatbot, provider_radio], chatbot
    )
    submit.click(user_msg, [msg, chatbot], [msg, chatbot], queue=True).then(
        bot_msg, [chatbot, provider_radio], chatbot
    )
    clear.click(lambda: [], None, chatbot, queue=False)
```

Replace with:

```python
    disable_submit = lambda: gr.update(value="⏳ Thinking...", interactive=False)
    enable_submit = lambda: gr.update(value="🚀 Send", interactive=True)

    msg.submit(user_msg, [msg, chatbot], [msg, chatbot], queue=True).then(
        disable_submit, None, submit
    ).then(
        bot_msg, [chatbot, provider_radio], chatbot
    ).then(
        enable_submit, None, submit
    )
    submit.click(user_msg, [msg, chatbot], [msg, chatbot], queue=True).then(
        disable_submit, None, submit
    ).then(
        bot_msg, [chatbot, provider_radio], chatbot
    ).then(
        enable_submit, None, submit
    )
    clear.click(lambda: [], None, chatbot, queue=False)
```

- [ ] **Step 4.2 — Smoke test loading indicator**

```bash
python app.py
```

Open `http://127.0.0.1:7860`. Click Send. Confirm the button changes to "⏳ Thinking..." and becomes non-interactive while the agent processes. Confirm it returns to "🚀 Send" when the response arrives.

- [ ] **Step 4.3 — Commit**

```bash
git add app.py
git commit -m "feat: add loading indicator to Submit button during agent processing"
```

---

## Task 5: Replace quick queries and add mobile CSS

**Files:**
- Modify: `app.py` (gr.Examples, custom_css)

- [ ] **Step 5.1 — Replace the 5 example queries with 10**

Find the `gr.Examples(...)` block (around lines 221–231):

```python
            with gr.Row():
                gr.Examples(
                    examples=[
                        ["List the top 3 verses preached each year."],
                        ["Which bible verses were preached most often in 2024?"],
                        ["Summarize the 'Bigger Fire' sermon and its key takeaways."],
                        ["Create a bar chart of how many sermons each speaker gave."],
                        ["Who spoke on the most recent Sunday in the database?"]
                    ],
                    inputs=msg,
                    label="⚡ Quick Inquiries"
                )
```

Replace with:

```python
            with gr.Row():
                gr.Examples(
                    examples=[
                        ["How many sermons are in the archive and who are the top 5 speakers?"],
                        ["Show a bar chart of how many sermons were preached each year"],
                        ["Show a bar chart of the top 10 most-preached Bible books"],
                        ["Create a bar chart of sermon count per speaker"],
                        ["What sermons have been preached on the book of Romans?"],
                        ["Find sermons about forgiveness, grace, and redemption"],
                        ["What have our pastors said about faith during trials and suffering?"],
                        ["Find sermons that cover John 3:16 or the topic of eternal life"],
                        ["Compare what different speakers have said about the Holy Spirit"],
                        ["What was the most recent sermon and what were its key points?"],
                    ],
                    inputs=msg,
                    label="⚡ Quick Inquiries"
                )
```

- [ ] **Step 5.2 — Add mobile responsive CSS to `custom_css`**

Find the `.stats-bar` block you added in Task 3. Add the following media query block directly after it, before the closing `"""`:

```css
@media (max-width: 768px) {
    .gradio-container {
        max-width: 100% !important;
    }
    .sidebar {
        border-right: none !important;
        border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    .stats-bar {
        flex-direction: column;
        gap: 6px;
        align-items: flex-start;
    }
    #title-text h1 {
        font-size: 1.8rem;
    }
    .message-user, .message-assistant {
        font-size: 0.9rem;
        padding: 10px 12px !important;
    }
}
```

- [ ] **Step 5.3 — Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS (includes the 15 new ui_helpers tests plus existing reranker and vector_tool tests).

- [ ] **Step 5.4 — Final smoke test**

```bash
python app.py
```

Open `http://127.0.0.1:7860` and verify all five features end-to-end:

1. **Stats bar** — visible below title with live counts
2. **10 quick queries** — visible in the Quick Inquiries panel
3. **Chart rendering** — click "Show a bar chart of how many sermons were preached each year", confirm inline image appears in chat
4. **Loading indicator** — button shows "⏳ Thinking..." during processing, reverts to "🚀 Send" after
5. **Fluid chatbot height** — resize browser window, confirm chatbot grows/shrinks

- [ ] **Step 5.5 — Commit**

```bash
git add app.py
git commit -m "feat: add 10 quick queries, mobile CSS, and fluid chatbot height"
```
