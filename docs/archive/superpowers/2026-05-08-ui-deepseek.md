# UI Redesign + DeepSeek Model Option — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `deepseek-v4-flash:cloud` as a selectable Ollama inference model and replace the dark Gradio UI with a terminal-CLI light theme (Source Code Pro throughout, macOS window chrome, horizontal quick-query pills).

**Architecture:** Three self-contained changes — (1) `src/llm.py` gets two named Ollama model constants and a cleaner `provider` key scheme; (2) `app.py` updates the provider mapping, radio choices, and badge logic; (3) `app.py` replaces the entire `custom_css` block and the `gr.Examples` call with the terminal-light theme and pill layout.

**Tech Stack:** Python, LangChain (`ChatOllama`, `ChatGroq`, `ChatGoogleGenerativeAI`), Gradio 6.14, Source Code Pro (Google Fonts), pytest

---

## File Map

| File | Change |
|---|---|
| `src/llm.py` | Add `OLLAMA_LOCAL_MODEL`, `OLLAMA_DEEPSEEK_MODEL`; change `get_llm()` to use `provider` keys instead of `ollama_model` kwarg |
| `tests/test_llm.py` | Remove `test_get_llm_uses_custom_model`; add `test_get_llm_ollama_local_model` and `test_get_llm_ollama_deepseek_model` |
| `app.py` | Update provider mapping, radio, badge; replace `custom_css`; update `gr.Examples` with `example_labels` |

---

## Task 1: Update LLM provider constants and signature

**Files:**
- Modify: `src/llm.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1.1: Write two failing tests for the new provider keys**

Replace the contents of `tests/test_llm.py` with:

```python
from unittest.mock import patch, MagicMock
from langchain_ollama import ChatOllama


def test_get_llm_returns_chat_ollama():
    with patch("src.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm
        get_llm()
        mock_cls.assert_called_once()


def test_get_llm_passes_temperature():
    with patch("src.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm
        get_llm(temperature=0.5)
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("temperature") == 0.5


def test_get_llm_ollama_local_model():
    with patch("src.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm, OLLAMA_LOCAL_MODEL
        get_llm(provider="ollama_local")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("model") == OLLAMA_LOCAL_MODEL


def test_get_llm_ollama_deepseek_model():
    with patch("src.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm, OLLAMA_DEEPSEEK_MODEL
        get_llm(provider="ollama_deepseek")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("model") == OLLAMA_DEEPSEEK_MODEL
```

- [ ] **Step 1.2: Run tests — expect 2 failures**

```bash
pytest tests/test_llm.py -v
```

Expected: `test_get_llm_ollama_local_model` FAIL (no `OLLAMA_LOCAL_MODEL`), `test_get_llm_ollama_deepseek_model` FAIL (no `OLLAMA_DEEPSEEK_MODEL`).

- [ ] **Step 1.3: Update `src/llm.py`**

Replace the full file:

```python
from langchain_ollama import ChatOllama
import os

GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-3-flash-preview"
OLLAMA_LOCAL_MODEL = "macdev/gpt-oss20b-large-ctx"
OLLAMA_DEEPSEEK_MODEL = "deepseek-v4-flash:cloud"


def get_llm(provider="ollama_local", temperature=0):
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        return ChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=api_key)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env")
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL, temperature=temperature, google_api_key=api_key
        )

    model = OLLAMA_DEEPSEEK_MODEL if provider == "ollama_deepseek" else OLLAMA_LOCAL_MODEL
    return ChatOllama(model=model, temperature=temperature)
```

- [ ] **Step 1.4: Run all tests — expect all pass**

```bash
pytest tests/test_llm.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add src/llm.py tests/test_llm.py
git commit -m "feat: add deepseek ollama provider key to get_llm"
```

---

## Task 2: Update provider mapping in app.py

**Files:**
- Modify: `app.py` (lines ~8, ~141-152, ~171-188, ~602-608, ~622-629)

- [ ] **Step 2.1: Update the import line at the top of app.py**

Find:
```python
from src.llm import get_llm, GROQ_MODEL
```

Replace with:
```python
from src.llm import get_llm, GROQ_MODEL, GEMINI_MODEL, OLLAMA_LOCAL_MODEL, OLLAMA_DEEPSEEK_MODEL
```

- [ ] **Step 2.2: Update `get_agent()` and startup pre-warm**

Find:
```python
    def get_agent(provider: str = "ollama"):
        if provider not in _agent_cache:
            _llm = get_llm(provider=provider, temperature=0.1)
            _agent_cache[provider] = create_react_agent(
                _llm,
                tools=[sql_tool, vector_tool, viz_tool, get_bible_versions_tool, search_bible_tool],
                prompt=SystemMessage(content=SYSTEM_PROMPT),
            )
        return _agent_cache[provider]

    # Pre-warm Ollama agent at startup
    get_agent("ollama")
```

Replace with:
```python
    def get_agent(provider: str = "ollama_local"):
        if provider not in _agent_cache:
            _llm = get_llm(provider=provider, temperature=0.1)
            _agent_cache[provider] = create_react_agent(
                _llm,
                tools=[sql_tool, vector_tool, viz_tool, get_bible_versions_tool, search_bible_tool],
                prompt=SystemMessage(content=SYSTEM_PROMPT),
            )
        return _agent_cache[provider]

    # Pre-warm local Ollama agent at startup only
    get_agent("ollama_local")
```

- [ ] **Step 2.3: Update `_inference_badge_html()`**

Find the entire `_inference_badge_html` function:
```python
def _inference_badge_html(provider: str) -> str:
    if provider == "groq":
        has_key = bool(os.getenv("GROQ_API_KEY"))
        status = "online" if has_key else "offline"
        label = f"groq · {GROQ_MODEL}" if has_key else "groq · no key"
    elif provider == "gemini":
        has_key = bool(os.getenv("GOOGLE_API_KEY"))
        status = "online" if has_key else "offline"
        label = "gemini · 2.5 pro" if has_key else "gemini · no key"
    else:
        status = _ollama_status
        label = "ollama · local"
    return (
        "<div style='display:flex;justify-content:space-between;align-items:center;margin-top:8px;'>"
        f"<span style='color:#94a3b8;'>Inference</span>"
        f"<span class='status-badge status-{status}'>{label}</span>"
        "</div>"
    )
```

Replace with:
```python
def _inference_badge_html(provider: str) -> str:
    if provider == "groq":
        has_key = bool(os.getenv("GROQ_API_KEY"))
        status = "online" if has_key else "offline"
        label = f"groq · {GROQ_MODEL}" if has_key else "groq · no key"
    elif provider == "gemini":
        has_key = bool(os.getenv("GOOGLE_API_KEY"))
        status = "online" if has_key else "offline"
        label = f"gemini · {GEMINI_MODEL}" if has_key else "gemini · no key"
    elif provider == "ollama_deepseek":
        status = _ollama_status
        label = f"deepseek-v4-flash · cloud"
    else:  # ollama_local
        status = _ollama_status
        label = "gpt-oss-20b · local"
    return (
        "<div style='display:flex;justify-content:space-between;align-items:center;margin-top:8px;'>"
        f"<span style='color:#555;font-family:\"Source Code Pro\",monospace;font-size:0.72rem;'>inference</span>"
        f"<span class='status-badge status-{status}'>{label}</span>"
        "</div>"
    )
```

- [ ] **Step 2.4: Update the radio and provider_state in the Gradio layout**

Find:
```python
            provider_radio = gr.Radio(
                choices=["Ollama (local)", "Groq (cloud)", "Gemini (cloud)"],
                value="Ollama (local)",
                show_label=False,
                interactive=True,
            )
            provider_state = gr.State("ollama")
```

Replace with:
```python
            provider_radio = gr.Radio(
                choices=[
                    "GPT-OSS 20B [local]",
                    "DeepSeek V4 Flash [cloud]",
                    "Groq [cloud]",
                    "Gemini 3 Flash [cloud]",
                ],
                value="GPT-OSS 20B [local]",
                show_label=False,
                interactive=True,
                elem_id="model-radio",
            )
            provider_state = gr.State("ollama_local")
```

- [ ] **Step 2.5: Update `_on_provider_change()`**

Find:
```python
    def _on_provider_change(radio_val):
        if "Groq" in radio_val:
            provider = "groq"
        elif "Gemini" in radio_val:
            provider = "gemini"
        else:
            provider = "ollama"
        return provider, _inference_badge_html(provider)
```

Replace with:
```python
    def _on_provider_change(radio_val):
        if "Groq" in radio_val:
            provider = "groq"
        elif "Gemini" in radio_val:
            provider = "gemini"
        elif "DeepSeek" in radio_val:
            provider = "ollama_deepseek"
        else:
            provider = "ollama_local"
        return provider, _inference_badge_html(provider)
```

- [ ] **Step 2.6: Update the inference_status initial render**

Find:
```python
            inference_status = gr.HTML(_inference_badge_html("ollama"))
```

Replace with:
```python
            inference_status = gr.HTML(_inference_badge_html("ollama_local"))
```

- [ ] **Step 2.7: Smoke-test by launching the app**

```bash
python app.py
```

Open the UI. Verify: 4 radio options appear, switching between them updates the inference badge, "GPT-OSS 20B [local]" is selected by default.

- [ ] **Step 2.8: Commit**

```bash
git add app.py
git commit -m "feat: add DeepSeek V4 Flash as inference option in UI"
```

---

## Task 3: Apply terminal-CLI light theme CSS

**Files:**
- Modify: `app.py` — replace `custom_css` string (lines ~292-532)

- [ ] **Step 3.1: Replace the entire `custom_css` variable**

Find the line `custom_css = """` through the closing `"""` and replace with:

```python
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

footer { visibility: hidden }
* { box-sizing: border-box; }

body {
    background: #f5f5f0 !important;
}

.gradio-container {
    background: transparent !important;
    color: #1a1a1a;
    font-family: 'Source Code Pro', monospace !important;
    max-width: 1440px !important;
    font-size: 15px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #f0f0eb; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.35); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.55); }

/* Header */
#title-container {
    padding: 14px 20px;
    background: #fafaf7;
    border: 1px solid #c8c8c0;
    border-radius: 6px;
    box-shadow: 2px 2px 0 #d4d4cc;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 10px;
}
#title-container img {
    height: 36px;
    opacity: 0.95;
}
#title-text h1 {
    font-family: 'Source Code Pro', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0 0 2px 0;
    background: linear-gradient(108deg, #2563eb 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.3px;
    line-height: 1.2;
}
#title-text p {
    color: #888;
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0;
    font-family: 'Source Code Pro', monospace;
}

/* Stats bar */
.stats-bar {
    display: flex;
    gap: 0;
    background: #fafaf7;
    border: 1px solid #c8c8c0;
    border-radius: 4px;
    padding: 8px 16px;
    margin-bottom: 12px;
    color: #555;
    font-size: 0.78rem;
    letter-spacing: 0.3px;
    font-family: 'Source Code Pro', monospace;
    box-shadow: 1px 1px 0 #e0e0d8;
}

/* Chat area */
.chatbot-container {
    border-radius: 4px !important;
    border: 1px solid #c8c8c0 !important;
    background: #fafaf7 !important;
    box-shadow: 2px 2px 0 #d4d4cc !important;
    overflow: hidden !important;
}

/* Messages */
.message-user {
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 0 6px 6px 6px !important;
    padding: 9px 14px !important;
    color: #1e3a8a !important;
    font-size: 0.84rem !important;
    font-family: 'Source Code Pro', monospace !important;
    line-height: 1.6;
}
.message-assistant {
    background: #f0fdf4 !important;
    border: 1px solid #bbf7d0 !important;
    border-radius: 0 6px 6px 6px !important;
    padding: 9px 14px !important;
    color: #14532d !important;
    font-size: 0.84rem !important;
    font-family: 'Source Code Pro', monospace !important;
    line-height: 1.65;
}

/* Plotly chart */
.gradio-plot {
    border-radius: 4px !important;
    overflow: hidden !important;
    border: 1px solid #c8c8c0 !important;
    margin-top: 8px !important;
    background: #fafaf7 !important;
}

/* Input row */
.input-container {
    background: #fafaf7 !important;
    border: 1px solid #c8c8c0 !important;
    border-radius: 4px !important;
    margin-top: 8px !important;
    padding: 0 !important;
    overflow: hidden;
    box-shadow: 1px 1px 0 #e0e0d8;
}
.input-container:focus-within {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.1) !important;
}
.input-container textarea {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.84rem !important;
    color: #1a1a1a !important;
    background: transparent !important;
    padding: 10px 14px !important;
}
.input-container textarea::placeholder {
    color: #aaa !important;
}

/* Submit button */
.btn-primary {
    background: #16a34a !important;
    border: none !important;
    border-left: 1px solid #15803d !important;
    border-radius: 0 !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 0.7rem !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: 'Source Code Pro', monospace !important;
    box-shadow: none !important;
    transition: background 0.15s ease;
}
.btn-primary:hover {
    background: #15803d !important;
    transform: none !important;
    filter: none !important;
    box-shadow: none !important;
}
.btn-primary:active {
    background: #166534 !important;
}

/* Secondary button (Clear) */
button.secondary {
    background: #f5f5f0 !important;
    border: 1px solid #c8c8c0 !important;
    color: #555 !important;
    border-radius: 3px !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    transition: all 0.12s ease;
}
button.secondary:hover {
    background: #fee2e2 !important;
    border-color: #f87171 !important;
    color: #7f1d1d !important;
}

/* Sidebar */
.sidebar {
    background: #fafaf7 !important;
    border: 1px solid #c8c8c0 !important;
    padding: 18px !important;
    border-radius: 4px !important;
    box-shadow: 2px 2px 0 #d4d4cc !important;
}
.sidebar h3 {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #888 !important;
    margin-bottom: 12px !important;
}
.sidebar p, .sidebar li {
    font-size: 0.8rem !important;
    color: #555 !important;
    line-height: 1.6 !important;
    font-family: 'Source Code Pro', monospace !important;
}

/* Status badges */
.status-badge {
    padding: 2px 8px;
    border-radius: 2px;
    font-size: 0.56rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Source Code Pro', monospace;
    border: 1px solid;
}
.status-online {
    background: #dcfce7;
    color: #166534;
    border-color: #4ade80;
}
.status-offline {
    background: #fee2e2;
    color: #7f1d1d;
    border-color: #f87171;
}

/* Radio model cards */
#model-radio label {
    background: #fafaf7 !important;
    border: 1px solid #e0e0d8 !important;
    border-radius: 3px !important;
    padding: 7px 10px !important;
    margin-bottom: 5px !important;
    cursor: pointer !important;
    transition: all 0.12s ease !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.78rem !important;
    color: #1a1a1a !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
#model-radio label:has(input:checked) {
    background: #ede9fe !important;
    border-color: #7c3aed !important;
    border-left: 3px solid #7c3aed !important;
}
#model-radio label:hover:not(:has(input:checked)) {
    background: #f5f3ff !important;
    border-color: #a78bfa !important;
}
"""
```

- [ ] **Step 3.2: Launch app and visually verify the theme**

```bash
python app.py
```

Check: background is off-white `#f5f5f0`, all text is Source Code Pro, chat bubbles are blue (user) and green (bot), buttons are green, clear button turns red on hover, selected model card has purple left border.

- [ ] **Step 3.3: Commit**

```bash
git add app.py
git commit -m "feat: apply terminal-CLI light theme to Gradio UI"
```

---

## Task 4: Quick Query pills

**Files:**
- Modify: `app.py` — replace `gr.Examples` call

- [ ] **Step 4.1: Add query constants above the Gradio layout block**

Find the line `with gr.Blocks(title="BBTC Sermon Intelligence") as demo:` and insert **above** it:

```python
_QUICK_QUERY_LABELS = [
    "📊 Scripture Coverage",
    "📖 Gap Analysis",
    "📈 Ministry Shifts",
    "🔍 Spiritual Warfare",
    "✝️ BBTC Theology",
    "🧭 Theological Themes",
    "📜 Bible Versions",
    "📖 Bible Passages",
    "👤 SP Chua Sermons",
    "📝 Last Week's Sermon",
]

_QUICK_QUERY_FULL = [
    ["Scripture Coverage: Generate a frequency heatmap of the most frequently preached Bible books."],
    ["Gap Analysis: List all Bible books that have never been preached in BBTC sermons."],
    ["Semantic Analysis: Identify shifts in ministry emphasis within BBTC over the last 5 years."],
    ["Semantic Search: Find the top 3 sermons related to 'Spiritual Warfare' from 2024 to 2026."],
    ["BBTC Theology: Explain the biblical sequence of End Times events based on BBTC teachings."],
    ["Identify the consistent theological themes in BBTC's vision statements and pulpit series between 2015 and 2026"],
    ["Bible Translation: List all Bible translations of 1 John 1:9 in the bible archives."],
    ["Find Bible passages about forgiveness and grace using the Bible archive."],
    ["Speaker Filter: List all sermons delivered by SP Chua Seng Lee in the year 2026."],
    ["Specific Sermon: Summarize the key message and scripture shared in last week's sermon."],
]
```

- [ ] **Step 4.2: Replace `gr.Examples` call**

Find the existing `gr.Examples(...)` block:
```python
            gr.Examples(
                examples=[
                    ["Scripture Coverage: Generate a frequency heatmap of the most frequently preached Bible books."],
                    ["Identify the consistent theological themes in BBTC's vision statements and pulpit series between 2015 and 2026"],
                    ["Gap Analysis: List all Bible books that have never been preached in BBTC sermons."],
                    ["Semantic Analysis: Identify shifts in ministry emphasis within BBTC over the last 5 years."],
                    ["BBTC Theology: Explain the biblical sequence of End Times events based on BBTC teachings."],
                    ["Semantic Search: Find the top 3 sermons related to 'Spiritual Warfare' from 2024 to 2026."],
                    ["Specific Sermon: Content: Summarize the key message and scripture shared in last week's sermon."],
                    ["Bible Translation: List all Bible translations of 1 John 1:9 in the bible archives."],
                    ["Speaker Filter: List all sermons delivered by SP Chua Seng Lee in the year 2026."],
                    ["Categorization: Search for all mentions of 'Mental Health' and categorize the biblical advice given."]
                ],
                inputs=msg,
                label="Example questions"
            )
```

Replace with:
```python
            gr.Examples(
                examples=_QUICK_QUERY_FULL,
                inputs=msg,
                example_labels=_QUICK_QUERY_LABELS,
                label=None,
                elem_id="quick-query-pills",
            )
```

- [ ] **Step 4.3: Append pill CSS to `custom_css`**

Find the closing `"""` of `custom_css` and insert the following **before** it:

```css

/* Quick query pills */
#quick-query-pills .examples-holder,
#quick-query-pills table,
#quick-query-pills tbody,
#quick-query-pills tr {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
    border: none !important;
    padding: 4px 0 !important;
    margin: 0 !important;
    background: transparent !important;
}
#quick-query-pills td { border: none !important; padding: 0 !important; }
#quick-query-pills .label-wrap { display: none !important; }
#quick-query-pills .example {
    border-radius: 2px !important;
    padding: 5px 13px !important;
    font-size: 0.73rem !important;
    font-family: 'Source Code Pro', monospace !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    transition: all 0.12s ease !important;
    line-height: 1.4 !important;
}
#quick-query-pills .example:hover {
    filter: brightness(0.92) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
}
/* Analytics — purple (examples 1-3) */
#quick-query-pills .example:nth-child(1),
#quick-query-pills .example:nth-child(2),
#quick-query-pills .example:nth-child(3) {
    background: #ede9fe !important; border: 1px solid #a78bfa !important; color: #3b0764 !important;
}
/* Semantic — green (examples 4-6) */
#quick-query-pills .example:nth-child(4),
#quick-query-pills .example:nth-child(5),
#quick-query-pills .example:nth-child(6) {
    background: #dcfce7 !important; border: 1px solid #4ade80 !important; color: #14532d !important;
}
/* Bible — amber (examples 7-8) */
#quick-query-pills .example:nth-child(7),
#quick-query-pills .example:nth-child(8) {
    background: #fef9c3 !important; border: 1px solid #facc15 !important; color: #713f12 !important;
}
/* People — red (example 9) */
#quick-query-pills .example:nth-child(9) {
    background: #fee2e2 !important; border: 1px solid #f87171 !important; color: #7f1d1d !important;
}
/* Content — blue (example 10) */
#quick-query-pills .example:nth-child(10) {
    background: #e0f2fe !important; border: 1px solid #38bdf8 !important; color: #0c4a6e !important;
}
```

- [ ] **Step 4.4: Launch and verify pills**

```bash
python app.py
```

Check: 10 pills render in a horizontal wrapping row below the input bar; each is color-coded by category; clicking a pill injects the full query into the textbox; the old "Example questions" table heading is gone.

If Gradio's internal class names differ (e.g., `.example` → `.example-btn`), inspect the DOM in browser devtools, find the actual class on each pill button, and update the CSS selectors accordingly. The `nth-child` logic remains the same.

- [ ] **Step 4.5: Commit**

```bash
git add app.py
git commit -m "feat: replace Examples table with terminal-style quick-query pills"
```

---

## Self-Review

**Spec coverage:**
- ✅ `deepseek-v4-flash:cloud` added as Ollama option — Task 1 + Task 2
- ✅ `OLLAMA_LOCAL_MODEL` / `OLLAMA_DEEPSEEK_MODEL` constants — Task 1
- ✅ `test_llm.py` updated — Task 1
- ✅ Radio choices updated with 4 named options — Task 2
- ✅ `_inference_badge_html` updated — Task 2
- ✅ `get_agent()` default and startup pre-warm updated — Task 2
- ✅ Terminal-CLI light theme CSS — Task 3
- ✅ Quick query pills via `example_labels` — Task 4
- ✅ Source Code Pro throughout — Task 3

**No placeholders found.**

**Type consistency:** `provider` string values (`"ollama_local"`, `"ollama_deepseek"`, `"groq"`, `"gemini"`) are consistent across `get_llm()`, `get_agent()`, `_on_provider_change()`, `_inference_badge_html()`, and `provider_state` initial value.
