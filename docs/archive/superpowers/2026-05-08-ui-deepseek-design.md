# UI Redesign + DeepSeek Model Option

**Date:** 2026-05-08  
**Scope:** `app.py`, `src/llm.py`

---

## Summary

Two changes:

1. **Add `deepseek-v4-flash:cloud` as a selectable inference model** alongside the existing `macdev/gpt-oss20b-large-ctx` Ollama model. Both use `ChatOllama` — no new API keys required.
2. **Redesign the Gradio UI** with a terminal-CLI light theme: macOS-style window chrome, monospace font throughout, prompt-prefixed chat bubbles, and horizontal quick-query pills replacing the old Examples table.

---

## 1. LLM Layer (`src/llm.py`)

### Changes

Add two named constants for Ollama model tags:

```python
OLLAMA_LOCAL_MODEL = "macdev/gpt-oss20b-large-ctx"
OLLAMA_DEEPSEEK_MODEL = "deepseek-v4-flash:cloud"
```

Change `get_llm()` signature to accept `provider` values of `"ollama_local"`, `"ollama_deepseek"`, `"groq"`, `"gemini"`. The fallback (default) remains `"ollama_local"`.

```python
def get_llm(provider="ollama_local", temperature=0):
    if provider == "groq": ...
    if provider == "gemini": ...
    model = OLLAMA_DEEPSEEK_MODEL if provider == "ollama_deepseek" else OLLAMA_LOCAL_MODEL
    return ChatOllama(model=model, temperature=temperature)
```

The `ollama_model` keyword argument is removed (no callers outside `app.py`).

---

## 2. App Layer (`app.py`)

### 2a. Provider mapping

Radio choices → provider keys:

| Radio label | Provider key |
|---|---|
| `GPT-OSS 20B [local]` | `"ollama_local"` |
| `DeepSeek V4 Flash [cloud]` | `"ollama_deepseek"` |
| `Groq [cloud]` | `"groq"` |
| `Gemini 3 Flash [cloud]` | `"gemini"` |

`_on_provider_change()` maps the radio string to the key above.

`get_agent()` uses `provider` as the cache key (unchanged structure, new key values).

Startup pre-warms `"ollama_local"` only (DeepSeek cloud model is not pre-warmed to avoid a cold start on a remote endpoint at boot).

### 2b. Inference badge

`_inference_badge_html()` gains cases for `"ollama_local"` and `"ollama_deepseek"`. Both resolve to `_ollama_status`. Labels: `"gpt-oss-20b · local"` and `"deepseek-v4-flash · cloud"`.

---

## 3. UI Theme (`app.py` — `custom_css` + layout)

### Design language

Terminal-CLI light theme. Reference mockup: `.superpowers/brainstorm/…/content/full-ui-mockup-v5-final.html`.

- **Background:** `#f5f5f0` (warm off-white)
- **Panels:** `#fafaf7` with `1px solid #c8c8c0` border, `2px 2px 0 #d4d4cc` stacked shadow (no blur)
- **Font:** `Source Code Pro` everywhere — body, inputs, buttons, labels, messages
- **Base font size:** `15px`
- **Accent:** `#16a34a` green (send button, prompt chars, online badges)
- **Selection accent:** `#7c3aed` purple (selected model option left border)

### Terminal chrome

Each major panel (header, chat area, sidebar) gets a title bar with macOS-style traffic-light dots (red/yellow/green) and a centered title label. Implemented as an `elem_id`-targeted CSS wrapper.

### Chat messages

User messages prefixed `user>`, assistant messages prefixed `agt>`. Implemented by wrapping the Gradio chatbot in a CSS layer — the bubble styling via `.message-user` / `.message-assistant` classes.

### Input bar

Single flush row: `$>` prefix tile (grey background, left border) → textarea → `[ Send ]` button (green, no border-radius). No gap between elements; they form one connected bar.

### Quick Query pills (replaces `gr.Examples`)

Replace `gr.Examples` with `gr.HTML` (static pill bar) + a `gr.State` holding the full query mapping, and 10 `gr.Button` components hidden behind the HTML via Gradio's `elem_id` + CSS, **OR** use `gr.Examples` with `example_labels` (available in Gradio 6.x) for short display labels while injecting full queries.

Preferred approach: `gr.Examples(examples=[...], inputs=msg, example_labels=[...])` with CSS overriding the default table layout to `display: flex; flex-wrap: wrap; gap: 7px` and pill styling per category class.

Pill categories and colors:

| Category | CSS class | Background | Border | Text |
|---|---|---|---|---|
| Analytics | `.pill.analytics` | `#ede9fe` | `#a78bfa` | `#3b0764` |
| Semantic | `.pill.semantic` | `#dcfce7` | `#4ade80` | `#14532d` |
| Bible | `.pill.bible` | `#fef9c3` | `#facc15` | `#713f12` |
| People | `.pill.people` | `#fee2e2` | `#f87171` | `#7f1d1d` |
| Content | `.pill.content` | `#e0f2fe` | `#38bdf8` | `#0c4a6e` |

Pill labels (short) → full queries:

| Label | Category | Full query |
|---|---|---|
| 📊 Scripture Coverage | analytics | "Scripture Coverage: Generate a frequency heatmap of the most frequently preached Bible books." |
| 📖 Gap Analysis | analytics | "Gap Analysis: List all Bible books that have never been preached in BBTC sermons." |
| 📈 Ministry Shifts | analytics | "Semantic Analysis: Identify shifts in ministry emphasis within BBTC over the last 5 years." |
| 🔍 Spiritual Warfare | semantic | "Semantic Search: Find the top 3 sermons related to 'Spiritual Warfare' from 2024 to 2026." |
| ✝️ BBTC Theology | semantic | "BBTC Theology: Explain the biblical sequence of End Times events based on BBTC teachings." |
| 🧭 Theological Themes | semantic | "Identify the consistent theological themes in BBTC's vision statements and pulpit series between 2015 and 2026" |
| 📜 Bible Versions | bible | "Bible Translation: List all Bible translations of 1 John 1:9 in the bible archives." |
| 📖 Bible Passages | bible | "Find Bible passages about forgiveness and grace using the Bible archive." |
| 👤 SP Chua Sermons | people | "Speaker Filter: List all sermons delivered by SP Chua Seng Lee in the year 2026." |
| 📝 Last Week's Sermon | content | "Specific Sermon: Summarize the key message and scripture shared in last week's sermon." |

### Sidebar radio (`gr.Radio`)

```python
gr.Radio(
    choices=[
        "GPT-OSS 20B [local]",
        "DeepSeek V4 Flash [cloud]",
        "Groq [cloud]",
        "Gemini 3 Flash [cloud]",
    ],
    value="GPT-OSS 20B [local]",
    ...
)
```

CSS makes each option a styled card with a left accent border when selected (`border-left: 3px solid #7c3aed`).

---

## 4. Files Changed

| File | Change |
|---|---|
| `src/llm.py` | Add `OLLAMA_LOCAL_MODEL`, `OLLAMA_DEEPSEEK_MODEL`; update `get_llm()` signature |
| `app.py` | New `custom_css` (terminal light theme); updated radio choices, provider mapping, badge logic, `gr.Examples` with `example_labels` |
| `tests/test_llm.py` | Replace `test_get_llm_uses_custom_model` (uses removed `ollama_model` kwarg) with two new tests: `test_get_llm_ollama_local_model` and `test_get_llm_ollama_deepseek_model` that assert the correct model string is passed for each provider key |

No new files. No changes to ingestion, storage, or agent tools.

---

## 5. Out of Scope

- Changes to agent tools, prompts, or retrieval logic
- New API integrations (DeepSeek uses existing Ollama runtime)
- Dark/light theme toggle (light terminal theme replaces dark theme entirely)
