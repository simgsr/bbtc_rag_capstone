# UI Redesign: Professional, User-Friendly BBTC Sermon Intelligence

**Date:** 2026-04-26  
**Approach:** Focused Enhancement (Approach A)  
**Scope:** `app.py` only — no changes to tools, agent, or backend

---

## Goals

1. Inline chart rendering — charts appear as images inside chat bubbles
2. Live stats bar — sermon count, speaker count, year range shown at top
3. Loading indicator — visible feedback while agent is processing (5–15s)
4. 10 meaningful quick queries — sermon stats, biblical content, specific searches
5. Mobile responsiveness — sidebar stacks, chatbot height is fluid, queries wrap

---

## Section 1: Inline Chart Rendering

**Problem:** `matplotlib_tool` returns a file path string (e.g. `/tmp/bbtc_chart_abc123.png`). The agent echoes this path as plain text — no image ever renders.

**Solution:** In `bot_msg`, after `respond()` returns, scan the response string for the pattern `/tmp/bbtc_chart_[a-f0-9]+\.png` using `re.search`. If a match is found:
- Strip the raw path from the surrounding text
- Reconstruct the chatbot message as a multimodal content list:

```python
{"role": "assistant", "content": [
    {"type": "text",  "text": "<agent explanation text>"},
    {"type": "image", "url": "/tmp/bbtc_chart_abc123.png"}
]}
```

Gradio's `gr.Chatbot` renders `{"type": "image", "url": ...}` content blocks as inline images natively. No changes to `matplotlib_tool`, the agent tools list, or the system prompt.

**Edge cases:**
- After stripping the path, call `.strip()` on the remaining text and remove any trailing colon or space artifacts before using it as the label
- If the stripped text is empty, use the default label: `"Here is the chart:"`
- If the regex finds no match, fall back to plain text as before

---

## Section 2: Live Stats Bar

A slim `gr.HTML` component placed between the title header and the main chat row. Populated once at app startup.

**Queries (run at module load, before `gr.Blocks`):**
```sql
SELECT COUNT(*) FROM sermons;
SELECT COUNT(DISTINCT speaker) FROM sermons WHERE speaker IS NOT NULL;
SELECT MIN(year), MAX(year) FROM sermons WHERE year IS NOT NULL;
SELECT COUNT(DISTINCT language) FROM sermons WHERE language IS NOT NULL AND language != '';
```

**Rendered output (example):**
```
📚 847 sermons  ·  👤 14 speakers  ·  📅 2018 – 2024  ·  🌐 2 languages
```

**Styling:** Horizontal pill bar using existing dark theme colors (`#1e293b` background, `#94a3b8` text, subtle border). Wraps to vertical stack on narrow screens.

**Fallback:** If DB is unavailable at startup, renders `"📚 Archive stats unavailable"` without crashing the app.

---

## Section 3: Loading Indicator

Chain Gradio event handlers to mutate the Submit button across the request lifecycle:

```
user clicks Send
  → user_msg()         — appends user bubble instantly, clears input
  → submit button:     gr.update(value="⏳ Thinking...", interactive=False)
  → bot_msg()          — agent processes (5–15s)
  → submit button:     gr.update(value="🚀 Send", interactive=True)
```

Implementation uses two thin lambdas chained via `.then()` on both `msg.submit` and `submit.click` event chains. No extra components added.

---

## Section 4: Quick Queries (10 flat list)

Replaces the current 5 examples. Covers four query categories:

**Sermon Stats & Charts**
1. `"How many sermons are in the archive and who are the top 5 speakers?"`
2. `"Show a bar chart of how many sermons were preached each year"`
3. `"Show a bar chart of the top 10 most-preached Bible books"`
4. `"Create a bar chart of sermon count per speaker"`

**Biblical Content**
5. `"What sermons have been preached on the book of Romans?"`
6. `"Find sermons about forgiveness, grace, and redemption"`
7. `"What have our pastors said about faith during trials and suffering?"`
8. `"Find sermons that cover John 3:16 or the topic of eternal life"`

**Specific Searches**
9. `"Compare what different speakers have said about the Holy Spirit"`
10. `"What was the most recent sermon and what were its key points?"`

---

## Section 5: Mobile Responsiveness & CSS Polish

**Changes to `custom_css`:**
- Add `@media (max-width: 768px)` block:
  - `.sidebar` stacks below chat (no longer side-by-side)
  - Font sizes reduce (~10%)
  - Padding and gap values tighten
  - Stats bar pills wrap to column
  - Quick queries grid wraps to single column

**Change to `gr.Chatbot`:**
- Replace fixed `height=600` with `min_height=400` so the component grows with content on desktop and avoids overflow on mobile

---

## Files Changed

| File | Change |
|------|--------|
| `app.py` | All changes — stats bar queries, `bot_msg` image detection, loading indicator event chains, quick queries list, CSS additions |

No other files are modified.

---

## Out of Scope

- Streaming token-by-token responses (separate task)
- "Analytics" or "Browse Sermons" tabs (Approach B — future work)
- New chart types in `matplotlib_tool` (separate task)
- Bible collection population in ChromaDB (separate task)
