import gradio as gr
import os
import subprocess
import time
import urllib.request
from dotenv import load_dotenv
from src.storage.chroma_store import SermonVectorStore
from src.llm import get_llm, GROQ_MODEL, GEMINI_MODEL, OLLAMA_LOCAL_MODEL, OLLAMA_DEEPSEEK_MODEL
from src.ui_helpers import extract_chart_path, fetch_archive_stats, render_stats_bar
from src.storage.sqlite_store import SermonRegistry
from src.tools.sql_tool import make_sql_tool
from src.tools.vector_tool import make_vector_tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from src.tools.viz_tool import make_viz_tool
from src.tools.bible_tool import make_bible_tool
import plotly.io as pio
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Ensure Plotly uses light template for consistency
pio.templates.default = "plotly_white"

def _ensure_ollama(timeout: int = 20) -> bool:
    def _is_up() -> bool:
        try:
            urllib.request.urlopen("http://127.0.0.1:11434", timeout=2)
            return True
        except Exception:
            return False

    if _is_up():
        return True

    print("🦙 Ollama not running — starting it now...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1)
        if _is_up():
            print("🦙 Ollama is ready.")
            return True

    print("⚠️  Ollama did not start within the timeout.")
    return False
_ensure_ollama()

try:
    registry = SermonRegistry()
    vector_store = SermonVectorStore()

    sql_tool = make_sql_tool(registry.db_path)
    vector_tool = make_vector_tool(vector_store)
    viz_tool = make_viz_tool(registry)
    get_bible_versions_tool, search_bible_tool = make_bible_tool(vector_store)

    from datetime import date as _date
    _today = _date.today().isoformat()
    _cur_year = _date.today().year

    SYSTEM_PROMPT = (
        f"You are the BBTC Sermon Intelligence Assistant for Bethesda Bedok-Tampines Church.\n"
        f"Today is {_today}. Archive covers 2015–{_cur_year}. "
        f"'Last N years' → min_year = {_cur_year} - N + 1.\n\n"

        "## CRITICAL RULES — follow these before doing anything else\n"
        "1. YOU MUST CALL A TOOL BEFORE GIVING ANY ANSWER. No exceptions.\n"
        "2. NEVER answer from memory, training data, or prior knowledge. "
        "Every fact, name, number, verse, or date in your reply MUST come from a tool result in THIS conversation.\n"
        "3. If you have not called a tool yet in this turn, STOP and call one now.\n"
        "4. If tools return no results, say 'I found no records matching that query.' Do not guess or invent data.\n"
        "5. Do not say 'Based on my knowledge' or 'Typically…' — only 'The database shows…' or 'The search returned…'.\n"
        "6. When asked for 'Semantic Analysis', 'trends', or 'shifts in emphasis', DO NOT just list books or counts. You MUST analyze and synthesize actual theological themes and messages using search_sermons_tool.\n\n"

        "## Which tool to call\n"
        "- Counts, lists, statistics, verses, speakers, years → sql_query_tool (ALWAYS call this first)\n"
        "- Content, doctrine, theology, what was preached → sql_query_tool THEN search_sermons_tool\n"
        "- Semantic Analysis, trends, or shifts in emphasis → sql_query_tool (for Themes by year) THEN search_sermons_tool (for semantic synthesis)\n"
        "- Chart/plot/graph requests → viz_tool only\n"
        "- Translation Audit / compare Bible versions of a verse → get_bible_versions_tool\n"
        "- Find Bible passages about a topic/theme → search_bible_tool\n\n"

        "## sql_query_tool — ready-to-use queries\n"
        "Schema: sermons(sermon_id, date, year, language, speaker, topic, theme, summary, key_verse) "
        "| verses(id, sermon_id, verse_ref, book, chapter, verse_start, verse_end, is_key_verse)\n\n"
        "Top 5 verses per year:\n"
        "  SELECT year, verse_ref, cnt FROM ("
        "SELECT s.year, v.verse_ref, COUNT(*) AS cnt, "
        "ROW_NUMBER() OVER (PARTITION BY s.year ORDER BY COUNT(*) DESC) AS rn "
        "FROM verses v JOIN sermons s ON v.sermon_id=s.sermon_id "
        "GROUP BY s.year, v.verse_ref) WHERE rn<=5 ORDER BY year DESC, rn\n\n"
        "Most preached book:\n"
        "  SELECT ba.canonical, COUNT(*) n FROM verses v "
        "LEFT JOIN book_aliases ba ON LOWER(TRIM(v.book))=ba.alias "
        "WHERE v.book IS NOT NULL GROUP BY ba.canonical ORDER BY n DESC LIMIT 10\n\n"
        "Never preached:\n"
        "  SELECT bb.book_name, bb.testament FROM bible_books bb "
        "WHERE bb.book_name NOT IN (SELECT DISTINCT COALESCE(ba.canonical,v.book) FROM verses v "
        "LEFT JOIN book_aliases ba ON LOWER(TRIM(v.book))=ba.alias "
        "WHERE v.book IS NOT NULL AND v.book!='') ORDER BY bb.book_order\n\n"
        "Favourite verse by speaker:\n"
        "  SELECT v.verse_ref, COUNT(*) n FROM verses v "
        "JOIN sermons s USING(sermon_id) WHERE s.speaker LIKE '%Name%' "
        "GROUP BY v.verse_ref ORDER BY n DESC LIMIT 5\n\n"

        "## Speaker Naming Guidelines\n"
        "- Speakers are stored with canonical titles: 'Ps [Name]', 'SP [Name]', 'Elder [Name]'.\n"
        "- Examples: 'Ps Jeffrey Aw', 'SP Chua Seng Lee', 'Elder Lok Vi Ming'.\n"
        "- The 'speaker' column is case-insensitive (COLLATE NOCASE), but try to use the correct casing for better quality.\n"
        "- If a speaker query returns no results, check the suggestions provided by the tool.\n\n"

        "Themes by year:\n"
        "  SELECT year, COUNT(*) sermon_count, GROUP_CONCAT(DISTINCT theme) themes "
        "FROM sermons WHERE year>=2015 AND year IS NOT NULL GROUP BY year ORDER BY year\n\n"
        "Doctrine search (try synonyms):\n"
        "  SELECT topic, speaker, date, summary FROM sermons "
        "WHERE lower(topic) LIKE '%keyword%' OR lower(summary) LIKE '%keyword%' LIMIT 8\n\n"

        "## search_sermons_tool\n"
        "Semantic search over sermon text and summaries. "
        "Use short concept phrases (3-5 words). k=8 for broad topics. Try 2-3 query variants.\n\n"

        "## get_bible_versions_tool\n"
        "Returns all Bible translations (NIV, ESV, ASV) of a specific verse from the bible archive. "
        "Pass the canonical reference exactly: 'Book Chapter:Verse' (e.g. '1 John 1:9'). "
        "Use for Translation Audit requests or any question comparing Bible versions.\n\n"

        "## search_bible_tool\n"
        "Semantic search over the full Bible archive. "
        "Use for 'find passages about forgiveness', 'verses on hope', etc. "
        "Use short concept phrases (3-6 words).\n\n"

        "## viz_tool\n"
        "Only for explicit chart requests. Return the file path verbatim in your answer.\n"
        "Valid chart types: sermons_per_speaker · sermons_per_year · verses_per_book · sermons_scatter\n"
    )

    _agent_cache: dict = {}

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
    _init_ok = True

except Exception as e:
    print(f"⚠️ Initialization warning: {e}")
    _init_ok = False
    registry = None
    vector_store = None
    get_agent = None

_stats_bar_html = (
    render_stats_bar(fetch_archive_stats(registry.db_path))
    if registry is not None
    else render_stats_bar(None)
)

_ollama_status = "online" if (vector_store and vector_store._embeddings is not None) else "offline"


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
        label = "deepseek-v4-flash · cloud"
    else:  # ollama_local
        status = _ollama_status
        label = "gpt-oss-20b · local"
    return (
        "<div style='display:flex;justify-content:space-between;align-items:center;margin-top:8px;'>"
        f"<span style='color:#555;font-family:\"Source Code Pro\",monospace;font-size:0.72rem;'>inference</span>"
        f"<span class='status-badge status-{status}'>{label}</span>"
        "</div>"
    )


_TOOL_LABELS = {
    "sql_query_tool": "SQL",
    "search_sermons_tool": "Vector Search",
    "viz_tool": "Visualization",
    "get_bible_versions_tool": "Bible Versions",
    "search_bible_tool": "Bible Search",
}


def _build_meta_footer(tools_used: list, token_info: dict) -> str:
    left = ""
    if tools_used:
        labels = [_TOOL_LABELS.get(t, t) for t in tools_used]
        left = "🔧 " + " · ".join(labels)

    right = ""
    total = token_info.get("total") or (token_info.get("input", 0) + token_info.get("output", 0))
    if total:
        right = f"📊 {total:,} tokens"

    if not left and not right:
        return ""

    return (
        "<div style='margin-top:10px;padding-top:8px;"
        "border-top:1px solid rgba(255,255,255,0.06);"
        "display:flex;justify-content:space-between;align-items:center;"
        "font-size:0.68rem;font-family:monospace;color:#475569;letter-spacing:0.4px;'>"
        f"<span>{left}</span>"
        f"<span>{right}</span>"
        "</div>"
    )


def respond(message, history, provider="ollama_local"):
    if not _init_ok or get_agent is None:
        return "⚠️ Agent not initialized. Check that Ollama is running.", [], {}

    try:
        agent = get_agent(provider)
    except Exception as e:
        return f"⚠️ Could not load {provider} agent: {e}", [], {}

    truncated_history = history[-2:] if len(history) > 2 else history
    messages = []
    for turn in truncated_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            content = turn["content"]
            if isinstance(content, list):
                text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
                content = " ".join(text_parts)
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=message))

    try:
        result = agent.invoke({"messages": messages})
        final_msg = result["messages"][-1]
        final = final_msg.content
        if not isinstance(final, str):
            final = str(final)

        # If the LLM dropped the chart path from its response, recover it from ToolMessage
        if "/tmp/bbtc_chart_" not in final:
            import re
            for msg in result["messages"]:
                match = re.search(r'/tmp/bbtc_chart_[a-f0-9]+\.(png|json)', str(msg.content))
                if match:
                    final = final.rstrip() + "\n" + match.group(0)
                    break

        # Extract tool names from ToolMessages
        tools_used = []
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage) and getattr(msg, "name", None) and msg.name not in tools_used:
                tools_used.append(msg.name)

        # Extract token counts (LangChain usage_metadata takes priority)
        token_info: dict = {}
        usage = getattr(final_msg, "usage_metadata", None)
        if usage:
            token_info = {
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
                "total": usage.get("total_tokens", 0),
            }
        else:
            meta = getattr(final_msg, "response_metadata", {}) or {}
            raw = meta.get("token_usage") or meta.get("usage") or {}
            if raw:
                inp = raw.get("prompt_tokens") or raw.get("input_tokens") or raw.get("prompt_eval_count", 0)
                out = raw.get("completion_tokens") or raw.get("output_tokens") or raw.get("eval_count", 0)
                token_info = {"input": inp, "output": out, "total": raw.get("total_tokens") or (inp + out)}

        return final, tools_used, token_info
    except Exception as e:
        return f"⚠️ An error occurred while processing your request: {e}", [], {}


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

with gr.Blocks(title="BBTC Sermon Intelligence") as demo:
    with gr.Row(elem_id="header"):
        with gr.Column(scale=4):
            gr.HTML("""
                <div id='title-container'>
                    <img src='https://www.bbtc.com.sg/wp-content/uploads/2021/04/BBTC-Logo-Header.png' alt='BBTC Logo'>
                    <div id='title-text'>
                        <h1>Sermon Intelligence</h1>
                        <p>Bethesda Bedok-Tampines Church &nbsp;·&nbsp; Agentic RAG</p>
                    </div>
                </div>
            """)

    gr.HTML(_stats_bar_html)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                min_height=550,
                show_label=False,
                elem_classes="chatbot-container",
                avatar_images=(None, "https://www.bbtc.com.sg/wp-content/uploads/2021/04/BBTC-Logo-Header.png")
            )
            with gr.Row(elem_classes="input-container"):
                msg = gr.Textbox(
                    placeholder="Describe the data you need or ask a theological question...",
                    container=False,
                    scale=7,
                )
                submit = gr.Button("Send", variant="primary", scale=1, elem_classes="btn-primary")

            gr.Examples(
                examples=[
                    ["Scripture Coverage: Generate a frequency heatmap of the most frequently preached Bible books."],
                    ["Identify the consistent theological themes in BBTC’s vision statements and pulpit series between 2015 and 2026"],
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

        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("### System Health")

            vec_status = "online" if vector_store else "offline"
            gr.HTML(f"""
                <div style='display: flex; flex-direction: column; gap: 12px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='color: #94a3b8;'>Vector Store</span>
                        <span class='status-badge status-{vec_status}'>{vec_status}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='color: #94a3b8;'>SQL Registry</span>
                        <span class='status-badge status-online'>active</span>
                    </div>
                </div>
            """)
            inference_status = gr.HTML(_inference_badge_html("ollama_local"))

            gr.Markdown("---")
            gr.Markdown("### Inference Engine")
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

            gr.Markdown("---")
            gr.Markdown("### Capabilities")
            gr.Markdown(
                "- **Semantic Search**: Retrieval of sermon content across a decade of archives.\n"
                "- **SQL Analytics**: High-precision metadata querying for statistics and counts.\n"
                "- **Dynamic Viz**: Real-time generation of interactive Plotly visualizations.\n"
                "- **Bible Context**: Multi-version Bible referencing and cross-comparison."
            )

            gr.Markdown("---")
            clear = gr.Button("Clear Chat", variant="secondary")

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

    provider_radio.change(
        _on_provider_change,
        inputs=provider_radio,
        outputs=[provider_state, inference_status],
    )

    def user_msg(user_message, history: list):
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]

    def bot_msg(history: list, provider: str):
        if not history or history[-1]["role"] != "user":
            return history

        user_message = history[-1]["content"]
        chat_history = history[:-1]
        bot_message, tools_used, token_info = respond(user_message, chat_history, provider)

        text, chart_path = extract_chart_path(bot_message)
        meta_footer = _build_meta_footer(tools_used, token_info)

        content = []
        if text:
            content.append({"type": "text", "text": text + meta_footer})

        if chart_path:
            if chart_path.endswith('.json'):
                try:
                    import plotly.io as pio
                    fig = pio.read_json(chart_path)
                    content.append(gr.Plot(fig))
                except Exception as e:
                    content.append({"type": "text", "text": f"\n⚠️ Error loading interactive chart: {e}"})
            else:
                content.append({"type": "image", "image": {"path": chart_path}})

        if not content:
            content = bot_message

        history.append({"role": "assistant", "content": content})
        return history

    disable_submit = lambda: gr.update(value="Processing…", interactive=False)
    enable_submit = lambda: gr.update(value="Send", interactive=True)

    msg.submit(user_msg, [msg, chatbot], [msg, chatbot], queue=True).then(
        disable_submit, None, submit
    ).then(
        bot_msg, [chatbot, provider_state], chatbot
    ).then(
        enable_submit, None, submit
    )
    submit.click(user_msg, [msg, chatbot], [msg, chatbot], queue=True).then(
        disable_submit, None, submit
    ).then(
        bot_msg, [chatbot, provider_state], chatbot
    ).then(
        enable_submit, None, submit
    )
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 0)) or None
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=custom_css,
        allowed_paths=["/tmp"]
    )
