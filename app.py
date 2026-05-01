import gradio as gr
import os
import subprocess
import time
import urllib.request
from dotenv import load_dotenv
from src.storage.chroma_store import SermonVectorStore
from src.llm import get_llm, GROQ_MODEL
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

# Ensure Plotly uses dark template for consistency
pio.templates.default = "plotly_dark"

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
    else:
        status = _ollama_status
        label = "ollama · local"
    return (
        "<div style='display:flex;justify-content:space-between;align-items:center;margin-top:8px;'>"
        f"<span style='color:#94a3b8;'>Inference</span>"
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


def respond(message, history, provider="ollama"):
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
@import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500;600&family=Inter:wght@300;400;500;600&family=Outfit:wght@700;800&display=swap');

footer { visibility: hidden }

* { box-sizing: border-box; }

body {
    background: #060e1e !important;
    background-image:
        radial-gradient(ellipse at 12% 35%, rgba(37, 99, 235, 0.09) 0%, transparent 52%),
        radial-gradient(ellipse at 88% 12%, rgba(124, 58, 237, 0.07) 0%, transparent 52%),
        radial-gradient(ellipse at 50% 90%, rgba(16, 185, 129, 0.04) 0%, transparent 50%) !important;
}

.gradio-container {
    background: transparent !important;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif !important;
    max-width: 1440px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.3); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.55); }

/* Header */
#title-container {
    padding: 28px 0 22px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
#title-container img {
    height: 44px;
    opacity: 0.92;
    filter: drop-shadow(0 0 14px rgba(99, 102, 241, 0.35));
}
#title-text h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 3px 0;
    background: linear-gradient(108deg, #93c5fd 0%, #a78bfa 52%, #f0abfc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.6px;
    line-height: 1.12;
}
#title-text p {
    color: #475569;
    font-size: 0.7rem;
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
    background: rgba(10, 16, 34, 0.92);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 10px;
    padding: 10px 22px;
    margin-bottom: 16px;
    color: #64748b;
    font-size: 0.8rem;
    letter-spacing: 0.3px;
    backdrop-filter: blur(24px);
    font-family: 'Source Code Pro', monospace;
}

/* Chat area */
.chatbot-container {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.055) !important;
    background: rgba(6, 14, 30, 0.94) !important;
    box-shadow:
        0 0 0 1px rgba(99, 102, 241, 0.05),
        0 28px 64px -12px rgba(0, 0, 0, 0.75),
        inset 0 1px 0 rgba(255, 255, 255, 0.035);
    overflow: hidden !important;
}

/* Messages */
.message-user {
    background: linear-gradient(135deg, #1a3578 0%, #1d4ed8 100%) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 18px !important;
    color: #bfdbfe !important;
    font-size: 0.88rem;
    line-height: 1.6;
    box-shadow: 0 2px 14px -3px rgba(37, 99, 235, 0.4);
}
.message-assistant {
    background: rgba(22, 32, 54, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 12px 18px !important;
    font-size: 0.88rem;
    line-height: 1.65;
}

/* Plotly chart embed — give it breathing room */
.gradio-plot {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(99, 102, 241, 0.1) !important;
    margin-top: 8px !important;
    background: rgba(10, 16, 34, 0.6) !important;
}

/* Input row */
.input-container {
    background: rgba(10, 16, 34, 0.88) !important;
    border: 1px solid rgba(99, 102, 241, 0.18) !important;
    border-radius: 14px !important;
    margin-top: 12px !important;
    padding: 5px 5px 5px 8px !important;
    transition: border-color 0.22s ease, box-shadow 0.22s ease;
    backdrop-filter: blur(12px);
}
.input-container:focus-within {
    border-color: rgba(99, 102, 241, 0.48) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.09);
}
.input-container textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}

/* Submit button */
.btn-primary {
    background: linear-gradient(138deg, #1d4ed8 0%, #5b21b6 100%) !important;
    border: none !important;
    color: #e0e7ff !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    border-radius: 10px !important;
    transition: all 0.22s ease;
    box-shadow: 0 2px 12px -2px rgba(37, 99, 235, 0.5);
    font-family: 'Source Code Pro', monospace !important;
}
.btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 22px -4px rgba(37, 99, 235, 0.65);
    filter: brightness(1.08);
}
.btn-primary:active {
    transform: translateY(0);
    filter: brightness(0.93);
}

/* Secondary button (Clear) */
button.secondary {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #94a3b8 !important;
    border-radius: 10px !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.5px;
    transition: all 0.18s ease;
}
button.secondary:hover {
    background: rgba(30, 41, 59, 1) !important;
    border-color: rgba(255,255,255,0.14) !important;
    color: #cbd5e1 !important;
}

/* Example pills */
.examples-holder .example-btn,
.gr-examples .example {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.75rem !important;
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    transition: all 0.18s ease !important;
}
.examples-holder .example-btn:hover,
.gr-examples .example:hover {
    background: rgba(30, 41, 80, 0.9) !important;
    border-color: rgba(99, 102, 241, 0.35) !important;
    color: #c7d2fe !important;
}

/* Sidebar */
.sidebar {
    background: rgba(8, 14, 28, 0.8) !important;
    backdrop-filter: blur(24px);
    border: 1px solid rgba(255, 255, 255, 0.055) !important;
    padding: 22px !important;
    border-radius: 16px;
}
.sidebar h3 {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    color: #475569 !important;
    margin-bottom: 14px !important;
}
.sidebar p, .sidebar li {
    font-size: 0.82rem !important;
    color: #64748b !important;
    line-height: 1.6 !important;
}

/* Status badges */
.status-badge {
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.58rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-family: 'Source Code Pro', monospace;
}
.status-online {
    background: rgba(34, 197, 94, 0.1);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.2);
}
.status-offline {
    background: rgba(239, 68, 68, 0.1);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.2);
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
            inference_status = gr.HTML(_inference_badge_html("ollama"))

            gr.Markdown("---")
            gr.Markdown("### Inference Engine")
            provider_radio = gr.Radio(
                choices=["Ollama (local)", "Groq (cloud)"],
                value="Ollama (local)",
                show_label=False,
                interactive=True,
            )
            provider_state = gr.State("ollama")

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
        provider = "groq" if "Groq" in radio_val else "ollama"
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
