import gradio as gr
import os
import subprocess
import time
import urllib.request
from dotenv import load_dotenv
from src.storage.chroma_store import SermonVectorStore
from src.llm import get_llm
from src.ui_helpers import extract_chart_path, fetch_archive_stats, render_stats_bar
from src.storage.sqlite_store import SermonRegistry
from src.tools.sql_tool import make_sql_tool
from src.tools.vector_tool import make_vector_tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.tools.viz_tool import make_viz_tool
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
    llm = get_llm(temperature=0.1)

    sql_tool = make_sql_tool(registry.db_path)
    vector_tool = make_vector_tool(vector_store)
    viz_tool = make_viz_tool(registry)

    from datetime import date as _date
    _today = _date.today().isoformat()
    _cur_year = _date.today().year

    SYSTEM_PROMPT = (
        f"You are the BBTC Sermon Intelligence Assistant for Bethesda Bedok-Tampines Church.\n"
        f"Today is {_today}. Archive covers 2015–{_cur_year}. "
        f"'Last N years' → min_year = {_cur_year} - N + 1.\n\n"
        "## Search approach\n"
        "For every factual question (content, doctrine, theology, what was preached): "
        "call sql_query_tool AND search_sermons_tool. Do not stop at one tool. "
        "Report whatever either tool returns — partial matches are useful context.\n"
        "For chart requests: use viz_tool only.\n\n"
        "## Tools\n"
        "sql_query_tool — Schema:\n"
        "  sermons(sermon_id, date, year, speaker, topic, theme, summary, key_verse)\n"
        "  verses(sermon_id, verse_ref, book, chapter, verse_start, verse_end, is_key_verse)\n"
        "  Most preached book: SELECT ba.canonical, COUNT(*) n FROM verses v "
        "LEFT JOIN book_aliases ba ON LOWER(TRIM(v.book))=ba.alias "
        "WHERE v.book IS NOT NULL GROUP BY ba.canonical ORDER BY n DESC LIMIT 10\n"
        "  Never preached: SELECT bb.book_name, bb.testament FROM bible_books bb "
        "WHERE bb.book_name NOT IN (SELECT DISTINCT COALESCE(ba.canonical,v.book) FROM verses v "
        "LEFT JOIN book_aliases ba ON LOWER(TRIM(v.book))=ba.alias "
        "WHERE v.book IS NOT NULL AND v.book!='') ORDER BY bb.book_order\n"
        "  Favourite verse by speaker: SELECT v.verse_ref, COUNT(*) n FROM verses v "
        "JOIN sermons s USING(sermon_id) WHERE s.speaker LIKE '%Name%' "
        "GROUP BY v.verse_ref ORDER BY n DESC LIMIT 5\n"
        "  Emphasis / themes by year: SELECT year, COUNT(*) sermon_count, "
        "GROUP_CONCAT(DISTINCT theme) themes FROM sermons "
        "WHERE year>=2015 AND year IS NOT NULL GROUP BY year ORDER BY year\n"
        "  Doctrine / BBTC position on X: SELECT topic, speaker, date, summary FROM sermons "
        "WHERE lower(topic) LIKE '%keyword%' OR lower(summary) LIKE '%keyword%' LIMIT 8 "
        "(try synonyms: 'once saved' → 'assurance','eternal security'; "
        "'end times' → 'rapture','tribulation','eschatol')\n\n"
        "search_sermons_tool — semantic search over sermon text and summaries. "
        "Use short concept phrases. k=8 for broad topics. Try 2-3 query variants.\n\n"
        "viz_tool — only for chart requests. Return file path verbatim. "
        "Valid: sermons_per_speaker · sermons_per_year · verses_per_book · sermons_scatter\n\n"
        "## Rules\n"
        "- Never answer from memory. All facts must come from tool results.\n"
        "- If tools return nothing, say so. Do not speculate.\n"
    )

    agent = create_react_agent(
        llm,
        tools=[sql_tool, vector_tool, viz_tool],
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )

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

    truncated_history = history[-2:] if len(history) > 2 else history
    messages = []
    for turn in truncated_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            content = turn["content"]
            if isinstance(content, list):
                # Handle complex content (text + plot)
                text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
                content = " ".join(text_parts)
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=message))

    try:
        result = agent.invoke({"messages": messages})
        final = result["messages"][-1].content
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

        return final
    except Exception as e:
        return f"⚠️ An error occurred while processing your request: {e}"


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
                    ["Trend analysis: Plot the number of sermons preached per year since 2010."],
                    ["Speaker breakdown: Create a stacked bar chart of sermon counts by speaker per year."],
                    ["Scripture coverage: Show a heatmap of Bible books most frequently preached."],
        
                     # Qualitative / Doctrine Intent
                    ["Doctrine: What is BBTC's theological position on 'Once Saved, Always Saved'?"],
                    ["Eschatology: Explain the believed sequence of End Times events."],
                    ["Evolution of Themes: Compare the primary ministry focus in 2015 versus today."],
        
                    # Specific Content Search
                    ["Find the top 3 sermons related to 'Spiritual Warfare' from the last two years."],
                    ["Which Minor Prophets have not been the focus of a sermon in the last 5 years?"],
                    ["Summarize the 5 most frequent emphasized Bible verses in BBTC sermons."],
                    ["Search for all mentions of 'Mental Health' and categorize the biblical advice given."]
                ],
                inputs=msg,
                label="Example questions"
            )

        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("### System Health")

            vec_status = "online" if vector_store else "offline"
            ollama_status = "online" if (vector_store and vector_store._embeddings is not None) else "offline"
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
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='color: #94a3b8;'>Inference</span>
                        <span class='status-badge status-{ollama_status}'>ollama</span>
                    </div>
                </div>
            """)

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

    def user_msg(user_message, history: list):
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]

    def bot_msg(history: list):
        if not history or history[-1]["role"] != "user":
            return history

        user_message = history[-1]["content"]
        chat_history = history[:-1]
        bot_message = respond(user_message, chat_history)

        text, chart_path = extract_chart_path(bot_message)
        
        content = []
        if text:
            content.append({"type": "text", "text": text})
            
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
        allowed_paths=["/tmp"]
    )
