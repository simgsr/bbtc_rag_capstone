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

    SYSTEM_PROMPT = (
        "You are the BBTC Sermon Intelligence Assistant for Bethesda Bedok-Tampines Church.\n\n"
        "## Tool routing\n"
        "- Use 'sql_query_tool' for: counts, lists of speakers/years, verse statistics, "
        "questions that need numbers. The database has:\n"
        "    • sermons(sermon_id, date, year, language, speaker, topic, theme, summary, key_verse, status)\n"
        "    • verses(id, sermon_id, verse_ref, book, chapter, verse_start, verse_end, is_key_verse)\n"
        "  For 'most preached book': SELECT book, COUNT(*) as n FROM verses GROUP BY book ORDER BY n DESC LIMIT 10\n"
        "  For 'verses by speaker': SELECT v.verse_ref, COUNT(*) FROM verses v JOIN sermons s USING(sermon_id) WHERE s.speaker LIKE '%Name%' GROUP BY v.verse_ref ORDER BY COUNT(*) DESC\n"
        "- Use 'search_sermons_tool' for: questions about sermon content, what a pastor said about a topic, "
        "summaries of specific sermons. Pass year/speaker filters when specified.\n"
        "- Use 'viz_tool' only when the user asks for a chart or visualization. "
        "Valid chart_name values: 'sermons_per_speaker', 'sermons_per_year', 'verses_per_book', 'sermons_scatter'. "
        "When viz_tool returns a file path, include that exact path verbatim in your response.\n\n"
        "## Rules\n"
        "- Answer ONLY from tool results. Never invent speaker names, dates, or verses.\n"
        "- If tools return no data, say so explicitly.\n"
        "- For 'most recent sermon': use sql_query_tool with ORDER BY date DESC LIMIT 1.\n"
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

    truncated_history = history[-6:] if len(history) > 6 else history
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@700;800&display=swap');

footer { visibility: hidden }

* { box-sizing: border-box; }

body {
    background: #07101f !important;
    background-image:
        radial-gradient(ellipse at 15% 40%, rgba(37, 99, 235, 0.07) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 15%, rgba(124, 58, 237, 0.06) 0%, transparent 55%) !important;
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
::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.25); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.45); }

/* Header */
#title-container {
    padding: 28px 0 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 18px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
#title-container img {
    height: 42px;
    opacity: 0.9;
    filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.3));
}
#title-text h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0 0 2px 0;
    background: linear-gradient(110deg, #93c5fd 0%, #a78bfa 55%, #f9a8d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    line-height: 1.15;
}
#title-text p {
    color: #475569;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin: 0;
}

/* Stats bar */
.stats-bar {
    display: flex;
    gap: 0;
    background: rgba(13, 20, 40, 0.9);
    border: 1px solid rgba(99, 102, 241, 0.1);
    border-radius: 10px;
    padding: 10px 20px;
    margin-bottom: 16px;
    color: #64748b;
    font-size: 0.82rem;
    letter-spacing: 0.2px;
    backdrop-filter: blur(20px);
}

/* Chat area */
.chatbot-container {
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    background: rgba(7, 16, 31, 0.92) !important;
    box-shadow:
        0 0 0 1px rgba(99, 102, 241, 0.04),
        0 24px 60px -10px rgba(0, 0, 0, 0.7),
        inset 0 1px 0 rgba(255, 255, 255, 0.03);
    overflow: hidden !important;
}

/* Messages */
.message-user {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 11px 16px !important;
    color: #bfdbfe !important;
    font-size: 0.9rem;
    line-height: 1.55;
    box-shadow: 0 2px 12px -3px rgba(37, 99, 235, 0.35);
}
.message-assistant {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 4px 16px 16px 16px !important;
    padding: 11px 16px !important;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Input row */
.input-container {
    background: rgba(13, 20, 40, 0.85) !important;
    border: 1px solid rgba(99, 102, 241, 0.16) !important;
    border-radius: 12px !important;
    margin-top: 10px !important;
    padding: 4px 4px 4px 6px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    backdrop-filter: blur(10px);
}
.input-container:focus-within {
    border-color: rgba(99, 102, 241, 0.42) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.07);
}

/* Submit button */
.btn-primary {
    background: linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%) !important;
    border: none !important;
    color: #e0e7ff !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    border-radius: 9px !important;
    transition: all 0.2s ease;
    box-shadow: 0 2px 10px -2px rgba(37, 99, 235, 0.45);
}
.btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px -4px rgba(37, 99, 235, 0.6);
    filter: brightness(1.1);
}
.btn-primary:active {
    transform: translateY(0);
    filter: brightness(0.95);
}

/* Sidebar */
.sidebar {
    background: rgba(10, 16, 32, 0.75) !important;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 20px !important;
    border-radius: 14px;
}

/* Status badges */
.status-badge {
    padding: 3px 9px;
    border-radius: 5px;
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
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
                    ["Show an interactive chart of how many sermons were preached each year"],
                    ["Create an interactive bar chart of sermon count per speaker"],
                    ["Show a scatter plot of sermon counts by speaker and year"],
                    ["How many sermons are in the archive and who are the top 5 speakers?"],
                    ["What was the most recent sermon and what were its key points?"],
                    ["What have our pastors said about faith during trials and suffering?"],
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
