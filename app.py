import gradio as gr
import os
from dotenv import load_dotenv
from src.storage.chroma_store import SermonVectorStore
from src.llm import get_llm
from src.ui_helpers import extract_chart_path, fetch_archive_stats, render_stats_bar

load_dotenv()

# Map GEMINI_API_KEY to GOOGLE_API_KEY if needed
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

from src.storage.sqlite_store import SermonRegistry
from src.tools.sql_tool import make_sql_tool
from src.tools.vector_tool import make_vector_tool
from src.tools.bible_tool import make_bible_tool
from src.tools.matplotlib_tool import make_matplotlib_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

# Initialize
try:
    registry = SermonRegistry()
    vector_store = SermonVectorStore()
    llm = get_llm(provider_type="groq", temperature=0.1)
    
    # Tools
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
        "Valid chart_name values: 'sermons_per_speaker', 'sermons_per_year', 'top_bible_books'.\n\n"
        "## Grounding rules\n"
        "- Answer ONLY from data returned by the tools. Never invent sermon content, speaker names, "
        "dates, or verses.\n"
        "- When answering from search_sermons_tool results, cite the sermon filename and speaker name for every excerpt quoted.\n"
        "- If the tools return no relevant data, say so explicitly — do not guess or fill gaps.\n"
        "- If you need more information to answer precisely, call the relevant tool again with "
        "a refined query before responding.\n"
    )
    
    agent = create_react_agent(llm, tools=[sql_tool, vector_tool, bible_tool, viz_tool], prompt=SYSTEM_PROMPT)

except Exception as e:
    print(f"⚠️ Initialization warning: {e}")
    agent = None
    registry = None

_stats_bar_html = (
    render_stats_bar(fetch_archive_stats(registry.db_path))
    if registry is not None
    else render_stats_bar(None)
)

def respond(message, history, provider):
    # Dynamic LLM selection
    try:
        if provider == "Gemini (Cloud)":
            current_llm = get_llm(provider_type="gemini", temperature=0.1, gemini_model="gemini-1.5-flash")
        elif provider == "Groq (Cloud)":
            current_llm = get_llm(provider_type="groq", temperature=0.1)
        else:
            current_llm = get_llm(provider_type="ollama", temperature=0.1)
        
        # Re-initialize agent with the chosen LLM
        current_agent = create_react_agent(current_llm, tools=[sql_tool, vector_tool, bible_tool, viz_tool], prompt=SYSTEM_PROMPT)
    except Exception as e:
        return f"⚠️ Initialization error: {e}"

    # Convert Gradio history to LangChain messages
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
    
    print(f"DEBUG: Processing request [{provider}]: {message}")
    try:
        result = current_agent.invoke({"messages": messages})
        print(f"DEBUG: Request completed successfully.")
        return result["messages"][-1].content
    except Exception as e:
        print(f"DEBUG: Request failed with error: {e}")
        return f"⚠️ An error occurred while processing your request: {e}"

# Custom CSS for a professional, premium look
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

footer {visibility: hidden}
.gradio-container {
    background-color: #0f172a !important;
    color: #f8fafc;
    font-family: 'Inter', sans-serif !important;
    max-width: 1200px !important;
}

/* Sidebar Styling */
.sidebar {
    background: rgba(30, 41, 59, 0.7) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 20px !important;
    border-radius: 12px;
}

/* Chatbot Area */
.chatbot-container {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background: rgba(30, 41, 59, 0.4) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

/* Message Bubbles */
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

/* Input Area */
.input-container {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 5px !important;
}

/* Buttons */
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
        # Main Chat Area
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                type="messages",
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
                        ["List the top 3 verses preached each year."],
                        ["Which bible verses were preached most often in 2024?"],
                        ["Summarize the 'Bigger Fire' sermon and its key takeaways."],
                        ["Create a bar chart of how many sermons each speaker gave."],
                        ["Who spoke on the most recent Sunday in the database?"]
                    ],
                    inputs=msg,
                    label="⚡ Quick Inquiries"
                )

        # Sidebar Settings
        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("### ⚙️ Engine Settings")
            
            providers = []
            if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"): providers.append("Gemini (Cloud)")
            if os.getenv("GROQ_API_KEY"): providers.append("Groq (Cloud)")
            providers.append("Ollama (Local)")
            
            provider_radio = gr.Radio(
                choices=providers,
                value=providers[0],
                label="AI Brain",
                info="Switch providers if you hit rate limits."
            )
            
            gr.Markdown("---")
            gr.Markdown("### 📡 System Status")
            
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
                </div>
            """)
            
            gr.Markdown("---")
            gr.Markdown("### 📖 About")
            gr.Markdown(
                "This assistant uses a hybrid Agentic RAG pipeline. "
                "It can query SQL for statistics and search sermon text for semantic context."
            )
            
            clear = gr.Button("🗑️ Reset Conversation", variant="secondary")

    # Message Handling Logic
    def user_msg(user_message, history: list):
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]

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

        if not isinstance(bot_message, str):
            bot_message = str(bot_message)

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

if __name__ == "__main__":
    demo.launch(css=custom_css, theme=gr.themes.Default(), allowed_paths=["/tmp"])
