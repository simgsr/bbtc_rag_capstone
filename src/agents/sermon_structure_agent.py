# src/agents/sermon_structure_agent.py
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from src.storage.sqlite_store import SermonRegistry
from src.storage.chroma_store import SermonVectorStore
from src.tools.sql_tool import make_sql_tool
from src.tools.matplotlib_tool import make_matplotlib_tool
from src.graph.state import AgentState

load_dotenv()

_registry = SermonRegistry()
_store = SermonVectorStore()
# Try to get LLM from env or fallback
if os.getenv("GROQ_API_KEY"):
    _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
else:
    from src.llm import get_llm
    _llm = get_llm(temperature=0.1)

_sql_tool = make_sql_tool(_registry)
_matplotlib_tool = make_matplotlib_tool(_registry)

_SYSTEM = (
    "You are the Sermon Structure Agent for BBTC. "
    "You answer quantitative questions about sermon history using SQL. "
    "Use sql_query_tool to query the sermons table. "
    "Schema: sermons(sermon_id, filename, url, speaker, date, series, bible_book, "
    "primary_verse, language, file_type, year, status). "
    "When the user asks for a chart, plot, or visualisation, use matplotlib_tool. "
    "Choose the best chart_name from: sermons_per_speaker, sermons_per_year, top_bible_books. "
    "Return the file path exactly as returned by the tool. "
    "Always show SQL used. Format answers clearly."
)

_agent = create_react_agent(_llm, tools=[_sql_tool, _matplotlib_tool], prompt=_SYSTEM)

_CHART_PATH_RE = re.compile(r"/tmp/bbtc_chart_\S+\.png")


def _extract_text(result) -> str:
    msgs = result.get("messages", [])
    if msgs:
        last_msg = msgs[-1]
        return getattr(last_msg, "content", str(last_msg))
    return str(result)


def sermon_structure_agent_node(state: AgentState) -> dict:
    result = _agent.invoke({"messages": [HumanMessage(content=state["query"])]})
    response_text = _extract_text(result)
    match = _CHART_PATH_RE.search(response_text)
    return {
        "response": response_text,
        "chart_path": match.group(0) if match else None,
        "debug_log": state.get("debug_log", "") + "\n📊 Sermon Structure Agent answered.",
    }
