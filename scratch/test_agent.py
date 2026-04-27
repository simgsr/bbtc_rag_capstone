import os
from dotenv import load_dotenv
from src.storage.sqlite_store import SermonRegistry
from src.storage.chroma_store import SermonVectorStore
from src.llm import get_llm
from src.tools.sql_tool import make_sql_tool
from src.tools.vector_tool import make_vector_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

load_dotenv()

def test_agent():
    registry = SermonRegistry()
    vector_store = SermonVectorStore()
    llm = get_llm(temperature=0)
    
    sql_tool = make_sql_tool(registry)
    vector_tool = make_vector_tool(vector_store)
    
    SYSTEM_PROMPT = (
        "You are the BBTC Sermon Intelligence Assistant. "
        "You have access to tools to answer questions about sermons.\n\n"
        "1. For quantitative/statistical questions, use the sql_query_tool.\n"
        "   CRITICAL: The column for the bible verse is 'primary_verse' (NOT 'verse').\n"
        "   Schema: sermons(sermon_id, filename, speaker, bible_book, primary_verse, year).\n"
        "   TIP: To get top items per year in one go, use window functions. Example:\n"
        "   SELECT year, primary_verse, count FROM (\n"
        "     SELECT year, primary_verse, COUNT(*) as count, RANK() OVER (PARTITION BY year ORDER BY COUNT(*) DESC) as rank\n"
        "     FROM sermons WHERE primary_verse IS NOT NULL AND primary_verse != 'null'\n"
        "     GROUP BY year, primary_verse\n"
        "   ) WHERE rank <= 3;\n\n"
        "2. For content/semantic questions, use the search_sermons_tool.\n"
        "3. For visualizations, use the matplotlib_tool.\n\n"
        "If a tool returns an error, read the error message and correct your query."
    )
    agent = create_react_agent(llm, tools=[sql_tool, vector_tool], prompt=SYSTEM_PROMPT)
    
    query = "List the top 3 verses preached each year?"
    print(f"Testing query: {query}")
    
    for chunk in agent.stream({"messages": [HumanMessage(content=query)]}):
        print(f"--- Chunk ---\n{chunk}")

if __name__ == "__main__":
    test_agent()
