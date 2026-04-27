import os
import sys
from dotenv import load_dotenv

# Ensure root is in path
sys.path.append(os.getcwd())
load_dotenv()

# Map GEMINI_API_KEY to GOOGLE_API_KEY if needed
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

try:
    from src.storage.sqlite_store import SermonRegistry
    from src.storage.chroma_store import SermonVectorStore
    from src.llm import get_llm
    from src.tools.sql_tool import make_sql_tool
    from src.tools.vector_tool import make_vector_tool
    from src.tools.bible_tool import make_bible_tool
    from src.tools.matplotlib_tool import make_matplotlib_tool
    from langgraph.prebuilt import create_react_agent
    
    print("✅ Imports successful")
    
    registry = SermonRegistry()
    vector_store = SermonVectorStore()
    llm = get_llm(provider_type="groq", temperature=0.1)
    
    sql_tool = make_sql_tool(registry)
    vector_tool = make_vector_tool(vector_store)
    bible_tool = make_bible_tool(vector_store)
    viz_tool = make_matplotlib_tool(registry)
    
    print("✅ Tool initialization successful")
    
    agent = create_react_agent(llm, tools=[sql_tool, vector_tool, bible_tool, viz_tool], prompt="You are a test bot.")
    print("✅ Agent initialization successful")

except Exception as e:
    print(f"❌ Error during initialization: {e}")
    import traceback
    traceback.print_exc()
