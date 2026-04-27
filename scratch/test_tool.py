
from src.storage.chroma_store import SermonVectorStore
from src.tools.bible_tool import make_bible_tool

store = SermonVectorStore()
compare_tool = make_bible_tool(store)

result = compare_tool.invoke({"reference": "John 1:1"})
print(result)
