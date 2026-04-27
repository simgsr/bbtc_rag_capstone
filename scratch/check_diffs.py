
from src.storage.chroma_store import SermonVectorStore
from src.tools.bible_tool import make_bible_tool

store = SermonVectorStore()
compare_tool = make_bible_tool(store)

for ref in ["Matthew 17:21", "Mark 16:9", "John 7:53", "Isaiah 14:12"]:
    print(f"\n--- Checking {ref} ---")
    result = compare_tool.invoke({"reference": ref})
    print(result)
