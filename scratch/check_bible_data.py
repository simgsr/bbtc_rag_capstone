
from src.storage.chroma_store import SermonVectorStore

store = SermonVectorStore()
counts = store.counts()
print(f"Counts: {counts}")

# Check John 1:1
ref_id = "JOH_001_001"
versions = store.get_bible_versions(ref_id)
print(f"John 1:1 versions found: {len(versions)}")
for v in versions:
    print(f"Version: {v['metadata'].get('version')}, Content: {v['content']}")
