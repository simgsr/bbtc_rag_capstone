
import os
from src.ingestion.bible.epub_parser import BibleEpubParser
from src.storage.chroma_store import SermonVectorStore

def main():
    bible_dir = "data/bibles"
    filename = "Bible - American Standard Version.epub"
    version = "ASV"
    
    filepath = os.path.join(bible_dir, filename)
    print(f"--- Processing {version} ---")
    parser = BibleEpubParser(filepath, version)
    verses = parser.parse()
    print(f"Extracted {len(verses)} verses from {version}")

    store = SermonVectorStore()
    BATCH_SIZE = 100
    for i in range(0, len(verses), BATCH_SIZE):
        batch = verses[i:i+BATCH_SIZE]
        chunks = []
        metadatas = []
        ids = []
        for v in batch:
            chunks.append(v['text'])
            ids.append(f"{v['version']}_{v['ref_id']}")
            metadatas.append({
                "book": v['book'], "chapter": v['chapter'], "verse": v['verse'],
                "version": v['version'], "reference": v['reference'],
                "ref_id": v['ref_id'], "verse_meaning": "Meaning generation disabled."
            })
        store.upsert_bible_chunks(chunks, metadatas, ids)
        if i % 1000 == 0:
            print(f"Uploaded batch {i//BATCH_SIZE} for {version}")

if __name__ == "__main__":
    main()
