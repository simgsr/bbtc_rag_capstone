import os
import json
from dotenv import load_dotenv
from src.ingestion.bible.epub_parser import BibleEpubParser
from src.storage.chroma_store import SermonVectorStore
from src.llm import get_llm

load_dotenv()

def generate_verse_meaning(llm, verse_obj):
    """Generates a brief theological meaning for the verse."""
    prompt = f"Explain the meaning of {verse_obj['reference']} in the context of the book of {verse_obj['book']}. Keep it concise (2-3 sentences)."
    try:
        # Using a very lightweight call
        res = llm.invoke(prompt)
        return res.content.strip()
    except:
        return "Explanation pending research."

def main():
    # Setup paths
    bible_dir = "data/bibles"
    files = {
        "NIV.epub": "NIV",
        "ESV The Holy Bible.epub": "ESV",
        "Bible - American Standard Version.epub": "ASV"
    }

    store = SermonVectorStore()
    llm = get_llm(provider_type="groq", temperature=0) # Use Groq for speed

    for filename, version in files.items():
        filepath = os.path.join(bible_dir, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {filename} (not found)")
            continue

        print(f"--- Processing {version} ---")
        parser = BibleEpubParser(filepath, version)
        verses = parser.parse()
        print(f"Extracted {len(verses)} verses from {version}")

        # Batching for ChromaDB
        BATCH_SIZE = 100
        for i in range(0, len(verses), BATCH_SIZE):
            batch = verses[i:i+BATCH_SIZE]
            
            chunks = []
            metadatas = []
            ids = []

            for v in batch:
                # Add meaning (only once per ref_id to save time/cost?)
                # Actually, for research, we might want it for every record
                v['verse_meaning'] = "Meaning generation disabled for bulk ingest to save costs."
                
                chunks.append(v['text'])
                ids.append(f"{v['version']}_{v['ref_id']}")
                metadatas.append({
                    "book": v['book'],
                    "chapter": v['chapter'],
                    "verse": v['verse'],
                    "version": v['version'],
                    "reference": v['reference'],
                    "ref_id": v['ref_id'],
                    "verse_meaning": v['verse_meaning']
                })
            
            store.upsert_bible_chunks(chunks, metadatas, ids)
            print(f"Uploaded batch {i//BATCH_SIZE + 1} for {version}")

if __name__ == "__main__":
    main()
