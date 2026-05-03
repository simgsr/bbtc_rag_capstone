
import re

# Mock data
all_verses = [
    {"verse_ref": "Luke 9:23", "book": "Luke", "chapter": 9, "verse_start": 23, "is_key_verse": 1}
]
llm_verse_refs = ["Luke 9:23", "John 3:16", "Romans 8:28"]

def normalize_book(book):
    return book

# Simulation of updated ingest.py logic
existing_refs = {v["verse_ref"].lower().replace(" ", "") for v in all_verses}

for ref in llm_verse_refs:
    norm_ref = ref.lower().replace(" ", "")
    if norm_ref in existing_refs:
        print(f"Skipping duplicate: {ref}")
        continue
    
    print(f"Adding new verse: {ref}")
    m = re.match(r'^(\w+(?:\s\w+)?)\s+(\d+)(?::(\d+)(?:-(\d+))?)?$', ref)
    if m:
        canonical_book = normalize_book(m.group(1))
        all_verses.append({
            "verse_ref": ref, 
            "book": canonical_book,
            "chapter": int(m.group(2)),
            "verse_start": int(m.group(3)) if m.group(3) else None,
            "verse_end": int(m.group(4)) if m.group(4) else None,
            "is_key_verse": 0,
        })
        existing_refs.add(norm_ref)

print("\nFinal all_verses:")
for v in all_verses:
    print(f"- {v['verse_ref']} (Key: {v['is_key_verse']})")
