"""
BBTC Sermon Ingestion Pipeline

Usage:
  python ingest.py              # incremental — skip already-indexed NGs
  python ingest.py --wipe       # full rebuild from staging/
  python ingest.py --year 2024  # process only files for a specific year
"""

import argparse, os, re, sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.file_classifier import classify_file
from src.ingestion.sermon_grouper import group_sermon_files
from src.ingestion.ng_extractor import extract_ng_metadata, extract_ng_body
from src.ingestion.ps_extractor import (
    parse_verses_from_filename, extract_ps_text, extract_verses_from_text
)
from src.storage.sqlite_store import SermonRegistry
from src.storage.chroma_store import SermonVectorStore
from src.storage.normalize_speaker import normalize_speaker
from src.storage.normalize_book import normalize_book
from src.llm import get_llm

STAGING_DIR = "data/staging"
CHROMA_DIR = "data/chroma_db"
DB_PATH = "data/sermons.db"


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:80]


def _make_sermon_id(date: str | None, topic: str | None, ng_file: str) -> str:
    if date and topic:
        return f"{date}-{_slugify(topic)}"
    if date:
        return f"{date}-{_slugify(os.path.splitext(ng_file)[0][-40:])}"
    return _slugify(os.path.splitext(ng_file)[0][-60:])


def _extract_text_from_pdf(filepath: str) -> str:
    import fitz
    try:
        doc = fitz.open(filepath)
        return "\n".join(page.get_text() for page in doc).strip()
    except Exception:
        return ""


def _generate_summary(ng_body: str, topic: str | None, theme: str | None,
                      speaker: str | None, verse_refs: list[str], ps_text: str,
                      llm) -> str | None:
    if not llm or not ng_body:
        return None
    verses_str = ", ".join(verse_refs) if verse_refs else "not specified"
    prompt = (
        "Write a concise 3-5 sentence sermon summary capturing the main message, "
        "key spiritual insight, and practical application. Be specific — reference "
        "the topic and verses.\n\n"
        f"Topic: {topic or 'Unknown'}\n"
        f"Theme: {theme or 'Unknown'}\n"
        f"Speaker: {speaker or 'Unknown'}\n"
        f"Key Verses: {verses_str}\n\n"
        f"Sermon Notes:\n{ng_body[:2000]}\n\n"
        f"Slides Text:\n{ps_text[:500] if ps_text else 'Not available'}\n\n"
        "Summary:"
    )
    try:
        response = llm.invoke(prompt)
        return (response.content if hasattr(response, "content") else str(response)).strip()
    except Exception as e:
        print(f"  ⚠️  Summary generation failed: {e}")
        return None


def _detect_language(filename: str) -> str:
    if filename.startswith("Mandarin_"):
        return "Mandarin"
    return "English"


def process_group(group, registry: SermonRegistry, vector_store: SermonVectorStore,
                  llm, splitter: RecursiveCharacterTextSplitter, incremental: bool, force: bool = False):
    ng_file = group.ng
    ps_files = group.ps

    if not ng_file and not ps_files:
        return

    # Skip if already indexed in incremental mode (unless force is True)
    if incremental and not force and ng_file and registry.ng_file_indexed(ng_file):
        return

    ng_path = os.path.join(STAGING_DIR, ng_file) if ng_file else None
    ng_text = _extract_text_from_pdf(ng_path) if ng_path else ""

    # Extract NG metadata
    meta = extract_ng_metadata(ng_text, ng_file or "") if ng_text else {}
    date = meta.get("date")
    speaker = meta.get("speaker")
    topic = meta.get("topic")
    theme = meta.get("theme")
    language = _detect_language(ng_file or (ps_files[0] if ps_files else "English_"))
    ng_body = extract_ng_body(ng_text) if ng_text else ""

    # Extract PS verses
    all_verses = []
    ps_text_combined = ""
    ps_file = ps_files[0] if ps_files else None
    for pf in ps_files:
        verses = parse_verses_from_filename(pf)
        all_verses.extend(verses)
        ps_path = os.path.join(STAGING_DIR, pf)
        ps_text_combined += extract_ps_text(ps_path) + "\n"

    # LLM verse extraction from PS text (always try if text is available)
    if ps_text_combined.strip() and llm:
        llm_verse_refs = extract_verses_from_text(ps_text_combined, llm)
        
        # De-duplicate: track existing normalized refs
        # We normalize by removing spaces and lowercasing
        existing_refs = {v["verse_ref"].lower().replace(" ", "") for v in all_verses}

        for ref in llm_verse_refs:
            norm_ref = ref.lower().replace(" ", "")
            if norm_ref in existing_refs:
                continue
            
            m = re.match(r'^(\w+(?:\s\w+)?)\s+(\d+)(?::(\d+)(?:-(\d+))?)?$', ref)
            if m:
                canonical_book = normalize_book(m.group(1))
                if canonical_book is None:
                    continue
                all_verses.append({
                    "verse_ref": ref, 
                    "book": canonical_book,
                    "chapter": int(m.group(2)),
                    "verse_start": int(m.group(3)) if m.group(3) else None,
                    "verse_end": int(m.group(4)) if m.group(4) else None,
                    "is_key_verse": 0,
                })
                existing_refs.add(norm_ref)

    # If we found verses but none are marked as key, mark the first one
    if all_verses and not any(v.get("is_key_verse") for v in all_verses):
        all_verses[0]["is_key_verse"] = 1

    key_verse = all_verses[0]["verse_ref"] if all_verses else None
    verse_refs = [v["verse_ref"] for v in all_verses]

    # Generate unified summary
    summary = _generate_summary(ng_body, topic, theme, speaker, verse_refs, ps_text_combined, llm)

    sermon_id = _make_sermon_id(date, topic, ng_file or (ps_files[0] if ps_files else "unknown"))

    if force:
        print(f"  🔄 Force re-ingesting {sermon_id}...")
        registry.delete_verses(sermon_id)

    print(f"  📖 {sermon_id} | {speaker} | {date} | {key_verse}")

    # Store in SQLite
    registry.upsert_sermon({
        "sermon_id": sermon_id,
        "date": date,
        "year": int(date[:4]) if date else None,
        "language": language,
        "speaker": speaker,
        "topic": topic,
        "theme": theme,
        "summary": summary,
        "key_verse": key_verse,
        "ng_file": ng_file,
        "ps_file": ps_file,
        "status": "extracted",
    })

    for verse in all_verses:
        registry.insert_verse({"sermon_id": sermon_id, **verse})

    # Build ChromaDB docs
    chunk_meta = {
        "sermon_id": sermon_id,
        "speaker": speaker or "",
        "date": date or "",
        "year": int(date[:4]) if date else 0,
        "topic": topic or "",
        "theme": theme or "",
        "language": language,
        "key_verse": key_verse or "",
    }

    docs, metas, ids = [], [], []

    # Body chunks
    if ng_body:
        chunks = splitter.split_text(ng_body) or [ng_body[:800]]
        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            metas.append({**chunk_meta, "doc_type": "body"})
            ids.append(f"{sermon_id}_body_{i}")

    # Summary chunk
    if summary:
        docs.append(summary)
        metas.append({**chunk_meta, "doc_type": "summary"})
        ids.append(f"{sermon_id}_summary")

    if docs:
        vector_store.upsert_sermon_chunks(docs, metas, ids)

    registry.mark_status(sermon_id, "indexed")


def run_pipeline(wipe: bool = False, year: int | None = None, incremental: bool = True, force: bool = False):
    print("🚀 BBTC Sermon Ingestion Pipeline")

    registry = SermonRegistry(db_path=DB_PATH)
    vector_store = SermonVectorStore(persist_dir=CHROMA_DIR)
    llm = get_llm()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    if wipe:
        print("🗑️  Wiping SQLite and ChromaDB...")
        registry.wipe()
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection("sermon_collection")
        except Exception:
            pass
        vector_store = SermonVectorStore(persist_dir=CHROMA_DIR)
        incremental = False

    if not os.path.isdir(STAGING_DIR):
        print(f"⚠️ Staging directory not found. Creating {STAGING_DIR}...")
        os.makedirs(STAGING_DIR, exist_ok=True)
        print("💡 Hint: Run 'make scrape' to download sermon files before ingesting.")
        sys.exit(0)

    all_files = os.listdir(STAGING_DIR)
    if not all_files:
        print(f"⚠️ Staging directory ({STAGING_DIR}) is empty.")
        print("💡 Hint: Run 'make scrape' to download sermon files before ingesting.")
        sys.exit(0)

    if year:
        all_files = [f for f in all_files if f"_{year}_" in f]
    # Only NG and PS
    sermon_files = [f for f in all_files if classify_file(f) in ("ng", "ps")]
    if not sermon_files:
        print("⚠️ No valid NG/PS files found in staging.")
        print("💡 Hint: Run 'make scrape' to download sermon files before ingesting.")
        sys.exit(0)

    print(f"📁 Found {len(sermon_files)} NG/PS files in staging/")

    groups = group_sermon_files(sermon_files)
    print(f"📦 Formed {len(groups)} sermon groups")

    indexed = 0
    skipped = 0
    failed = 0
    for group in groups:
        try:
            ng = group.ng
            if incremental and not force and ng and registry.ng_file_indexed(ng):
                skipped += 1
                continue
            process_group(group, registry, vector_store, llm, splitter, incremental, force)
            indexed += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1

    print(f"\n✅ Done: {indexed} indexed, {skipped} skipped, {failed} failed")
    counts = vector_store.counts()
    print(f"📊 ChromaDB: {counts['sermon_collection']} chunks in sermon_collection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BBTC Sermon Ingestion Pipeline")
    parser.add_argument("--wipe", action="store_true", help="Wipe and rebuild from scratch")
    parser.add_argument("--year", type=int, help="Process only files for this year")
    parser.add_argument("--force", action="store_true", help="Re-process even if already indexed (without full wipe)")
    args = parser.parse_args()
    run_pipeline(wipe=args.wipe, year=args.year, incremental=not args.wipe, force=args.force)
