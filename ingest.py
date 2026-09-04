"""
BBTC Sermon Ingestion Pipeline

Usage:
  python ingest.py              # incremental — skip already-indexed NGs
  python ingest.py --wipe       # full rebuild from staging/
  python ingest.py --year 2024  # process only files for a specific year
"""

import argparse, os, re
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.file_classifier import classify_file
from src.ingestion.sermon_grouper import group_sermon_files
from src.ingestion.title_chunk import build_sermon_title_text
from src.ingestion.ng_extractor import extract_ng_metadata, extract_ng_body
from src.ingestion.ps_extractor import (
    parse_verses_from_filename, parse_verses_from_text,
    extract_ps_text, extract_verses_from_text
)
from src.ingestion.vision_extractor import (
    render_pdf_pages, extract_from_images, extract_verses_from_images
)
from src.storage.sqlite_store import SermonRegistry
from src.storage.chroma_store import SermonVectorStore
from src.storage.normalize_book import normalize_book
from src.llm import get_ingest_llm, get_vision_llm

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
        print(f"  ⚠️  Summary generation failed: {e}", flush=True)
        return None


def _detect_language(filename: str) -> str:
    if filename.startswith("Mandarin_"):
        return "Mandarin"
    return "English"


# Tokens that leak into a speaker name from the filename parser (e.g.
# "Edric Sng Member27S Guide", "Daniel Foo_Notes") — a speaker containing any
# of these is garbage and should be replaced by the vision model's reading.
_SPEAKER_ARTIFACTS = (
    "member27s", "guide", "copy", "members", "notes", "slides", "ppt",
    "compressed", "final", "v2", "v3",
)


def _is_garbage_speaker(speaker: str) -> bool:
    low = speaker.lower()
    return any(tok in low for tok in _SPEAKER_ARTIFACTS)


def _add_verse_refs(refs: list[str], all_verses: list[dict], existing_refs: set[str]) -> None:
    """Parse raw verse-ref strings (e.g. from LLM/vision extraction), dedup, and
    append to ``all_verses``. Refs that don't match a canonical book are dropped."""
    for ref in refs:
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


def process_group(group, registry: SermonRegistry, vector_store: SermonVectorStore,
                  llm, splitter: RecursiveCharacterTextSplitter, incremental: bool,
                  force: bool = False, vision_llm=None):
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
    # Use website publish date as fallback — more reliable than filename heuristics
    if not date and group.page_date:
        date = group.page_date
    speaker = meta.get("speaker")
    topic = meta.get("topic")
    theme = meta.get("theme")
    language = _detect_language(ng_file or (ps_files[0] if ps_files else "English_"))
    ng_body = extract_ng_body(ng_text) if ng_text else ""

    # Vision fallback: when the NG text is missing or yields no usable topic
    # (textless image-based PDF, a PS-only group with no NG file, or a PDF whose
    # labeled fields extract to garbage like "A"/"K"), render the pages and ask
    # the multimodal vision model to read them. Vision fills gaps only — it
    # never overwrites text-derived values.
    vision_meta = {}
    if vision_llm:
        if ng_file and (not ng_text or not topic or len(topic.strip()) < 2):
            print(f"    👁️  Vision-extracting metadata from {ng_file} ...", flush=True)
            vision_meta = extract_from_images(render_pdf_pages(ng_path), vision_llm)
        elif not ng_file and ps_files:
            ps0_path = os.path.join(STAGING_DIR, ps_files[0])
            print(f"    👁️  Vision-extracting metadata from PS-only {ps_files[0]} ...", flush=True)
            vision_meta = extract_from_images(render_pdf_pages(ps0_path), vision_llm)
        # Vision fills gaps only — but a text-derived topic of < 2 chars is
        # garbage (e.g. "A"/"K" from a broken layout), so vision's topic wins.
        if vision_meta.get("topic") and (not topic or len(topic.strip()) < 2):
            topic = vision_meta["topic"]
        # Same for speakers polluted by filename-parser artifacts.
        if vision_meta.get("speaker") and (not speaker or _is_garbage_speaker(speaker)):
            speaker = vision_meta["speaker"]
        # And for themes that extract to a single char (e.g. "7").
        if vision_meta.get("theme") and (not theme or len(theme.strip()) < 2):
            theme = vision_meta["theme"]
        date = date or vision_meta.get("date")

    # Extract PS verses
    all_verses = []
    ps_text_combined = ""
    ps_file = ps_files[0] if ps_files else None
    for pf in ps_files:
        verses = parse_verses_from_filename(pf)
        all_verses.extend(verses)
        ps_path = os.path.join(STAGING_DIR, pf)
        ps_text_combined += extract_ps_text(ps_path) + "\n"

    # Extract verse refs from NG topic and body text (deterministic, no LLM cost).
    # This is the primary source for sermons whose PS file is image-only or missing.
    existing_refs = {v["verse_ref"].lower().replace(" ", "") for v in all_verses}
    for src in [topic, ng_body]:
        if not src:
            continue
        # Strip the speaker's own name tokens so e.g. "Daniel Foo" isn't parsed as
        # the book of Daniel. A real citation keeps its chapter number (e.g.
        # "Mark 3:16"), so only strip a name token when no digit follows it.
        for tok in re.findall(r'[A-Za-z]{3,}', speaker or ""):
            src = re.sub(rf'\b{re.escape(tok)}\b(?!\s*\d)', ' ', src, flags=re.IGNORECASE)
        for v in parse_verses_from_text(src):
            norm = v["verse_ref"].lower().replace(" ", "")
            if norm not in existing_refs:
                all_verses.append(v)
                existing_refs.add(norm)

    # LLM verse extraction from PS text (always try if text is available)
    existing_refs = {v["verse_ref"].lower().replace(" ", "") for v in all_verses}
    if ps_text_combined.strip() and llm:
        _add_verse_refs(extract_verses_from_text(ps_text_combined, llm), all_verses, existing_refs)

    # Vision fallback for textless PS slides: no text to read, so render the
    # pages and ask the vision model for the verse refs shown on the slides.
    if not ps_text_combined.strip() and ps_files and vision_llm:
        ps0_path = os.path.join(STAGING_DIR, ps_files[0])
        print(f"    👁️  Vision-extracting verses from {ps_files[0]} ...", flush=True)
        _add_verse_refs(extract_verses_from_images(render_pdf_pages(ps0_path), vision_llm),
                        all_verses, existing_refs)

    # Vision key verse (from the metadata pass) also feeds the verse list.
    if vision_meta.get("key_verse"):
        _add_verse_refs(re.split(r'[;,]\s*', vision_meta["key_verse"]), all_verses, existing_refs)

    # Drop book-only references (no chapter): a bare book name is too unreliable
    # to store as a preached verse — it collides with speaker names and common
    # words. Keep only refs with an actual chapter number.
    all_verses = [v for v in all_verses if v.get("chapter")]

    # If we found verses but none are marked as key, mark the first one
    if all_verses and not any(v.get("is_key_verse") for v in all_verses):
        all_verses[0]["is_key_verse"] = 1

    key_verse = all_verses[0]["verse_ref"] if all_verses else None
    verse_refs = [v["verse_ref"] for v in all_verses]

    # Generate unified summary. For textless NG / PS-only groups there is no
    # body text to summarise — fall back to the vision model's own summary.
    print(f"    🧠 Summarising ({topic or 'unknown topic'}) ...", flush=True)
    if ng_body:
        summary = _generate_summary(ng_body, topic, theme, speaker, verse_refs, ps_text_combined, llm)
    else:
        summary = vision_meta.get("summary")

    sermon_id = _make_sermon_id(date, topic, ng_file or (ps_files[0] if ps_files else "unknown"))

    if force:
        print(f"  🔄 Force re-ingesting {sermon_id}...", flush=True)
        # The sermon_id may have changed since the last ingest (e.g. vision
        # recovered a real topic where the old row had "A"/"K"). Remove the
        # previously-indexed version so we don't leave an orphaned duplicate.
        old = registry.get_sermon_by_file(ng_file, ps_file)
        if old and old["sermon_id"] != sermon_id:
            print(f"    🧹 Removing old version {old['sermon_id']} ...", flush=True)
            registry.delete_sermon(old["sermon_id"])
            vector_store.delete_sermon_chunks(old["sermon_id"])
        registry.delete_verses(sermon_id)

    print(f"  📖 {sermon_id} | {speaker} | {date} | {key_verse}", flush=True)

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

    # Title/metadata chunk — always indexed (for every sermon, including textless
    # ones). Body+summary chunks encode prose, but a query that matches the sermon's
    # *title* (topic) or *theme* often won't surface them because the topic string
    # lives only in metadata, not in any embedded text. Indexing a compact
    # "Topic | Theme | Speaker | Key verse | Date" chunk makes topical/title queries
    # retrieve the right sermon directly. doc_type="metadata" so it can be told apart
    # from body/summary chunks at display time if needed. The text format is shared
    # with backfill_title_chunks.py via build_sermon_title_text so the two paths
    # can't drift.
    title_text = build_sermon_title_text(topic, theme, speaker, key_verse, date)
    if title_text:
        docs.append(title_text)
        metas.append({**chunk_meta, "doc_type": "metadata"})
        ids.append(f"{sermon_id}_metadata")

    if docs:
        vector_store.upsert_sermon_chunks(docs, metas, ids)

    registry.mark_status(sermon_id, "indexed")


def run_pipeline(wipe: bool = False, year: int | None = None, incremental: bool = True, force: bool = False):
    print("🚀 BBTC Sermon Ingestion Pipeline")

    # --- Cheap setup: SQLite only ---
    registry = SermonRegistry(db_path=DB_PATH)

    if wipe:
        print("🗑️  Wiping SQLite and ChromaDB...")
        registry.wipe()
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection("sermon_collection")
        except Exception:
            pass
        incremental = False

    # --- Early-exit checks (before loading BGE-M3 / LLM) ---
    if not os.path.isdir(STAGING_DIR):
        print(f"⚠️ Staging directory not found. Creating {STAGING_DIR}...")
        os.makedirs(STAGING_DIR, exist_ok=True)
        print("💡 Hint: Run 'make scrape' to download sermon files before ingesting.")
        return

    all_files = os.listdir(STAGING_DIR)
    if not all_files:
        print(f"⚠️ Staging directory ({STAGING_DIR}) is empty.")
        print("💡 Hint: Run 'make scrape' to download sermon files before ingesting.")
        return

    if year:
        all_files = [f for f in all_files if f"_{year}_" in f]
    sermon_files = [f for f in all_files if classify_file(f) in ("ng", "ps")]
    if not sermon_files:
        print("⚠️ No valid NG/PS files found in staging.")
        print("💡 Hint: Run 'make scrape' to download sermon files before ingesting.")
        return

    # In incremental mode, skip all expensive setup if nothing is new
    if incremental and not force:
        ng_files = [f for f in sermon_files if classify_file(f) == "ng"]
        ps_files = [f for f in sermon_files if classify_file(f) == "ps"]
        new_ngs = [f for f in ng_files if not registry.ng_file_indexed(f)]
        new_pss = [f for f in ps_files if not registry.ps_file_indexed(f)]
        if not new_ngs and not new_pss:
            print("✅ Nothing new to ingest.")
            return

    print(f"📁 Found {len(sermon_files)} NG/PS files in staging/")

    # --- Expensive setup: embeddings + LLM (only reached when there is work to do) ---
    vector_store = SermonVectorStore(persist_dir=CHROMA_DIR)
    llm = get_ingest_llm()
    # Vision model is a cheap object to create (the weights load on first use),
    # so it's built eagerly but only invoked for textless / PS-only groups.
    vision_llm = get_vision_llm()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    groups = group_sermon_files(sermon_files, staging_dir=STAGING_DIR)
    print(f"📦 Formed {len(groups)} sermon groups")

    indexed = 0
    skipped = 0
    failed = 0
    total = len(groups)
    for group in groups:
        try:
            ng = group.ng
            ps0 = group.ps[0] if group.ps else None
            if incremental and not force:
                if ng and registry.ng_file_indexed(ng):
                    skipped += 1
                    continue
                if not ng and ps0 and registry.ps_file_indexed(ps0):
                    skipped += 1
                    continue
            label = ng or (group.ps[0] if group.ps else "unknown")
            print(f"  ⏳ [{indexed + 1}/{total - skipped}] {label} ...", flush=True)
            process_group(group, registry, vector_store, llm, splitter, incremental, force, vision_llm)
            indexed += 1
        except Exception as e:
            print(f"  ❌ Error: {e}", flush=True)
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
