"""
Bible collection ingest pipeline.

Sources:
  - Scrollmapper (public domain): KJV, ASV, WEB, YLT, BBE — downloaded as JSON
  - Local EPUBs (owned copies): NIV, ESV — parsed via epub_parser.py

Both sources produce the same verse dict format:
  {book, chapter, verse, text, version, reference}

Usage:
  python -m src.ingestion.bible.bible_ingest            # ingest all configured versions
  python -m src.ingestion.bible.bible_ingest --wipe     # wipe bible_collection first
  python -m src.ingestion.bible.bible_ingest --versions KJV WEB NIV
"""

import argparse
import json
import sqlite3
import sys
import urllib.request
from datetime import datetime

from src.storage.chroma_store import SermonVectorStore
from src.storage.normalize_book import normalize_book

_SCROLLMAPPER_BASE = (
    "https://raw.githubusercontent.com/scrollmapper/"
    "bible_databases/master/formats/json/{version_id}.json"
)

# Scrollmapper book names → our canonical names
_BOOK_NAME_MAP: dict[str, str] = {
    # Roman numeral prefixes
    "I Samuel":          "1 Samuel",
    "II Samuel":         "2 Samuel",
    "I Kings":           "1 Kings",
    "II Kings":          "2 Kings",
    "I Chronicles":      "1 Chronicles",
    "II Chronicles":     "2 Chronicles",
    "I Corinthians":     "1 Corinthians",
    "II Corinthians":    "2 Corinthians",
    "I Thessalonians":   "1 Thessalonians",
    "II Thessalonians":  "2 Thessalonians",
    "I Timothy":         "1 Timothy",
    "II Timothy":        "2 Timothy",
    "I Peter":           "1 Peter",
    "II Peter":          "2 Peter",
    "I John":            "1 John",
    "II John":           "2 John",
    "III John":          "3 John",
    # Other variants
    "Revelation of John": "Revelation",
    "Song of Solomon":    "Song of Songs",
    "Psalm":              "Psalms",
}

# Translations available from scrollmapper (public domain)
SCROLLMAPPER_VERSIONS: dict[str, str] = {
    "KJV": "King James Version",
    "ASV": "American Standard Version (1901)",
    "YLT": "Young's Literal Translation",
    "BBE": "Bible in Basic English",
}

import glob

# Local EPUB files (owned, copyrighted)
LOCAL_EPUB_VERSIONS: dict[str, str] = {
    "NIV": "data/bibles/NIV.epub",
    "ESV": "data/bibles/ESV The Holy Bible.epub",
}

# Default set to ingest
DEFAULT_VERSIONS = ["KJV", "ASV", "YLT", "NIV", "ESV"]

# Auto-detect other EPUBs in data/bibles/
for epub_file in glob.glob("data/bibles/*.epub"):
    import os
    filename = os.path.basename(epub_file)
    version_id = os.path.splitext(filename)[0].upper()
    # Don't override if already exists with a different path (like ESV)
    if version_id not in LOCAL_EPUB_VERSIONS and "ESV" not in filename:
        LOCAL_EPUB_VERSIONS[version_id] = epub_file
        if version_id not in DEFAULT_VERSIONS:
            DEFAULT_VERSIONS.append(version_id)



# ── Verse dict format ─────────────────────────────────────────────────────────

def _make_verse(book: str, chapter: int, verse: int, text: str, version: str) -> dict:
    return {
        "book":      book,
        "chapter":   chapter,
        "verse":     verse,
        "text":      text.strip(),
        "version":   version,
        "reference": f"{book} {chapter}:{verse}",
    }


# ── Scrollmapper source ───────────────────────────────────────────────────────

def _fetch_scrollmapper(version_id: str, logger=print) -> list[dict]:
    url = _SCROLLMAPPER_BASE.format(version_id=version_id)
    logger(f"  Downloading {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger(f"  ✗ Download failed: {e}")
        return []

    verses: list[dict] = []
    for book_obj in data.get("books", []):
        raw_name = book_obj.get("name", "")
        canonical = _BOOK_NAME_MAP.get(raw_name) or normalize_book(raw_name) or raw_name
        for ch_obj in book_obj.get("chapters", []):
            chapter = ch_obj.get("chapter", 0)
            for v_obj in ch_obj.get("verses", []):
                verse = v_obj.get("verse", 0)
                text  = v_obj.get("text", "").strip()
                if text and canonical and chapter and verse:
                    verses.append(_make_verse(canonical, chapter, verse, text, version_id))

    logger(f"  Parsed {len(verses):,} verses from scrollmapper/{version_id}")
    return verses


# ── Local EPUB source ─────────────────────────────────────────────────────────

def _parse_epub(version_id: str, filepath: str, logger=print) -> list[dict]:
    try:
        from src.ingestion.bible.epub_parser import BibleEpubParser
    except ImportError as e:
        logger(f"  ✗ epub_parser unavailable: {e}")
        return []

    import os
    if not os.path.exists(filepath):
        logger(f"  ✗ File not found: {filepath}")
        return []

    logger(f"  Parsing {filepath} ...")
    parser = BibleEpubParser(filepath, version_id)
    raw_verses = parser.parse()

    verses: list[dict] = []
    for v in raw_verses:
        canonical = normalize_book(v["book"]) or v["book"]
        text = v.get("text", "").strip()
        if text and v.get("chapter") and v.get("verse"):
            verses.append(
                _make_verse(canonical, v["chapter"], v["verse"], text, version_id)
            )

    logger(f"  Parsed {len(verses):,} verses from epub/{version_id}")
    return verses


# ── ChromaDB upsert ───────────────────────────────────────────────────────────

_BATCH_SIZE = 200

def _upsert_verses(store: SermonVectorStore, verses: list[dict], logger=print):
    total = len(verses)
    for start in range(0, total, _BATCH_SIZE):
        batch = verses[start : start + _BATCH_SIZE]
        chunks   = [v["text"] for v in batch]
        ids      = [f"{v['version']}_{v['reference']}" for v in batch]
        metadatas = [
            {
                "book":      v["book"],
                "chapter":   v["chapter"],
                "verse":     v["verse"],
                "version":   v["version"],
                "reference": v["reference"],
            }
            for v in batch
        ]
        store.upsert_bible_chunks(chunks, metadatas, ids)
        if start % (_BATCH_SIZE * 10) == 0 and start > 0:
            logger(f"    Uploaded {start + len(batch):,}/{total:,} ...")
    logger(f"  ✓ Upserted {total:,} verses to bible_collection")


# ── SQLite bible_versions tracking ───────────────────────────────────────────

def _is_indexed(db_path: str, version_id: str) -> bool:
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT status FROM bible_versions WHERE version_id=?", (version_id,)
            )
            row = cursor.fetchone()
            if row and row[0] == "indexed":
                return True
    except sqlite3.OperationalError:
        pass
    return False

def _mark_indexed(db_path: str, version_id: str, source: str):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bible_versions (
                version_id   TEXT PRIMARY KEY,
                filename     TEXT,
                status       TEXT,
                date_indexed TEXT
            )
        """)
        conn.execute(
            "INSERT OR REPLACE INTO bible_versions VALUES (?,?,?,?)",
            (version_id, source, "indexed", datetime.utcnow().isoformat()),
        )

# ── Main pipeline ─────────────────────────────────────────────────────────────

def ingest_bible(
    versions: list[str] | None = None,
    wipe: bool = False,
    db_path: str = "data/sermons.db",
    chroma_dir: str = "data/chroma_db",
    logger=print,
):
    if versions is None:
        versions = DEFAULT_VERSIONS

    store = SermonVectorStore(persist_dir=chroma_dir)

    if wipe:
        logger("🗑  Wiping bible_collection ...")
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        try:
            client.delete_collection("bible_collection")
            logger("   Collection deleted.")
        except Exception:
            pass
        # Re-init store so it recreates the collection
        store = SermonVectorStore(persist_dir=chroma_dir)

    for version_id in versions:
        if not wipe and _is_indexed(db_path, version_id):
            logger(f"  ⏭️  Skipping {version_id} (already indexed)")
            continue

        logger(f"\n📖 Ingesting {version_id} ...")

        if version_id in SCROLLMAPPER_VERSIONS:
            verses = _fetch_scrollmapper(version_id, logger)
            source = f"scrollmapper/{version_id}.json"
        elif version_id in LOCAL_EPUB_VERSIONS:
            filepath = LOCAL_EPUB_VERSIONS[version_id]
            verses = _parse_epub(version_id, filepath, logger)
            source = filepath
        else:
            logger(f"  ✗ Unknown version '{version_id}' — skipping")
            continue

        if not verses:
            logger(f"  ✗ No verses extracted for {version_id} — skipping")
            continue

        _upsert_verses(store, verses, logger)
        _mark_indexed(db_path, version_id, source)
        logger(f"  ✓ {version_id} done")

    counts = store.counts()
    logger(f"\n✅ bible_collection: {counts['bible_collection']:,} chunks total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Bible translations into ChromaDB")
    parser.add_argument(
        "--wipe", action="store_true",
        help="Wipe bible_collection before ingesting"
    )
    parser.add_argument(
        "--versions", nargs="+",
        default=DEFAULT_VERSIONS,
        metavar="VERSION",
        help=f"Versions to ingest (default: {' '.join(DEFAULT_VERSIONS)})",
    )
    args = parser.parse_args()
    ingest_bible(versions=args.versions, wipe=args.wipe)
