"""
Bible collection ingest pipeline.

Sources:
  - Scrollmapper (public domain): KJV, ASV, YLT, BBE — downloaded as JSON
  - Local EPUBs: any *.epub in data/bibles/ — version_id is auto-derived from
    the filename (first 2-5 letter all-caps token, e.g. "NIV.epub" → "NIV",
    "ESV The Holy Bible.epub" → "ESV", "圣经 CUV.epub" → "CUV").

Both sources produce the same verse dict format:
  {book, chapter, verse, text, version, reference}

Behaviour:
  - The data/bibles/ directory is the source of truth — drop an EPUB in,
    re-run, and it gets ingested. Remove a file and it stays in the index
    (data already there; explicit --wipe if you want to clean up).
  - A version is re-ingested only if it isn't already in bible_versions
    with status="indexed".

Usage:
  python -m src.ingestion.bible.bible_ingest                    # ingest everything new
  python -m src.ingestion.bible.bible_ingest --wipe             # wipe + full rebuild
  python -m src.ingestion.bible.bible_ingest --versions KJV NIV # filter to specific IDs
"""

import argparse
import glob
import json
import os
import re
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

# Public-domain translations available from scrollmapper.
# These are remote sources (no equivalent "folder to scan"), so they stay
# enumerated here. Listed scrollmapper IDs are always fetched on ingest.
SCROLLMAPPER_VERSIONS: dict[str, str] = {
    "KJV": "King James Version",
    "ASV": "American Standard Version (1901)",
    "YLT": "Young's Literal Translation",
    "BBE": "Bible in Basic English",
}

BIBLES_DIR = "data/bibles"

# Match the first 2-5 letter all-caps token in a filename.
# "NIV.epub" → NIV, "ESV The Holy Bible.epub" → ESV, "圣经 CUV.epub" → CUV.
_VERSION_TOKEN_RE = re.compile(r'\b([A-Z]{2,5})\b')


def _version_id_from_filename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = _VERSION_TOKEN_RE.search(stem.upper())
    return m.group(1) if m else stem.upper()


def discover_epubs(bibles_dir: str = BIBLES_DIR) -> list[tuple[str, str]]:
    """Scan bibles_dir for *.epub files. Returns [(version_id, filepath), ...]."""
    found: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(bibles_dir, "*.epub"))):
        vid = _version_id_from_filename(path)
        # If two files derive the same version_id, the first one wins (sorted order).
        found.setdefault(vid, path)
    return list(found.items())



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
            row = conn.execute(
                "SELECT 1 FROM bible_versions WHERE version_id=? AND status='indexed'",
                (version_id,),
            ).fetchone()
            return row is not None
    except sqlite3.OperationalError:
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

def _build_source_list(version_filter: list[str] | None) -> list[tuple[str, str, str | None]]:
    """Returns [(source_type, version_id, filepath_or_None), ...]
    source_type is 'scrollmapper' or 'epub'. Filter is applied case-insensitively."""
    sources: list[tuple[str, str, str | None]] = []
    for vid in SCROLLMAPPER_VERSIONS:
        sources.append(("scrollmapper", vid, None))
    for vid, path in discover_epubs():
        sources.append(("epub", vid, path))
    if version_filter:
        wanted = {v.upper() for v in version_filter}
        sources = [s for s in sources if s[1].upper() in wanted]
    return sources


def ingest_bible(
    versions: list[str] | None = None,
    wipe: bool = False,
    db_path: str = "data/sermons.db",
    chroma_dir: str = "data/chroma_db",
    logger=print,
):
    sources = _build_source_list(versions)
    if not sources:
        logger("⚠️  No bible sources found "
               f"(no scrollmapper versions configured and no EPUBs in {BIBLES_DIR}/).")
        return

    pending = sources if wipe else [s for s in sources if not _is_indexed(db_path, s[1])]
    if not pending:
        logger("✅ All discovered Bible versions already indexed — nothing to do.")
        return

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
        store = SermonVectorStore(persist_dir=chroma_dir)

    for source_type, version_id, filepath in pending:
        logger(f"\n📖 Ingesting {version_id} ({source_type}) ...")

        if source_type == "scrollmapper":
            verses = _fetch_scrollmapper(version_id, logger)
            source = f"scrollmapper/{version_id}.json"
        else:
            verses = _parse_epub(version_id, filepath, logger)
            source = filepath

        if not verses:
            logger(f"  ✗ No verses extracted for {version_id} — skipping (will retry next run)")
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
        "--versions", nargs="+", default=None, metavar="VERSION",
        help="Optional filter: only ingest these version IDs "
             "(default: all scrollmapper versions + every EPUB in data/bibles/)",
    )
    args = parser.parse_args()
    ingest_bible(versions=args.versions, wipe=args.wipe)
