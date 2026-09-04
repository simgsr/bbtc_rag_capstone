"""Re-ingest textless / PS-only sermon groups with the vision fallback.

The pre-vision ingest indexed image-based PDFs and PS-only groups with empty
or unknown metadata (no topic/speaker/theme/summary). This re-processes just
those groups with ``force=True`` so the vision model (``OLLAMA_VISION_MODEL``,
default ``gemma4:e4b``) can recover topic/speaker/theme/key-verse/summary from
the rendered pages.

A group is a "hard case" when any of:
  * it has no NG (notes) file at all — PS-only group
  * its NG text is empty or too short to extract metadata (< 50 chars)
  * any of its PS (slides) files is image-only with no extractable text
  * its sermon is already in the DB with missing/garbage metadata (topic
    missing or < 2 chars, or speaker missing) — e.g. labeled fields that
    extracted to "A"/"K"

Usage:
  python scripts/reingest_textless.py            # re-ingest all hard cases
  python scripts/reingest_textless.py --dry-run  # list what would be re-ingested
"""

import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import (
    STAGING_DIR, CHROMA_DIR, DB_PATH,
    _extract_text_from_pdf, process_group,
)
from src.ingestion.file_classifier import classify_file
from src.ingestion.sermon_grouper import group_sermon_files
from src.ingestion.ps_extractor import extract_ps_text
from src.storage.sqlite_store import SermonRegistry
from src.storage.chroma_store import SermonVectorStore
from src.llm import get_ingest_llm, get_vision_llm

# Text shorter than this is treated as "textless" (image-only PDF).
_TEXT_MIN = 50


def _poor_ng_files() -> set[str]:
    """NG files whose sermon is already in the DB with missing/garbage metadata."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT ng_file FROM sermons "
            "WHERE ng_file IS NOT NULL AND ng_file != '' "
            "AND (topic IS NULL OR length(topic) < 2 OR speaker IS NULL OR speaker = '')"
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def _is_hard_case(group, poor_ng: set[str]) -> tuple[bool, str]:
    """Return (is_hard_case, reason) for a sermon group."""
    if not group.ng:
        return True, "PS-only (no NG file)"
    if group.ng in poor_ng:
        return True, "poor metadata in DB (topic/speaker missing or garbage)"
    ng_text = _extract_text_from_pdf(os.path.join(STAGING_DIR, group.ng))
    if len(ng_text.strip()) < _TEXT_MIN:
        return True, f"textless NG ({len(ng_text.strip())} chars)"
    for pf in group.ps:
        ps_text = extract_ps_text(os.path.join(STAGING_DIR, pf))
        if len(ps_text.strip()) < _TEXT_MIN:
            return True, f"textless PS ({pf})"
    return False, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="list hard cases without re-ingesting")
    args = parser.parse_args()

    files = sorted(os.listdir(STAGING_DIR))
    sermon_files = [f for f in files if classify_file(f) in ("ng", "ps")]
    groups = group_sermon_files(sermon_files, staging_dir=STAGING_DIR)
    poor_ng = _poor_ng_files()

    hard = []
    for g in groups:
        is_hard, reason = _is_hard_case(g, poor_ng)
        if is_hard:
            hard.append((g, reason))

    print(f"📁 {len(groups)} groups total, {len(hard)} hard cases (textless / PS-only)")
    for g, reason in hard:
        label = g.ng or (g.ps[0] if g.ps else "unknown")
        print(f"  • {label}  [{reason}]")

    if args.dry_run or not hard:
        return

    print("\n🚀 Re-ingesting hard cases with vision fallback ...")
    registry = SermonRegistry(db_path=DB_PATH)
    vector_store = SermonVectorStore(persist_dir=CHROMA_DIR)
    llm = get_ingest_llm()
    vision_llm = get_vision_llm()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    done = 0
    failed = 0
    for g, reason in hard:
        label = g.ng or (g.ps[0] if g.ps else "unknown")
        print(f"  ⏳ [{done + 1}/{len(hard)}] {label} ...", flush=True)
        try:
            process_group(g, registry, vector_store, llm, splitter,
                          incremental=False, force=True, vision_llm=vision_llm)
            done += 1
        except Exception as e:
            print(f"  ❌ Error: {e}", flush=True)
            failed += 1

    print(f"\n✅ Done: {done} re-ingested, {failed} failed")
    counts = vector_store.counts()
    print(f"📊 ChromaDB: {counts['sermon_collection']} chunks in sermon_collection")


if __name__ == "__main__":
    main()
