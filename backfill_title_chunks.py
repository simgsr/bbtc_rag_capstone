"""Backfill ``doc_type="metadata"`` title chunks into ``sermon_collection``.

The ingest pipeline now indexes a compact "Topic | Theme | Speaker | Key verse |
Date" chunk for every sermon (see ``ingest.py``) so topical/title queries
retrieve the right sermon instead of depending on body-text overlap. Existing
sermons ingested before that change lack this chunk. This script backfills it
for every indexed sermon directly from ``sermons.db`` — no PDF re-parse, no LLM
summary regeneration, just a short embedding per sermon — so the retrieval
improvement is immediate without a full ``ingest.py --wipe``.

Usage:
    python backfill_title_chunks.py          # backfill all indexed sermons
    python backfill_title_chunks.py --dry    # show what would be upserted, write nothing

Idempotent: chunk id is ``{sermon_id}_metadata``, so re-runs upsert (overwrite)
the same row rather than duplicate.
"""
import argparse
from src.storage.sqlite_store import SermonRegistry
from src.storage.chroma_store import SermonVectorStore
from src.ingestion.title_chunk import build_sermon_title_text

DB_PATH = "data/sermons.db"
CHROMA_DIR = "data/chroma_db"


def main():
    p = argparse.ArgumentParser(description="Backfill sermon title/metadata chunks")
    p.add_argument("--dry", action="store_true", help="print plan, write nothing")
    args = p.parse_args()

    registry = SermonRegistry(db_path=DB_PATH)
    # Use the registry's public accessor (returns list[dict] with every column,
    # including `language`) rather than reaching into a private _conn. Filtering to
    # indexed sermons in Python avoids a second hand-rolled SQL connection.
    rows = [r for r in registry.get_all_sermons() if r.get("status") == "indexed"]

    docs, metas, ids = [], [], []
    skipped = 0
    for r in rows:
        text = build_sermon_title_text(
            r.get("topic"), r.get("theme"), r.get("speaker"),
            r.get("key_verse"), r.get("date"),
        )
        if not text:
            skipped += 1
            continue
        docs.append(text)
        metas.append({
            "sermon_id": r["sermon_id"],
            "speaker": r.get("speaker") or "",
            "date": r.get("date") or "",
            "year": int(r["date"][:4]) if r.get("date") else 0,
            "topic": r.get("topic") or "",
            "theme": r.get("theme") or "",
            "language": r.get("language") or "English",  # preserve actual language
            "key_verse": r.get("key_verse") or "",
            "doc_type": "metadata",
        })
        ids.append(f"{r['sermon_id']}_metadata")

    print(f"Prepared {len(docs)} title chunks ({skipped} sermons skipped — no metadata).")
    if args.dry:
        for d, i in zip(docs[:5], ids[:5]):
            print(f"  [{i}] {d}")
        if len(docs) > 5:
            print(f"  … and {len(docs)-5} more")
        print("Dry run — nothing written.")
        return

    if not docs:
        print("Nothing to backfill.")
        return

    vs = SermonVectorStore(persist_dir=CHROMA_DIR)
    vs.upsert_sermon_chunks(docs, metas, ids)
    counts = vs.counts()
    print(f"✅ Backfilled. sermon_collection now has {counts['sermon_collection']} chunks.")


if __name__ == "__main__":
    main()