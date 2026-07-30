"""Migrate ChromaDB collections from L2 to cosine distance IN PLACE.

BGE-M3 (and the MLX embedding variants) are trained for **cosine** similarity, but
the collections were created with Chroma's default **squared-L2** space (see
``src/storage/chroma_store.py``). Switching the metric does NOT require
re-embedding — the stored vectors are byte-for-byte identical; only the distance
function used at query time changes. This script therefore preserves every stored
embedding / document / metadata: **no LLM call, no re-embedding, no re-scrape.**

For each collection whose space is not already ``cosine``:
  1. page every row out (ids, embeddings, documents, metadatas) to temp files on disk
  2. delete the collection
  3. recreate it with ``hnsw:space=cosine``
  4. re-add the saved rows with their ORIGINAL embeddings
  5. verify the row count matches the original

A full filesystem backup of ``data/chroma_db`` is taken first (unless
``--no-backup``) so the operation is fully reversible: if anything goes wrong,
delete ``data/chroma_db`` and rename the backup back.

Usage:
    python scripts/migrate_chroma_cosine.py            # migrate (with fs backup)
    python scripts/migrate_chroma_cosine.py --dry      # report only, change nothing
    python scripts/migrate_chroma_cosine.py --no-backup
"""
import argparse
import pickle
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "data" / "chroma_db"
COLLECTIONS = ("sermon_collection", "bible_collection")
TARGET_SPACE = "cosine"
PAGE = 2000  # rows per page — bounds peak memory during dump/re-add


def _space_of(coll) -> str | None:
    cfg = getattr(coll, "configuration_json", None) or {}
    return (cfg.get("hnsw") or {}).get("space")


def _dump_pages(coll, out_dir: Path) -> int:
    """Page all rows to pickle files in out_dir. Returns the row count dumped."""
    total = coll.count()
    dumped = 0
    page_idx = 0
    while dumped < total:
        got = coll.get(
            include=["embeddings", "documents", "metadatas"],
            limit=PAGE, offset=dumped,
        )
        ids = got["ids"]
        if not ids:
            break
        # Embeddings come back as a numpy array; store as plain lists so the
        # re-add path is independent of numpy and Chroma's return typing.
        embs = [[float(x) for x in row] for row in got["embeddings"]]
        with (out_dir / f"page_{page_idx:05d}.pkl").open("wb") as f:
            pickle.dump(
                {"ids": ids, "embeddings": embs,
                 "documents": got["documents"], "metadatas": got["metadatas"]},
                f,
            )
        dumped += len(ids)
        page_idx += 1
        print(f"    dumped {dumped:,}/{total:,}", flush=True)
    return dumped


def _readd_pages(coll, in_dir: Path) -> int:
    added = 0
    for page_file in sorted(in_dir.glob("page_*.pkl")):
        with page_file.open("rb") as f:
            p = pickle.load(f)
        coll.add(ids=p["ids"], embeddings=p["embeddings"],
                 documents=p["documents"], metadatas=p["metadatas"])
        added += len(p["ids"])
        print(f"    re-added {added:,}", flush=True)
    return added


def migrate_collection(client, name: str, work_dir: Path, dry: bool) -> bool:
    try:
        coll = client.get_collection(name)
    except Exception:
        print(f"  {name}: not present — skipping")
        return True
    space, count = _space_of(coll), coll.count()
    if space == TARGET_SPACE:
        print(f"  {name}: already '{TARGET_SPACE}' ({count:,} rows) — skipping")
        return True
    print(f"  {name}: '{space}' → '{TARGET_SPACE}' ({count:,} rows)")
    if dry:
        return True
    if count == 0:
        client.delete_collection(name)
        client.create_collection(name, metadata={"hnsw:space": TARGET_SPACE})
        print(f"  {name}: recreated empty as '{TARGET_SPACE}'")
        return True

    out_dir = work_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dumped = _dump_pages(coll, out_dir)
    if dumped != count:
        print(f"  ❌ {name}: dumped {dumped:,} != count {count:,} — ABORTING before delete")
        return False

    client.delete_collection(name)
    new = client.create_collection(name, metadata={"hnsw:space": TARGET_SPACE})
    added = _readd_pages(new, out_dir)
    final_space, final_count = _space_of(new), new.count()
    if added != count or final_count != count or final_space != TARGET_SPACE:
        print(f"  ❌ {name}: verify FAILED (added={added:,} count={final_count:,} "
              f"space={final_space}). Dump kept at {out_dir}; restore from the fs backup.")
        return False
    print(f"  ✅ {name}: {final_count:,} rows, space='{final_space}'")
    shutil.rmtree(out_dir, ignore_errors=True)
    return True


def main():
    ap = argparse.ArgumentParser(description="Migrate ChromaDB collections to cosine distance")
    ap.add_argument("--dry", action="store_true", help="report only, change nothing")
    ap.add_argument("--no-backup", action="store_true", help="skip the filesystem backup")
    args = ap.parse_args()

    if not CHROMA_DIR.exists():
        print(f"No ChromaDB at {CHROMA_DIR} — nothing to migrate.")
        return 0

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    pending = []
    for name in COLLECTIONS:
        try:
            if _space_of(client.get_collection(name)) != TARGET_SPACE:
                pending.append(name)
        except Exception:
            pass
    if not pending:
        print(f"All collections already on '{TARGET_SPACE}'. Nothing to do.")
        return 0
    print(f"Collections needing migration: {', '.join(pending)}")

    if not args.dry and not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = CHROMA_DIR.with_name(f"chroma_db.pre-cosine.{stamp}.bak")
        print(f"📦 Backing up {CHROMA_DIR} → {backup} ...", flush=True)
        # Release the client's file locks before copying the store.
        del client
        shutil.copytree(CHROMA_DIR, backup)
        print("   backup complete.")
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    ok = True
    with tempfile.TemporaryDirectory(prefix="chroma_cosine_") as tmp:
        work_dir = Path(tmp)
        for name in COLLECTIONS:
            ok = migrate_collection(client, name, work_dir, args.dry) and ok

    if not ok:
        print("\n⚠️  Migration incomplete — see messages above. The fs backup is intact.")
        return 1
    print("\n✅ Migration complete." if not args.dry else "\n(dry run — nothing changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
