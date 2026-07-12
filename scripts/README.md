# scripts/

One-time maintenance / migration scripts. These are **not** part of the normal
runtime — fresh ingests build the correct schema and normalized data directly
from `src/`. They are kept here for reproducibility when restoring `data/sermons.db`
from an older snapshot.

| Script | Purpose | Safe to re-run? |
|---|---|---|
| `migrate_db.py` | Adds `COLLATE NOCASE` to the text columns of `sermons` / `verses`. Already applied to the canonical DB. | Yes — idempotent; no-op if already migrated |
| `normalize_books.py` | Back-fills canonical book names in the `verses` table using `src/storage/normalize_book.py`. Supports `--dry-run`. | Yes — run with `--dry-run` first to preview |

## Usage

```bash
source .venv/bin/activate

# Preview book-name normalization without writing
python scripts/normalize_books.py --dry-run

# Apply
python scripts/normalize_books.py

# One-time collation migration (only needed on old snapshots)
python scripts/migrate_db.py
```

> New installs do not need these scripts — the schema in
> `src/storage/sqlite_store.py` and the write-time normalization in
> `SermonRegistry` already produce case-insensitive columns and canonical names.
