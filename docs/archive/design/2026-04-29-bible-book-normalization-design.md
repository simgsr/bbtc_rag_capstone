# Bible Book Name Normalization — Design Spec

**Date:** 2026-04-29
**Status:** Approved

## Problem

The `verses.book` column stores inconsistent values — ALL_CAPS (`HEBREWS`), abbreviations (`Lk`, `Heb`), unnumbered ambiguous names (`Samuel`, `Corinthians`), and garbage (`Jericho`). This makes book-based queries (gap analysis, frequency charts) unreliable.

Root causes:
1. `_VERSE_RE` in `ps_extractor.py` does not capture the leading number prefix from filenames like `1-SAMUEL-9V1`, so "1 Samuel" is stored as "Samuel".
2. There is no normalization at the storage layer — every variant lands verbatim in the DB.
3. ~1,400 existing verse rows are already dirty.

## Approach

**Approach B: `normalize_book()` at storage layer + fix extraction + migrate existing data.**

Mirrors the existing `normalize_speaker` pattern already used for speaker names.

## Components

### 1. `src/storage/normalize_book.py` (new)

- `BOOK_MAP: dict[str, str]` — case-insensitive keys mapping every known raw variant to one of the 66 canonical book names (e.g. `"ACTS"`, `"Act"`, `"acts"` → `"Acts"`; `"1samuel"`, `"1 samuel"`, `"1-samuel"` → `"1 Samuel"`).
- Ambiguous unnumbered books (`"Samuel"`, `"Kings"`, `"Corinthians"`, `"Timothy"`, `"Peter"`, `"Chronicles"`) are intentionally excluded from `BOOK_MAP` — the migration handles them using chapter numbers.
- `normalize_book(raw: str) -> str | None` — case-insensitive lookup; returns `None` for unrecognized or confirmed garbage values (e.g. `"Jericho"`).

### 2. Fix `ps_extractor.py`

**`_BOOKS` dict:** Add numbered variants alongside existing base entries:
```python
"1samuel": "1 Samuel", "2samuel": "2 Samuel",
"1kings": "1 Kings",   "2kings": "2 Kings",
"1chronicles": "1 Chronicles", "2chronicles": "2 Chronicles",
"1corinthians": "1 Corinthians", "2corinthians": "2 Corinthians",
"1thessalonians": "1 Thessalonians", "2thessalonians": "2 Thessalonians",
"1timothy": "1 Timothy", "2timothy": "2 Timothy",
"1peter": "1 Peter",   "2peter": "2 Peter",
"1john": "1 John",     "2john": "2 John",    "3john": "3 John",
```
Base unnumbered entries (`"samuel": "Samuel"` etc.) stay as fallbacks.

**`_VERSE_RE`:** Extend to optionally capture a leading `1`, `2`, or `3` with any separator before the book name:
```python
rf'(?<![A-Za-z\d])([123][-_ ]?)?({_BOOK_PATTERN})...'
```

**`parse_verses_from_filename`:** Combine the captured prefix + book name to form the lookup key (e.g. `"1" + "samuel"` → `"1samuel"` → `"1 Samuel"`), falling back to the base key if the numbered key is not found.

### 3. Wire `normalize_book` into `sqlite_store.py`

In `insert_verse()`:
- Call `normalize_book(record["book"])` before inserting.
- If canonical name returned: update `record["book"]` and recompute `record["verse_ref"]` via `normalize_verse_ref()` from `ps_extractor.py`.
- If `None` returned (garbage): skip the insert.

In `ingest.py` (LLM path, ~line 126):
- Apply `normalize_book` to `m.group(1)` before storing `"book"` in the verse dict.

### 4. Migration script: `scripts/normalize_books.py`

One-time script run manually after deployment.

**Steps:**
1. Load all rows from `verses` (`id`, `book`, `chapter`, `verse_ref`, `sermon_id`, `verse_start`, `verse_end`, `is_key_verse`).
2. Apply `normalize_book()` to each row's `book` — resolves ALL_CAPS and abbreviation variants.
3. Disambiguate unnumbered numbered books using chapter numbers for rows `normalize_book` cannot resolve:
   - `"Samuel"`: chapter ≤ 24 is ambiguous (default `"1 Samuel"`); chapter 25–31 → `"1 Samuel"` only
   - `"Kings"`: chapter ≤ 22 ambiguous (default `"1 Kings"`); chapter 23–25 → `"2 Kings"`
   - `"Chronicles"`: chapter ≤ 29 ambiguous (default `"1 Chronicles"`); chapter 30–36 → `"2 Chronicles"`
   - `"Corinthians"`: chapter ≤ 13 ambiguous (default `"1 Corinthians"`); chapter 14–16 → `"1 Corinthians"`
   - `"Timothy"`: chapter ≤ 4 ambiguous (default `"1 Timothy"`); chapter 5–6 → `"1 Timothy"`
   - `"Peter"`: chapter ≤ 3 ambiguous (default `"1 Peter"`); chapter 4–5 → `"1 Peter"`
   - `"John"`: left as `"John"` (Gospel) — no reliable way to distinguish from epistles without full text
4. Recompute `verse_ref` from normalized book + stored chapter/verse fields.
5. Delete garbage rows (any row where normalized book is `None` after steps 2–3).
6. Handle post-normalization duplicates: within the same `sermon_id`, if two rows collapse to the same `verse_ref`, keep the one with `is_key_verse = 1`; otherwise keep the lower `id` and delete the other.
7. Print a summary: rows updated, rows deleted, rows with unresolved book (logged for inspection).

## Out of Scope

- Re-ingesting all sermons from scratch (migration fixes existing data without re-parsing PDFs).
- Normalizing `"John"` epistles vs Gospel (ambiguous without text context).
- Changing the ChromaDB metadata `key_verse` field (that is a denormalized display string and not used for book-level analytics).
