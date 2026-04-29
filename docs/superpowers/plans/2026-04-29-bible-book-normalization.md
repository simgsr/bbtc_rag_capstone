# Bible Book Name Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize all Bible book names stored in the `verses` table to the canonical 66-book set, fix existing dirty data, and prevent dirty data in future ingestion.

**Architecture:** New `normalize_book()` function at the storage layer (mirrors `normalize_speaker` pattern); fix `_VERSE_RE` in `ps_extractor.py` to capture numeric book prefixes; one-time migration script for existing dirty rows.

**Tech Stack:** Python 3.14, SQLite (sqlite3), pytest, re

---

## File Map

| Action   | File                                  | Responsibility                                          |
|----------|---------------------------------------|---------------------------------------------------------|
| Create   | `src/storage/normalize_book.py`       | `BOOK_MAP` + `normalize_book(raw) -> str\|None`         |
| Modify   | `src/ingestion/ps_extractor.py`       | Numbered `_BOOKS` entries; updated `_VERSE_RE`; group refs |
| Modify   | `src/storage/sqlite_store.py`         | Call `normalize_book` in `insert_verse()`               |
| Modify   | `ingest.py`                           | Call `normalize_book` in LLM verse path (~line 126)     |
| Create   | `scripts/normalize_books.py`          | One-time migration for existing dirty rows              |
| Create   | `tests/test_normalize_book.py`        | Tests for `normalize_book()`                            |
| Modify   | `tests/test_ps_extractor.py`          | Tests for numbered-book filename parsing                |
| Modify   | `tests/test_sqlite_store.py`          | Tests for normalization wiring in `insert_verse()`      |

---

## Task 1: `normalize_book()` — test then implement

**Files:**
- Create: `tests/test_normalize_book.py`
- Create: `src/storage/normalize_book.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/test_normalize_book.py`:

```python
import pytest
from src.storage.normalize_book import normalize_book


def test_canonical_name_passthrough():
    assert normalize_book("Matthew") == "Matthew"
    assert normalize_book("Revelation") == "Revelation"
    assert normalize_book("Psalms") == "Psalms"


def test_allcaps_variants():
    assert normalize_book("HEBREWS") == "Hebrews"
    assert normalize_book("ACTS") == "Acts"
    assert normalize_book("JOHN") == "John"
    assert normalize_book("MATTHEW") == "Matthew"
    assert normalize_book("ROMANS") == "Romans"
    assert normalize_book("ISAIAH") == "Isaiah"
    assert normalize_book("EXODUS") == "Exodus"
    assert normalize_book("DANIEL") == "Daniel"
    assert normalize_book("LUKE") == "Luke"
    assert normalize_book("MARK") == "Mark"
    assert normalize_book("PSALM") == "Psalms"
    assert normalize_book("PSALMS") == "Psalms"
    assert normalize_book("COLOSSIANS") == "Colossians"
    assert normalize_book("EPHESIANS") == "Ephesians"
    assert normalize_book("PHILIPPIANS") == "Philippians"
    assert normalize_book("JEREMIAH") == "Jeremiah"
    assert normalize_book("PROVERBS") == "Proverbs"
    assert normalize_book("GENESIS") == "Genesis"
    assert normalize_book("DEUTERONOMY") == "Deuteronomy"
    assert normalize_book("JOSHUA") == "Joshua"
    assert normalize_book("JUDGES") == "Judges"
    assert normalize_book("TITUS") == "Titus"
    assert normalize_book("REVELATION") == "Revelation"
    assert normalize_book("ECCLESIASTES") == "Ecclesiastes"
    assert normalize_book("HOSEA") == "Hosea"


def test_abbreviations():
    assert normalize_book("Lk") == "Luke"
    assert normalize_book("Heb") == "Hebrews"
    assert normalize_book("Rom") == "Romans"
    assert normalize_book("Rev") == "Revelation"
    assert normalize_book("Eph") == "Ephesians"
    assert normalize_book("Col") == "Colossians"
    assert normalize_book("Ps") == "Psalms"
    assert normalize_book("Psa") == "Psalms"
    assert normalize_book("Gen") == "Genesis"
    assert normalize_book("Isa") == "Isaiah"
    assert normalize_book("Jer") == "Jeremiah"
    assert normalize_book("Prov") == "Proverbs"
    assert normalize_book("Matt") == "Matthew"
    assert normalize_book("Jn") == "John"
    assert normalize_book("Act") == "Acts"
    assert normalize_book("Exo") == "Exodus"
    assert normalize_book("Ex") == "Exodus"
    assert normalize_book("Deu") == "Deuteronomy"
    assert normalize_book("Deut") == "Deuteronomy"
    assert normalize_book("Jos") == "Joshua"
    assert normalize_book("EPH") == "Ephesians"
    assert normalize_book("COL") == "Colossians"
    assert normalize_book("JER") == "Jeremiah"


def test_numbered_books():
    assert normalize_book("1 Samuel") == "1 Samuel"
    assert normalize_book("2 Samuel") == "2 Samuel"
    assert normalize_book("1Samuel") == "1 Samuel"
    assert normalize_book("1 Kings") == "1 Kings"
    assert normalize_book("2 Kings") == "2 Kings"
    assert normalize_book("1 Chronicles") == "1 Chronicles"
    assert normalize_book("2 Chronicles") == "2 Chronicles"
    assert normalize_book("1 Corinthians") == "1 Corinthians"
    assert normalize_book("2 Corinthians") == "2 Corinthians"
    assert normalize_book("1 Thessalonians") == "1 Thessalonians"
    assert normalize_book("2 Thessalonians") == "2 Thessalonians"
    assert normalize_book("1 Timothy") == "1 Timothy"
    assert normalize_book("2 Timothy") == "2 Timothy"
    assert normalize_book("1 Peter") == "1 Peter"
    assert normalize_book("2 Peter") == "2 Peter"
    assert normalize_book("1 John") == "1 John"
    assert normalize_book("2 John") == "2 John"
    assert normalize_book("3 John") == "3 John"


def test_revelations_variant():
    assert normalize_book("REVELATIONS") == "Revelation"
    assert normalize_book("revelations") == "Revelation"


def test_garbage_returns_none():
    assert normalize_book("Jericho") is None
    assert normalize_book("jericho") is None


def test_ambiguous_unnumbered_returns_none():
    # Ambiguous books without a number prefix are not in BOOK_MAP —
    # they require chapter-based disambiguation handled by the migration.
    assert normalize_book("Samuel") is None
    assert normalize_book("Kings") is None
    assert normalize_book("Chronicles") is None
    assert normalize_book("Corinthians") is None
    assert normalize_book("Timothy") is None
    assert normalize_book("Peter") is None


def test_empty_and_none_inputs():
    assert normalize_book("") is None
    assert normalize_book(None) is None
    assert normalize_book("   ") is None
```

- [ ] **Step 1.2: Run tests — verify they all fail**

```bash
pytest tests/test_normalize_book.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'src.storage.normalize_book'`

- [ ] **Step 1.3: Implement `src/storage/normalize_book.py`**

Create `src/storage/normalize_book.py`:

```python
BOOK_MAP: dict[str, str] = {
    # ── Old Testament ─────────────────────────────────────────────────────────
    "genesis": "Genesis", "gen": "Genesis",
    "exodus": "Exodus", "exo": "Exodus", "ex": "Exodus",
    "leviticus": "Leviticus", "lev": "Leviticus",
    "numbers": "Numbers", "num": "Numbers",
    "deuteronomy": "Deuteronomy", "deut": "Deuteronomy", "deu": "Deuteronomy",
    "joshua": "Joshua", "jos": "Joshua", "josh": "Joshua",
    "judges": "Judges",
    "ruth": "Ruth",
    "1 samuel": "1 Samuel", "1samuel": "1 Samuel", "1sam": "1 Samuel",
    "2 samuel": "2 Samuel", "2samuel": "2 Samuel", "2sam": "2 Samuel",
    "1 kings": "1 Kings", "1kings": "1 Kings", "1kgs": "1 Kings",
    "2 kings": "2 Kings", "2kings": "2 Kings", "2kgs": "2 Kings",
    "1 chronicles": "1 Chronicles", "1chronicles": "1 Chronicles", "1chr": "1 Chronicles",
    "2 chronicles": "2 Chronicles", "2chronicles": "2 Chronicles", "2chr": "2 Chronicles",
    "ezra": "Ezra",
    "nehemiah": "Nehemiah", "neh": "Nehemiah",
    "esther": "Esther",
    "job": "Job",
    "psalm": "Psalms", "psalms": "Psalms", "ps": "Psalms", "psa": "Psalms",
    "proverbs": "Proverbs", "prov": "Proverbs",
    "ecclesiastes": "Ecclesiastes", "eccl": "Ecclesiastes",
    "song of songs": "Song of Songs", "song": "Song of Songs",
    "song of solomon": "Song of Songs",
    "isaiah": "Isaiah", "isa": "Isaiah",
    "jeremiah": "Jeremiah", "jer": "Jeremiah",
    "lamentations": "Lamentations", "lam": "Lamentations",
    "ezekiel": "Ezekiel", "ezek": "Ezekiel",
    "daniel": "Daniel",
    "hosea": "Hosea", "hos": "Hosea",
    "joel": "Joel",
    "amos": "Amos",
    "obadiah": "Obadiah",
    "jonah": "Jonah",
    "micah": "Micah",
    "nahum": "Nahum",
    "habakkuk": "Habakkuk",
    "zephaniah": "Zephaniah",
    "haggai": "Haggai",
    "zechariah": "Zechariah",
    "malachi": "Malachi",
    # ── New Testament ─────────────────────────────────────────────────────────
    "matthew": "Matthew", "matt": "Matthew",
    "mark": "Mark",
    "luke": "Luke", "lk": "Luke",
    "john": "John", "jn": "John",
    "acts": "Acts", "act": "Acts",
    "romans": "Romans", "rom": "Romans",
    "1 corinthians": "1 Corinthians", "1corinthians": "1 Corinthians", "1cor": "1 Corinthians",
    "2 corinthians": "2 Corinthians", "2corinthians": "2 Corinthians", "2cor": "2 Corinthians",
    "galatians": "Galatians", "gal": "Galatians",
    "ephesians": "Ephesians", "eph": "Ephesians",
    "philippians": "Philippians", "phil": "Philippians",
    "colossians": "Colossians", "col": "Colossians",
    "1 thessalonians": "1 Thessalonians", "1thessalonians": "1 Thessalonians",
    "1thess": "1 Thessalonians",
    "2 thessalonians": "2 Thessalonians", "2thessalonians": "2 Thessalonians",
    "2thess": "2 Thessalonians",
    "1 timothy": "1 Timothy", "1timothy": "1 Timothy", "1tim": "1 Timothy",
    "2 timothy": "2 Timothy", "2timothy": "2 Timothy", "2tim": "2 Timothy",
    "titus": "Titus",
    "philemon": "Philemon",
    "hebrews": "Hebrews", "heb": "Hebrews",
    "james": "James",
    "1 peter": "1 Peter", "1peter": "1 Peter", "1pet": "1 Peter",
    "2 peter": "2 Peter", "2peter": "2 Peter", "2pet": "2 Peter",
    "1 john": "1 John", "1john": "1 John", "1jn": "1 John",
    "2 john": "2 John", "2john": "2 John", "2jn": "2 John",
    "3 john": "3 John", "3john": "3 John", "3jn": "3 John",
    "jude": "Jude",
    "revelation": "Revelation", "rev": "Revelation", "revelations": "Revelation",
}

_GARBAGE: frozenset[str] = frozenset({"jericho"})


def normalize_book(raw: str) -> str | None:
    """Return canonical 66-book name, or None for garbage/unrecognized input."""
    if not raw:
        return None
    key = raw.strip().lower()
    if not key:
        return None
    if key in _GARBAGE:
        return None
    return BOOK_MAP.get(key)
```

- [ ] **Step 1.4: Run tests — verify they all pass**

```bash
pytest tests/test_normalize_book.py -v
```

Expected: all tests `PASSED`

- [ ] **Step 1.5: Commit**

```bash
git add src/storage/normalize_book.py tests/test_normalize_book.py
git commit -m "feat: add normalize_book() with canonical 66-book BOOK_MAP"
```

---

## Task 2: Fix `ps_extractor.py` for numbered book prefixes

**Files:**
- Modify: `tests/test_ps_extractor.py`
- Modify: `src/ingestion/ps_extractor.py`

- [ ] **Step 2.1: Add failing tests for numbered-book filenames**

Append to `tests/test_ps_extractor.py`:

```python
def test_numbered_prefix_1_samuel():
    verses = parse_verses_from_filename("English_2019_1-SAMUEL-9V1-10.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "1 Samuel"
    assert verses[0]["chapter"] == 9
    assert verses[0]["verse_start"] == 1
    assert verses[0]["verse_end"] == 10


def test_numbered_prefix_2_kings():
    verses = parse_verses_from_filename("English_2022_2-KINGS-4V1-7.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "2 Kings"
    assert verses[0]["chapter"] == 4


def test_numbered_prefix_1_corinthians():
    verses = parse_verses_from_filename("English_2020_LOVE-CHAPTER-1-CORINTHIANS-13V4-7.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "1 Corinthians"
    assert verses[0]["chapter"] == 13


def test_numbered_prefix_2_timothy():
    verses = parse_verses_from_filename("English_2021_EQUIP-2-TIMOTHY-3V16.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "2 Timothy"
    assert verses[0]["chapter"] == 3
    assert verses[0]["verse_start"] == 16


def test_unnumbered_book_still_works():
    # Regression: books without a prefix must still parse
    verses = parse_verses_from_filename("English_2024_FAITH-HEBREWS-11V1.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "Hebrews"
    assert verses[0]["chapter"] == 11
    assert verses[0]["verse_start"] == 1
```

- [ ] **Step 2.2: Run new tests — verify they fail**

```bash
pytest tests/test_ps_extractor.py::test_numbered_prefix_1_samuel \
       tests/test_ps_extractor.py::test_numbered_prefix_2_kings \
       tests/test_ps_extractor.py::test_numbered_prefix_1_corinthians \
       tests/test_ps_extractor.py::test_numbered_prefix_2_timothy \
       tests/test_ps_extractor.py::test_unnumbered_book_still_works -v
```

Expected: all 5 FAIL (`assert verses[0]["book"] == "1 Samuel"` fails, returning `"Samuel"`)

- [ ] **Step 2.3: Update `src/ingestion/ps_extractor.py`**

Replace the existing `_BOOKS` dict, `_BOOK_PATTERN`, `_VERSE_RE`, and `parse_verses_from_filename` with the following. The rest of the file (imports, `_strip_prefix`, `normalize_verse_ref`, `extract_ps_text`, `extract_verses_from_text`) is unchanged.

```python
# Canonical Bible book names (lowercase key → display name).
# Numbered variants allow parse_verses_from_filename to resolve prefixed books.
_BOOKS = {
    "genesis": "Genesis", "exodus": "Exodus", "leviticus": "Leviticus",
    "numbers": "Numbers", "deuteronomy": "Deuteronomy", "joshua": "Joshua",
    "judges": "Judges", "ruth": "Ruth",
    "samuel": "Samuel",
    "1samuel": "1 Samuel", "2samuel": "2 Samuel",
    "kings": "Kings",
    "1kings": "1 Kings", "2kings": "2 Kings",
    "chronicles": "Chronicles",
    "1chronicles": "1 Chronicles", "2chronicles": "2 Chronicles",
    "ezra": "Ezra", "nehemiah": "Nehemiah", "esther": "Esther", "job": "Job",
    "psalms": "Psalms", "psalm": "Psalms", "proverbs": "Proverbs",
    "ecclesiastes": "Ecclesiastes", "song": "Song of Songs",
    "isaiah": "Isaiah", "jeremiah": "Jeremiah", "lamentations": "Lamentations",
    "ezekiel": "Ezekiel", "daniel": "Daniel", "hosea": "Hosea",
    "joel": "Joel", "amos": "Amos", "obadiah": "Obadiah", "jonah": "Jonah",
    "micah": "Micah", "nahum": "Nahum", "habakkuk": "Habakkuk",
    "zephaniah": "Zephaniah", "haggai": "Haggai", "zechariah": "Zechariah",
    "malachi": "Malachi", "matthew": "Matthew", "mark": "Mark",
    "luke": "Luke", "john": "John", "acts": "Acts", "romans": "Romans",
    "corinthians": "Corinthians",
    "1corinthians": "1 Corinthians", "2corinthians": "2 Corinthians",
    "galatians": "Galatians", "ephesians": "Ephesians",
    "philippians": "Philippians", "colossians": "Colossians",
    "thessalonians": "Thessalonians",
    "1thessalonians": "1 Thessalonians", "2thessalonians": "2 Thessalonians",
    "timothy": "Timothy",
    "1timothy": "1 Timothy", "2timothy": "2 Timothy",
    "titus": "Titus", "philemon": "Philemon", "hebrews": "Hebrews",
    "james": "James",
    "peter": "Peter",
    "1peter": "1 Peter", "2peter": "2 Peter",
    "1john": "1 John", "2john": "2 John", "3john": "3 John",
    "jude": "Jude", "revelation": "Revelation",
}

# Build alternation pattern sorted longest-first to avoid partial matches
_BOOK_PATTERN = "|".join(sorted(_BOOKS.keys(), key=len, reverse=True))

# Matches: LUKE-9V23, 1-SAMUEL-9V1-10, 2-TIMOTHY-3V16, HEBREWS
# Group 1: optional numeric prefix (1, 2, or 3) with optional trailing separator
# Group 2: book name
# Group 3: chapter; Group 4: verse_start; Group 5: verse_end
_VERSE_RE = re.compile(
    rf'(?<![A-Za-z\d])([123][-_ ]?)?({_BOOK_PATTERN})'
    r'(?:[-_ ](\d{1,3})(?:V(\d{1,3})(?:-(\d{1,3}))?)?)?'
    r'(?![A-Za-z])',
    re.IGNORECASE,
)


def parse_verses_from_filename(filename: str) -> list[dict]:
    """
    Return list of verse dicts parsed from the filename.
    Each dict: {verse_ref, book, chapter, verse_start, verse_end, is_key_verse}.
    First match is the key verse (is_key_verse=1).
    """
    core = _strip_prefix(filename)
    core = re.sub(r'\d{8}', ' ', core)
    core = re.sub(r'[-_]V\d+\b', ' ', core, flags=re.IGNORECASE)

    results = []
    for i, m in enumerate(_VERSE_RE.finditer(core)):
        prefix = m.group(1).rstrip('-_ ') if m.group(1) else ""
        book_raw = m.group(2)
        book_key = (prefix + book_raw).lower()
        book = _BOOKS.get(book_key) or _BOOKS.get(book_raw.lower())
        if not book:
            continue
        chapter = int(m.group(3)) if m.group(3) else None
        verse_start = int(m.group(4)) if m.group(4) else None
        verse_end = int(m.group(5)) if m.group(5) else None
        if chapter and chapter > 150:
            continue
        ref = normalize_verse_ref(book, chapter, verse_start, verse_end)
        results.append({
            "verse_ref": ref,
            "book": book,
            "chapter": chapter,
            "verse_start": verse_start,
            "verse_end": verse_end,
            "is_key_verse": 1 if i == 0 else 0,
        })
    return results
```

- [ ] **Step 2.4: Run all ps_extractor tests — verify they all pass**

```bash
pytest tests/test_ps_extractor.py -v
```

Expected: all tests `PASSED` (existing + 5 new)

- [ ] **Step 2.5: Commit**

```bash
git add src/ingestion/ps_extractor.py tests/test_ps_extractor.py
git commit -m "feat: capture numeric book prefix in ps_extractor (1-SAMUEL → 1 Samuel)"
```

---

## Task 3: Wire `normalize_book` into `sqlite_store.py`

**Files:**
- Modify: `tests/test_sqlite_store.py`
- Modify: `src/storage/sqlite_store.py`

- [ ] **Step 3.1: Add failing tests for normalization in `insert_verse`**

Append to `tests/test_sqlite_store.py`:

```python
def _sermon(reg, sermon_id="s1"):
    reg.upsert_sermon({
        "sermon_id": sermon_id, "date": "2024-01-06", "year": 2024,
        "language": "English", "speaker": None, "topic": None, "theme": None,
        "summary": None, "key_verse": None, "ng_file": "f.pdf",
        "ps_file": None, "status": "grouped",
    })


def test_insert_verse_normalizes_allcaps_book(reg):
    _sermon(reg)
    reg.insert_verse({
        "sermon_id": "s1", "verse_ref": "HEBREWS 11:1",
        "book": "HEBREWS", "chapter": 11, "verse_start": 1,
        "verse_end": None, "is_key_verse": 1,
    })
    with sqlite3.connect(reg.db_path) as conn:
        row = conn.execute(
            "SELECT book, verse_ref FROM verses WHERE sermon_id = 's1'"
        ).fetchone()
    assert row[0] == "Hebrews"
    assert row[1] == "Hebrews 11:1"


def test_insert_verse_normalizes_abbreviation(reg):
    _sermon(reg)
    reg.insert_verse({
        "sermon_id": "s1", "verse_ref": "Lk 9:23",
        "book": "Lk", "chapter": 9, "verse_start": 23,
        "verse_end": None, "is_key_verse": 1,
    })
    with sqlite3.connect(reg.db_path) as conn:
        row = conn.execute(
            "SELECT book, verse_ref FROM verses WHERE sermon_id = 's1'"
        ).fetchone()
    assert row[0] == "Luke"
    assert row[1] == "Luke 9:23"


def test_insert_verse_skips_garbage(reg):
    _sermon(reg)
    reg.insert_verse({
        "sermon_id": "s1", "verse_ref": "Jericho 1:1",
        "book": "Jericho", "chapter": 1, "verse_start": 1,
        "verse_end": None, "is_key_verse": 0,
    })
    with sqlite3.connect(reg.db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM verses WHERE sermon_id = 's1'"
        ).fetchone()[0]
    assert count == 0
```

- [ ] **Step 3.2: Run new tests — verify they fail**

```bash
pytest tests/test_sqlite_store.py::test_insert_verse_normalizes_allcaps_book \
       tests/test_sqlite_store.py::test_insert_verse_normalizes_abbreviation \
       tests/test_sqlite_store.py::test_insert_verse_skips_garbage -v
```

Expected: all 3 FAIL (`assert row[0] == "Hebrews"` fails with `"HEBREWS"`)

- [ ] **Step 3.3: Update `src/storage/sqlite_store.py`**

Add the import and update `insert_verse`. The rest of the file is unchanged.

At the top of `sqlite_store.py`, add the import after the existing `normalize_speaker` import:

```python
from src.storage.normalize_book import normalize_book
from src.ingestion.ps_extractor import normalize_verse_ref
```

Replace the existing `insert_verse` method:

```python
def insert_verse(self, record: dict):
    record = dict(record)
    canonical = normalize_book(record.get("book"))
    if canonical is None:
        return  # garbage or unrecognized book — skip silently
    record["book"] = canonical
    record["verse_ref"] = normalize_verse_ref(
        canonical,
        record.get("chapter"),
        record.get("verse_start"),
        record.get("verse_end"),
    )
    cols = ", ".join(record.keys())
    placeholders = ", ".join(["?"] * len(record))
    with sqlite3.connect(self.db_path) as conn:
        conn.execute(
            f"INSERT OR IGNORE INTO verses ({cols}) VALUES ({placeholders})",
            list(record.values()),
        )
```

- [ ] **Step 3.4: Run all sqlite_store tests — verify they all pass**

```bash
pytest tests/test_sqlite_store.py -v
```

Expected: all tests `PASSED`

- [ ] **Step 3.5: Run full test suite — check for regressions**

```bash
pytest tests/ -v --ignore=tests/__pycache__ -x
```

Expected: all tests `PASSED`

- [ ] **Step 3.6: Commit**

```bash
git add src/storage/sqlite_store.py tests/test_sqlite_store.py
git commit -m "feat: normalize book names in insert_verse; skip garbage rows"
```

---

## Task 4: Wire `normalize_book` into the LLM verse path in `ingest.py`

**Files:**
- Modify: `ingest.py` (lines ~1–20 imports; lines ~122–131 LLM verse block)

- [ ] **Step 4.1: Add the import**

At the top of `ingest.py`, alongside the existing imports, add:

```python
from src.storage.normalize_book import normalize_book
```

- [ ] **Step 4.2: Apply normalization in the LLM verse block**

Find this block (currently around line 122–131):

```python
    # LLM verse extraction from PS text (if text available and no filename verses)
    if ps_text_combined.strip() and not all_verses:
        llm_verse_refs = extract_verses_from_text(ps_text_combined, llm)
        for ref in llm_verse_refs:
            m = re.match(r'^(\w+(?:\s\w+)?)\s+(\d+)(?::(\d+)(?:-(\d+))?)?$', ref)
            if m:
                all_verses.append({
                    "verse_ref": ref, "book": m.group(1),
                    "chapter": int(m.group(2)),
                    "verse_start": int(m.group(3)) if m.group(3) else None,
                    "verse_end": int(m.group(4)) if m.group(4) else None,
                    "is_key_verse": 0,
                })
        if all_verses:
            all_verses[0]["is_key_verse"] = 1
```

Replace with:

```python
    # LLM verse extraction from PS text (if text available and no filename verses)
    if ps_text_combined.strip() and not all_verses:
        llm_verse_refs = extract_verses_from_text(ps_text_combined, llm)
        for ref in llm_verse_refs:
            m = re.match(r'^(\w+(?:\s\w+)?)\s+(\d+)(?::(\d+)(?:-(\d+))?)?$', ref)
            if m:
                canonical_book = normalize_book(m.group(1))
                if canonical_book is None:
                    continue
                all_verses.append({
                    "verse_ref": ref, "book": canonical_book,
                    "chapter": int(m.group(2)),
                    "verse_start": int(m.group(3)) if m.group(3) else None,
                    "verse_end": int(m.group(4)) if m.group(4) else None,
                    "is_key_verse": 0,
                })
        if all_verses:
            all_verses[0]["is_key_verse"] = 1
```

- [ ] **Step 4.3: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/__pycache__ -x
```

Expected: all tests `PASSED`

- [ ] **Step 4.4: Commit**

```bash
git add ingest.py
git commit -m "feat: normalize LLM-extracted book names before storing verse"
```

---

## Task 5: One-time migration script

**Files:**
- Create: `scripts/normalize_books.py`

- [ ] **Step 5.1: Create `scripts/normalize_books.py`**

```python
#!/usr/bin/env python3
"""One-time migration: normalize book names in the verses table.

Usage:
    python scripts/normalize_books.py [--db data/sermons.db] [--dry-run]
"""
import argparse
import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.normalize_book import normalize_book

# Ambiguous unnumbered books: key → (book1, book2, max_ch_book1, max_ch_book2)
# book1 is the default when the chapter is ambiguous.
_AMBIGUOUS: dict[str, tuple] = {
    "samuel":      ("1 Samuel",      "2 Samuel",      31, 24),
    "kings":       ("1 Kings",       "2 Kings",       22, 25),
    "chronicles":  ("1 Chronicles",  "2 Chronicles",  29, 36),
    "corinthians": ("1 Corinthians", "2 Corinthians", 16, 13),
    "timothy":     ("1 Timothy",     "2 Timothy",     6,  4),
    "peter":       ("1 Peter",       "2 Peter",       5,  3),
}


def _disambiguate(key: str, chapter) -> str:
    book1, book2, max1, max2 = _AMBIGUOUS[key]
    if chapter is None:
        return book1
    ch = int(chapter)
    if ch > max1 and ch > max2:
        return book1  # invalid chapter number — default
    if ch > max1:
        return book2  # exceeds book1's chapter count → must be book2
    if ch > max2:
        return book1  # exceeds book2's chapter count → must be book1
    return book1      # ambiguous overlap — default to book1


def _build_verse_ref(book: str, chapter, verse_start, verse_end) -> str:
    if chapter is None:
        return book
    if verse_start is None:
        return f"{book} {chapter}"
    if verse_end is not None:
        return f"{book} {chapter}:{verse_start}-{verse_end}"
    return f"{book} {chapter}:{verse_start}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/sermons.db")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without modifying the DB")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, sermon_id, verse_ref, book, chapter, "
        "verse_start, verse_end, is_key_verse FROM verses"
    ).fetchall()
    conn.row_factory = None

    updates: dict[int, tuple[str, str]] = {}   # row_id → (new_book, new_verse_ref)
    garbage: set[int] = set()                  # row_ids to delete (unrecognized book)
    unresolved: list[str] = []

    for row in rows:
        raw = row["book"]
        chapter = row["chapter"]

        canonical = normalize_book(raw)
        if canonical is None:
            key = raw.strip().lower() if raw else ""
            if key in _AMBIGUOUS:
                canonical = _disambiguate(key, chapter)
            else:
                garbage.add(row["id"])
                unresolved.append(raw or "(null)")
                continue

        new_ref = _build_verse_ref(
            canonical, row["chapter"], row["verse_start"], row["verse_end"]
        )
        if canonical != raw or new_ref != row["verse_ref"]:
            updates[row["id"]] = (canonical, new_ref)

    # Resolve post-normalization duplicates within the same sermon.
    # For each (sermon_id, new_verse_ref), keep is_key_verse=1 row; else lowest id.
    row_by_id: dict[int, sqlite3.Row] = {row["id"]: row for row in rows}

    def _new_key(row_id: int) -> tuple:
        if row_id in updates:
            new_book, new_ref = updates[row_id]
            return (row_by_id[row_id]["sermon_id"], new_ref)
        r = row_by_id[row_id]
        return (r["sermon_id"], r["verse_ref"])

    groups: dict[tuple, list[int]] = {}
    for row in rows:
        if row["id"] in garbage:
            continue
        k = _new_key(row["id"])
        groups.setdefault(k, []).append(row["id"])

    duplicates: set[int] = set()
    for ids in groups.values():
        if len(ids) <= 1:
            continue
        # Sort: is_key_verse=1 first, then lowest id first
        ids.sort(key=lambda rid: (-(row_by_id[rid]["is_key_verse"] or 0), rid))
        for loser in ids[1:]:
            duplicates.add(loser)
            updates.pop(loser, None)

    deletes = garbage | duplicates

    if args.dry_run:
        print(f"[DRY RUN] Would update: {len(updates)} rows")
        print(f"[DRY RUN] Would delete (garbage): {len(garbage)} rows")
        print(f"[DRY RUN] Would delete (duplicates): {len(duplicates)} rows")
        if unresolved:
            print(f"[DRY RUN] Unresolved books: {sorted(set(unresolved))}")
        conn.close()
        return

    with conn:
        for row_id, (new_book, new_ref) in updates.items():
            conn.execute(
                "UPDATE verses SET book = ?, verse_ref = ? WHERE id = ?",
                (new_book, new_ref, row_id),
            )
        for row_id in deletes:
            conn.execute("DELETE FROM verses WHERE id = ?", (row_id,))

    conn.close()

    print(f"Updated : {len(updates)} rows")
    print(f"Deleted (garbage)   : {len(garbage)} rows")
    print(f"Deleted (duplicates): {len(duplicates)} rows")
    if unresolved:
        uniq = sorted(set(unresolved))
        print(f"Unresolved books (manually inspect): {uniq}")
    else:
        print("No unresolved books.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Dry-run first**

```bash
python scripts/normalize_books.py --dry-run
```

Expected output (approximate):
```
[DRY RUN] Would update: ~400 rows
[DRY RUN] Would delete (garbage): ~1 rows
[DRY RUN] Would delete (duplicates): ~0 rows
[DRY RUN] Unresolved books: []
```

Verify the numbers look reasonable. If `Unresolved books` lists unexpected entries, add them to `BOOK_MAP` in `normalize_book.py` before continuing.

- [ ] **Step 5.3: Run the migration**

```bash
python scripts/normalize_books.py
```

Expected output:
```
Updated : <N> rows
Deleted (garbage)   : 1 rows
Deleted (duplicates): <M> rows
No unresolved books.
```

- [ ] **Step 5.4: Verify the migration**

```bash
sqlite3 data/sermons.db "SELECT DISTINCT book FROM verses ORDER BY book;"
```

Expected: only clean canonical names (e.g., `Acts`, `Genesis`, `1 Samuel`, `Hebrews`, `Luke`, `Psalms`, `Revelation` etc.) — no ALL_CAPS variants, no abbreviations, no `Jericho`.

```bash
sqlite3 data/sermons.db "SELECT book, COUNT(*) FROM verses GROUP BY book ORDER BY COUNT(*) DESC LIMIT 20;"
```

Expected: top entries are canonical book names only.

- [ ] **Step 5.5: Re-run gap analysis query to confirm it's fixed**

```bash
sqlite3 data/sermons.db "SELECT DISTINCT book FROM verses ORDER BY book;" | sort
```

Manually verify that books like `1 Samuel`, `2 Samuel`, `1 Kings`, `2 Corinthians`, `1 John` now appear (or confirm they're genuinely absent from the archive).

- [ ] **Step 5.6: Commit**

```bash
git add scripts/normalize_books.py
git commit -m "feat: one-time migration to normalize existing verse book names"
```

---

## Final verification

- [ ] **Run full test suite**

```bash
pytest tests/ -v --ignore=tests/__pycache__
```

Expected: all tests `PASSED`
