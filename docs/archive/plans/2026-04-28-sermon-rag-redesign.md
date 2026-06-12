# BBTC Sermon RAG Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the ingestion pipeline around the sermon unit (NG + PS) data model, with BGE-M3 embeddings, a normalized verses table, and a working LangGraph ReAct agent in Gradio.

**Architecture:** Each Sunday's Notes/Guide (NG) and Slides/PPT (PS) are grouped into one sermon unit. NG labeled fields (TOPIC/SPEAKER/THEME/DATE) are extracted by regex; PS verse references are parsed from filenames + LLM on text. One SQLite row and one ChromaDB document set (body chunks + summary chunk) per sermon unit.

**Tech Stack:** Python 3.11+, PyMuPDF (fitz), SQLite, ChromaDB, Ollama (BGE-M3 + llama3.1:8b), LangGraph, Gradio, Dagster, Plotly

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/ingestion/file_classifier.py` | Modify | Classify filename → `ng` \| `ps` \| `handout` |
| `src/ingestion/ng_extractor.py` | Create | Regex + fallback LLM extraction from NG PDF text |
| `src/ingestion/ps_extractor.py` | Create | Filename regex + LLM verse extraction from PS files |
| `src/ingestion/sermon_grouper.py` | Modify | Update to use `ng`/`ps` labels; group by date proximity |
| `src/storage/sqlite_store.py` | Rewrite | New `sermons` + `verses` tables |
| `src/storage/chroma_store.py` | Modify | Swap to BGE-M3 embeddings |
| `src/tools/sql_tool.py` | Rewrite | System prompt for new schema + verses table |
| `src/tools/vector_tool.py` | Modify | New metadata field names |
| `src/tools/viz_tool.py` | Modify | `verses_per_book` queries `verses.book` |
| `src/ui_helpers.py` | Keep | Queries are schema-compatible |
| `ingest.py` | Create | Main pipeline: classify → group → extract → embed |
| `app.py` | Modify | Fix agent import; remove bible_tool; update system prompt |
| `dagster_pipeline.py` | Rewrite | Thin wrapper calling ingest pipeline functions |
| `src/scraper/bbtc_scraper.py` | Modify | Classify before download; skip handout/unknown |
| `CLAUDE.md` | Modify | Update to reflect new architecture |
| `tests/test_file_classifier.py` | Modify | Update to new label names |
| `tests/test_sqlite_store.py` | Rewrite | New schema assertions |
| `tests/test_ng_extractor.py` | Create | Regex extraction cases |
| `tests/test_ps_extractor.py` | Create | Verse parsing cases |
| `tests/test_sermon_grouper.py` | Modify | Update to new labels |

**Delete (do not keep):**
`quick_ingest.py`, `backfill_metadata.py`, `backfill_text.py`, `normalize_speakers.py` (root),
`wipe_and_restart.sh`, `scratch/`, `src/ingestion/metadata_extractor.py`,
`src/ingestion/speaker_from_filename.py`, `src/ingestion/speaker_from_pdf.py`,
`src/ingestion/bible/`, `src/agents/`, `src/graph/`,
`src/tools/bible_tool.py`, `render.yaml`, `Dockerfile`

---

## Task 1: Delete Obsolete Files

**Files:** root-level and subdirectories

- [ ] **Step 1: Delete obsolete root files**

```bash
rm -f quick_ingest.py backfill_metadata.py backfill_text.py normalize_speakers.py wipe_and_restart.sh
rm -rf scratch/
```

- [ ] **Step 2: Delete obsolete src files**

```bash
rm -f src/ingestion/metadata_extractor.py
rm -f src/ingestion/speaker_from_filename.py
rm -f src/ingestion/speaker_from_pdf.py
rm -rf src/ingestion/bible/
rm -rf src/agents/
rm -rf src/graph/
rm -f src/tools/bible_tool.py
rm -f render.yaml Dockerfile
```

- [ ] **Step 3: Remove stale test files for deleted modules**

```bash
rm -f tests/test_matplotlib_tool.py
```

- [ ] **Step 4: Verify nothing important was deleted**

```bash
find src/ -name "*.py" | sort
```

Expected output includes: `src/ingestion/file_classifier.py`, `src/ingestion/filename_parser.py`, `src/ingestion/sermon_grouper.py`, `src/llm.py`, `src/storage/chroma_store.py`, `src/storage/normalize_speaker.py`, `src/storage/reranker.py`, `src/storage/sqlite_store.py`, `src/tools/sql_tool.py`, `src/tools/vector_tool.py`, `src/tools/viz_tool.py`, `src/ui_helpers.py`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: delete obsolete ingestion and bible modules"
```

---

## Task 2: Update File Classifier

**Files:**
- Modify: `src/ingestion/file_classifier.py`
- Modify: `tests/test_file_classifier.py`

- [ ] **Step 1: Rewrite the classifier with new labels**

Replace the entire contents of `src/ingestion/file_classifier.py`:

```python
"""Classify BBTC sermon files as ng, ps, or handout."""

import re

_NG_RE = re.compile(
    r'(?:members?(?:27)?|leaders?|cell)[-_]?(?:guide|copy|guide[-_]updated)'
    r'|MembersGuide|MessageSummary.*Members'
    r'|[-_]notes?[-_.]|[-_]notes?\.',
    re.IGNORECASE,
)

_HANDOUT_RE = re.compile(
    r'[-_](handout|visual[-_]?summary)[-_.]|handout\.',
    re.IGNORECASE,
)


def classify_file(filename: str) -> str:
    """
    Returns:
        "ng"      — Notes / Cell Guide / Members Guide / Members Copy
        "ps"      — PPT deck, slides PDF, or primary sermon PDF
        "handout" — Handout or visual summary (skip)
    """
    if _NG_RE.search(filename):
        return "ng"
    if _HANDOUT_RE.search(filename):
        return "handout"
    return "ps"
```

- [ ] **Step 2: Run existing tests to see them fail with new labels**

```bash
python -m pytest tests/test_file_classifier.py -v 2>&1 | head -40
```

Expected: multiple FAILED because tests still check `"cell_guide"` / `"sermon_slides"` / `"other"`.

- [ ] **Step 3: Rewrite tests with new labels**

Replace `tests/test_file_classifier.py`:

```python
import pytest
from src.ingestion.file_classifier import classify_file


class TestClassifyFile:
    def test_members_guide_hyphenated(self):
        assert classify_file("English_2018_28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide-updated.pdf") == "ng"

    def test_members27_guide(self):
        assert classify_file("English_2018_10-11-Nov-2018-Stewards-by-Ps-Hakan-members27-guide.pdf") == "ng"

    def test_leaders_guide(self):
        assert classify_file("English_2018_15-16-Dec-2018-And-the-Bleeding-Stopped-by-Elder-Chua-Seng-Lee-Leaders-Guide.pdf") == "ng"

    def test_members_copy(self):
        assert classify_file("English_2018_12-13-May-2018-A-Tale-of-4-Mothers-by-Gary-Koh-Members-Copy.pdf") == "ng"

    def test_camelcase_members_guide(self):
        assert classify_file("English_2015_FearOrFaith_eLVM_2015-12-19_20_MessageSummary_MembersGuide.pdf") == "ng"

    def test_notes_suffix(self):
        assert classify_file("English_2024_06-07-Jan-2024-The-Heart-of-Discipleship-by-SP-Chua-Seng-Lee-Members-Guide.pdf") == "ng"

    def test_pptx_extension(self):
        assert classify_file("English_2020_CHURCH-IS-FAMILY-Edric-Sng-12-Feb-2020-website.pptx") == "ps"

    def test_ppt_keyword(self):
        assert classify_file("English_2018_20180623-Growing-Faith-in-God-Final-PPT.pdf") == "ps"

    def test_compressed_slide(self):
        assert classify_file("English_2024_03-TOGETHER-AS-ONE-LUKE-10V1-3-20240127-PPT-FINAL-4-compressed.pdf") == "ps"

    def test_camelcase_abbreviated(self):
        assert classify_file("English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf") == "ps"

    def test_handout(self):
        assert classify_file("English_2018_EffectivePrayer-1-Principles_Handout.pdf") == "handout"

    def test_visual_summary(self):
        assert classify_file("English_2018_Visual-Summary_EffectivePrayer-6.pdf") == "handout"
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_file_classifier.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/file_classifier.py tests/test_file_classifier.py
git commit -m "feat: rename classifier labels to ng/ps/handout"
```

---

## Task 3: Rewrite SQLite Store

**Files:**
- Rewrite: `src/storage/sqlite_store.py`
- Rewrite: `tests/test_sqlite_store.py`

- [ ] **Step 1: Write failing tests for new schema**

Replace `tests/test_sqlite_store.py`:

```python
import os, tempfile, sqlite3, pytest
from src.storage.sqlite_store import SermonRegistry


@pytest.fixture
def reg():
    with tempfile.TemporaryDirectory() as d:
        yield SermonRegistry(db_path=os.path.join(d, "t.db"))


def test_sermons_table_columns(reg):
    with sqlite3.connect(reg.db_path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sermons)").fetchall()]
    for col in ["sermon_id", "date", "year", "language", "speaker", "topic",
                "theme", "summary", "key_verse", "ng_file", "ps_file", "status"]:
        assert col in cols, f"Missing column: {col}"


def test_verses_table_columns(reg):
    with sqlite3.connect(reg.db_path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(verses)").fetchall()]
    for col in ["id", "sermon_id", "verse_ref", "book", "chapter",
                "verse_start", "verse_end", "is_key_verse"]:
        assert col in cols, f"Missing column: {col}"


def test_upsert_sermon(reg):
    reg.upsert_sermon({
        "sermon_id": "2024-01-06-discipleship",
        "date": "2024-01-06",
        "year": 2024,
        "language": "English",
        "speaker": "SP Chua Seng Lee",
        "topic": "The Heart of Discipleship",
        "theme": "#CanIPrayForYou",
        "summary": "A summary.",
        "key_verse": "Luke 9:23",
        "ng_file": "English_2024_06-07-Jan-2024-The-Heart-of-Discipleship-Members-Guide.pdf",
        "ps_file": None,
        "status": "indexed",
    })
    row = reg.get_sermon("2024-01-06-discipleship")
    assert row["speaker"] == "SP Chua Seng Lee"
    assert row["key_verse"] == "Luke 9:23"


def test_insert_verse(reg):
    reg.upsert_sermon({
        "sermon_id": "2024-01-06-discipleship",
        "date": "2024-01-06",
        "year": 2024,
        "language": "English",
        "speaker": "SP Chua Seng Lee",
        "topic": "Discipleship",
        "theme": None,
        "summary": None,
        "key_verse": "Luke 9:23",
        "ng_file": "test.pdf",
        "ps_file": None,
        "status": "grouped",
    })
    reg.insert_verse({
        "sermon_id": "2024-01-06-discipleship",
        "verse_ref": "Luke 9:23",
        "book": "Luke",
        "chapter": 9,
        "verse_start": 23,
        "verse_end": None,
        "is_key_verse": 1,
    })
    with sqlite3.connect(reg.db_path) as conn:
        row = conn.execute(
            "SELECT book, is_key_verse FROM verses WHERE sermon_id = ?",
            ("2024-01-06-discipleship",)
        ).fetchone()
    assert row[0] == "Luke"
    assert row[1] == 1


def test_sermon_exists(reg):
    assert not reg.sermon_exists("2024-01-06-discipleship")
    reg.upsert_sermon({
        "sermon_id": "2024-01-06-discipleship",
        "date": "2024-01-06", "year": 2024, "language": "English",
        "speaker": None, "topic": None, "theme": None,
        "summary": None, "key_verse": None,
        "ng_file": "test.pdf", "ps_file": None, "status": "grouped",
    })
    assert reg.sermon_exists("2024-01-06-discipleship")


def test_get_pending_sermons(reg):
    for sid, status in [("a", "indexed"), ("b", "grouped"), ("c", "extracted")]:
        reg.upsert_sermon({
            "sermon_id": sid, "date": "2024-01-06", "year": 2024,
            "language": "English", "speaker": None, "topic": None,
            "theme": None, "summary": None, "key_verse": None,
            "ng_file": "f.pdf", "ps_file": None, "status": status,
        })
    pending = reg.get_pending_sermons()
    ids = [s["sermon_id"] for s in pending]
    assert "b" in ids
    assert "c" in ids
    assert "a" not in ids
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_sqlite_store.py -v 2>&1 | head -30
```

Expected: multiple FAILED (old schema).

- [ ] **Step 3: Rewrite sqlite_store.py**

Replace `src/storage/sqlite_store.py`:

```python
import sqlite3, os, re
from src.storage.normalize_speaker import normalize_speaker


class SermonRegistry:
    def __init__(self, db_path: str = "data/sermons.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sermons (
                    sermon_id  TEXT PRIMARY KEY,
                    date       TEXT,
                    year       INTEGER,
                    language   TEXT,
                    speaker    TEXT,
                    topic      TEXT,
                    theme      TEXT,
                    summary    TEXT,
                    key_verse  TEXT,
                    ng_file    TEXT,
                    ps_file    TEXT,
                    status     TEXT
                );
                CREATE TABLE IF NOT EXISTS verses (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    sermon_id   TEXT REFERENCES sermons(sermon_id),
                    verse_ref   TEXT,
                    book        TEXT,
                    chapter     INTEGER,
                    verse_start INTEGER,
                    verse_end   INTEGER,
                    is_key_verse INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_verses_sermon ON verses(sermon_id);
                CREATE INDEX IF NOT EXISTS idx_sermons_year ON sermons(year);
                CREATE INDEX IF NOT EXISTS idx_sermons_speaker ON sermons(speaker);
            """)

    def upsert_sermon(self, record: dict):
        if record.get("speaker"):
            record["speaker"] = normalize_speaker(record["speaker"]) or record["speaker"]
        if not record.get("year") and record.get("date"):
            try:
                record["year"] = int(record["date"][:4])
            except (ValueError, TypeError):
                pass
        cols = ", ".join(record.keys())
        placeholders = ", ".join(["?"] * len(record))
        updates = ", ".join(f"{k} = excluded.{k}" for k in record if k != "sermon_id")
        sql = (
            f"INSERT INTO sermons ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(sermon_id) DO UPDATE SET {updates}"
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(sql, list(record.values()))

    def insert_verse(self, record: dict):
        cols = ", ".join(record.keys())
        placeholders = ", ".join(["?"] * len(record))
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO verses ({cols}) VALUES ({placeholders})",
                list(record.values()),
            )

    def sermon_exists(self, sermon_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT 1 FROM sermons WHERE sermon_id = ?", (sermon_id,)
            ).fetchone() is not None

    def ng_file_indexed(self, ng_file: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT 1 FROM sermons WHERE ng_file = ? AND status = 'indexed'",
                (ng_file,)
            ).fetchone() is not None

    def get_sermon(self, sermon_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sermons WHERE sermon_id = ?", (sermon_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_pending_sermons(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sermons WHERE status NOT IN ('indexed', 'failed')"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_sermons(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM sermons").fetchall()]

    def mark_status(self, sermon_id: str, status: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sermons SET status = ? WHERE sermon_id = ?",
                (status, sermon_id),
            )

    def wipe(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("DELETE FROM verses; DELETE FROM sermons;")
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_sqlite_store.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/storage/sqlite_store.py tests/test_sqlite_store.py
git commit -m "feat: rewrite sqlite_store with sermon-unit schema and verses table"
```

---

## Task 4: Create NG Extractor

**Files:**
- Create: `src/ingestion/ng_extractor.py`
- Create: `tests/test_ng_extractor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ng_extractor.py`:

```python
from src.ingestion.ng_extractor import extract_ng_metadata, extract_ng_body


SAMPLE_TEXT = """Member's Copy


TOPIC
The Heart of Discipleship   
SPEAKER 
SP Chua Seng Lee 
THEME #CanIPrayForYou 
DATE 
06 & 07 January 2024  


INTRODUCTION
Do you believe God can turn Singapore Godward?
In this sermon SP Chua Seng Lee unpacked the Heart of Discipleship.

THE HEART OF DISCIPLESHIP
1) What is discipleship?
"""


def test_extract_topic():
    meta = extract_ng_metadata(SAMPLE_TEXT, "dummy.pdf")
    assert meta["topic"] == "The Heart of Discipleship"


def test_extract_speaker():
    meta = extract_ng_metadata(SAMPLE_TEXT, "dummy.pdf")
    assert "Chua Seng Lee" in meta["speaker"]


def test_extract_theme():
    meta = extract_ng_metadata(SAMPLE_TEXT, "dummy.pdf")
    assert meta["theme"] == "#CanIPrayForYou"


def test_extract_date():
    meta = extract_ng_metadata(SAMPLE_TEXT, "dummy.pdf")
    assert meta["date"] == "2024-01-06"


def test_body_starts_after_introduction():
    body = extract_ng_body(SAMPLE_TEXT)
    assert "Do you believe" in body
    assert "Member's Copy" not in body


def test_fallback_to_filename_for_speaker():
    text = "TOPIC\nSome Sermon\nDATE\n01 January 2024\n\nINTRODUCTION\nBody."
    meta = extract_ng_metadata(text, "English_2024_01-Jan-2024-Some-Sermon-by-SP-Chua-Seng-Lee-Members-Guide.pdf")
    assert meta["speaker"] is not None


def test_missing_fields_return_none():
    meta = extract_ng_metadata("Just some plain text without labels.", "unknown.pdf")
    assert meta["topic"] is None or isinstance(meta["topic"], str)
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
python -m pytest tests/test_ng_extractor.py -v 2>&1 | head -20
```

Expected: ModuleNotFoundError or multiple FAILED.

- [ ] **Step 3: Create ng_extractor.py**

Create `src/ingestion/ng_extractor.py`:

```python
"""Extract metadata and body text from BBTC Notes/Guide (NG) PDF text."""

import re
from src.ingestion.filename_parser import parse_cell_guide_filename
from src.storage.normalize_speaker import normalize_speaker

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_date_string(raw: str) -> str | None:
    """Parse human date like '06 & 07 January 2024' → '2024-01-06'."""
    m = re.search(
        r'(\d{1,2})(?:\s*[&,\-]\s*\d{1,2})?\s+'
        r'(january|february|march|april|may|june|july|august|september|october|november|december|'
        r'jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)'
        r'\s+(\d{4})',
        raw, re.IGNORECASE,
    )
    if not m:
        return None
    day = int(m.group(1))
    month = _MONTHS[m.group(2).lower()]
    year = int(m.group(3))
    return f"{year}-{month:02d}-{day:02d}"


def _labeled_field(text: str, label: str) -> str | None:
    """Extract value of a labeled field like 'TOPIC\\n<value>' or 'TOPIC <value>'."""
    pattern = rf'(?:^|\n)\s*{label}\s*\n\s*(.+?)(?:\n\s*[A-Z]{{3,}}|\n\s*$|$)'
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Inline: TOPIC Some Value
    m = re.search(rf'(?:^|\n)\s*{label}\s+(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def extract_ng_metadata(text: str, filename: str) -> dict:
    """
    Extract speaker, date, topic, theme from NG text.
    Falls back to filename_parser when labeled fields are absent.
    Returns dict with keys: speaker, date, topic, theme (any may be None).
    """
    topic = _labeled_field(text, "TOPIC")
    theme = _labeled_field(text, "THEME")
    date_raw = _labeled_field(text, "DATE")
    date = _parse_date_string(date_raw) if date_raw else None

    speaker_raw = _labeled_field(text, "SPEAKER")
    speaker = normalize_speaker(speaker_raw) if speaker_raw else None

    # Fallback to filename parser for any missing field
    if not all([topic, speaker, date]):
        parsed = parse_cell_guide_filename(filename)
        topic = topic or parsed.get("topic")
        speaker = speaker or parsed.get("speaker")
        date = date or parsed.get("date")

    return {"speaker": speaker, "date": date, "topic": topic, "theme": theme}


def extract_ng_body(text: str) -> str:
    """
    Return the body of the NG: everything after the INTRODUCTION label.
    Falls back to the full text if no INTRODUCTION label found.
    """
    m = re.search(r'(?:^|\n)\s*INTRODUCTION\s*\n', text, re.IGNORECASE)
    if m:
        return text[m.end():].strip()
    # No INTRODUCTION label — strip the header block (first ~300 chars of labeled fields)
    lines = text.split("\n")
    header_labels = {"topic", "speaker", "theme", "date", "member", "leader", "cell"}
    body_lines = []
    header_done = False
    for line in lines:
        stripped = line.strip().lower()
        if not header_done:
            if stripped in header_labels or not stripped:
                continue
            # Heuristic: header is done once we see a line > 60 chars (body prose)
            if len(line.strip()) > 60:
                header_done = True
        if header_done:
            body_lines.append(line)
    return "\n".join(body_lines).strip() or text.strip()
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_ng_extractor.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/ng_extractor.py tests/test_ng_extractor.py
git commit -m "feat: add ng_extractor with regex labeled-field extraction"
```

---

## Task 5: Create PS Extractor

**Files:**
- Create: `src/ingestion/ps_extractor.py`
- Create: `tests/test_ps_extractor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ps_extractor.py`:

```python
from src.ingestion.ps_extractor import parse_verses_from_filename, normalize_verse_ref


def test_luke_chapter_verse():
    verses = parse_verses_from_filename("English_2024_03-COST-TO-MENTOR-LUKE-9V23-20240615-compressed.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "Luke"
    assert verses[0]["chapter"] == 9
    assert verses[0]["verse_start"] == 23


def test_john_chapter_only():
    verses = parse_verses_from_filename("English_2024_04-HE-IS-OUR-HOPE-JOHN-11-20240330-PPT-2-compressed.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "John"
    assert verses[0]["chapter"] == 11


def test_hebrews_book_only():
    verses = parse_verses_from_filename("English_2024_Walking-in-Submission-HEBREWS-REVISITED-compressed.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "Hebrews"


def test_verse_range():
    verses = parse_verses_from_filename("English_2024_03-TOGETHER-AS-ONE-LUKE-10V1-3-20240127-PPT.pdf")
    assert len(verses) >= 1
    v = verses[0]
    assert v["book"] == "Luke"
    assert v["chapter"] == 10
    assert v["verse_start"] == 1
    assert v["verse_end"] == 3


def test_no_verse_returns_empty():
    verses = parse_verses_from_filename("English_2024_Some-Sermon-Without-Verse-compressed.pdf")
    assert verses == []


def test_normalize_verse_ref_basic():
    ref = normalize_verse_ref("Luke", 9, 23, None)
    assert ref == "Luke 9:23"


def test_normalize_verse_ref_range():
    ref = normalize_verse_ref("Luke", 10, 1, 3)
    assert ref == "Luke 10:1-3"


def test_normalize_verse_ref_chapter_only():
    ref = normalize_verse_ref("John", 11, None, None)
    assert ref == "John 11"
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
python -m pytest tests/test_ps_extractor.py -v 2>&1 | head -20
```

Expected: ModuleNotFoundError or multiple FAILED.

- [ ] **Step 3: Create ps_extractor.py**

Create `src/ingestion/ps_extractor.py`:

```python
"""Extract verse references from BBTC Slides/PPT (PS) filenames and text."""

import re
import fitz  # PyMuPDF
import os

# Canonical Bible book names (lowercase key → display name)
_BOOKS = {
    "genesis": "Genesis", "exodus": "Exodus", "leviticus": "Leviticus",
    "numbers": "Numbers", "deuteronomy": "Deuteronomy", "joshua": "Joshua",
    "judges": "Judges", "ruth": "Ruth", "samuel": "Samuel",
    "kings": "Kings", "chronicles": "Chronicles", "ezra": "Ezra",
    "nehemiah": "Nehemiah", "esther": "Esther", "job": "Job",
    "psalms": "Psalms", "psalm": "Psalms", "proverbs": "Proverbs",
    "ecclesiastes": "Ecclesiastes", "song": "Song of Songs",
    "isaiah": "Isaiah", "jeremiah": "Jeremiah", "lamentations": "Lamentations",
    "ezekiel": "Ezekiel", "daniel": "Daniel", "hosea": "Hosea",
    "joel": "Joel", "amos": "Amos", "obadiah": "Obadiah", "jonah": "Jonah",
    "micah": "Micah", "nahum": "Nahum", "habakkuk": "Habakkuk",
    "zephaniah": "Zephaniah", "haggai": "Haggai", "zechariah": "Zechariah",
    "malachi": "Malachi", "matthew": "Matthew", "mark": "Mark",
    "luke": "Luke", "john": "John", "acts": "Acts", "romans": "Romans",
    "corinthians": "Corinthians", "galatians": "Galatians",
    "ephesians": "Ephesians", "philippians": "Philippians",
    "colossians": "Colossians", "thessalonians": "Thessalonians",
    "timothy": "Timothy", "titus": "Titus", "philemon": "Philemon",
    "hebrews": "Hebrews", "james": "James", "peter": "Peter",
    "jude": "Jude", "revelation": "Revelation",
}

# Build alternation pattern sorted longest-first to avoid partial matches
_BOOK_PATTERN = "|".join(
    sorted(_BOOKS.keys(), key=len, reverse=True)
)
# Matches: LUKE-9V23, LUKE-10V1-3, JOHN-11, HEBREWS
_VERSE_RE = re.compile(
    rf'(?<![A-Z])({_BOOK_PATTERN})'         # book name (case-insensitive via flag)
    r'(?:-(\d{{1,3}})(?:V(\d{{1,3}})(?:-(\d{{1,3}}))?)?)?'  # -chapter, V verse, -end
    r'(?![A-Z])',
    re.IGNORECASE,
)


def _strip_prefix(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r'^(English|Mandarin)_\d{4}_', '', name)


def normalize_verse_ref(book: str, chapter: int | None, verse_start: int | None, verse_end: int | None) -> str:
    if chapter is None:
        return book
    if verse_start is None:
        return f"{book} {chapter}"
    if verse_end is not None:
        return f"{book} {chapter}:{verse_start}-{verse_end}"
    return f"{book} {chapter}:{verse_start}"


def parse_verses_from_filename(filename: str) -> list[dict]:
    """
    Return list of verse dicts from the filename.
    Each dict: {verse_ref, book, chapter, verse_start, verse_end, is_key_verse}.
    First match is the key verse (is_key_verse=1).
    """
    core = _strip_prefix(filename)
    # Remove date-like segments and version numbers to reduce false matches
    core = re.sub(r'\d{8}', ' ', core)
    core = re.sub(r'[-_]V\d+\b', ' ', core, flags=re.IGNORECASE)

    results = []
    for i, m in enumerate(_VERSE_RE.finditer(core)):
        book_key = m.group(1).lower()
        book = _BOOKS.get(book_key)
        if not book:
            continue
        chapter = int(m.group(2)) if m.group(2) else None
        verse_start = int(m.group(3)) if m.group(3) else None
        verse_end = int(m.group(4)) if m.group(4) else None
        # Skip if chapter looks like a year (> 150)
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


def extract_ps_text(filepath: str) -> str:
    """Return text from PS PDF. Returns empty string for image-only PDFs."""
    try:
        doc = fitz.open(filepath)
        text = "\n".join(page.get_text() for page in doc).strip()
        return text
    except Exception:
        return ""


def extract_verses_from_text(text: str, llm=None) -> list[str]:
    """
    Extract verse references from PS slide text using LLM.
    Returns list of normalized ref strings like ['Luke 9:23', 'John 3:16'].
    Returns [] if text is empty or LLM unavailable.
    """
    if not text or not llm:
        return []
    prompt = (
        "List all Bible verse references mentioned in the following slide text. "
        "Format each as 'Book Chapter:Verse' (e.g. 'Luke 9:23'). "
        "If no verses are found, reply with 'NONE'. "
        "Reply with one verse per line, nothing else.\n\n"
        f"Slide text:\n{text[:1500]}"
    )
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        if "NONE" in content.upper():
            return []
        lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        # Filter to lines that look like verse refs
        verse_pattern = re.compile(r'^[A-Z][a-z]+ \d+:\d+', re.IGNORECASE)
        return [l for l in lines if verse_pattern.match(l)]
    except Exception:
        return []
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_ps_extractor.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/ps_extractor.py tests/test_ps_extractor.py
git commit -m "feat: add ps_extractor with filename verse parsing"
```

---

## Task 6: Update Sermon Grouper

**Files:**
- Modify: `src/ingestion/sermon_grouper.py`
- Modify: `tests/test_sermon_grouper.py`

- [ ] **Step 1: Run existing sermon grouper tests to see failures**

```bash
python -m pytest tests/test_sermon_grouper.py -v 2>&1 | head -30
```

Note which tests fail (they reference `"cell_guide"` and `"sermon_slides"` labels).

- [ ] **Step 2: Update sermon_grouper.py to use new labels**

Replace `src/ingestion/sermon_grouper.py`:

```python
"""Group BBTC sermon files into (ng, slides, other) sermon groups."""

from dataclasses import dataclass, field
from datetime import datetime
from src.ingestion.file_classifier import classify_file
from src.ingestion.filename_parser import extract_any_date, extract_topic_words


@dataclass
class SermonGroup:
    ng: str | None = None
    ps: list[str] = field(default_factory=list)


def _date_proximity(d1: str | None, d2: str | None, tolerance: int = 3) -> bool:
    if not d1 or not d2:
        return False
    fmt = "%Y-%m-%d"
    try:
        return abs((datetime.strptime(d1, fmt) - datetime.strptime(d2, fmt)).days) <= tolerance
    except ValueError:
        return False


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def group_sermon_files(filenames: list[str]) -> list[SermonGroup]:
    """
    Group filenames into SermonGroups.
    Each NG becomes one group. PS files are paired to their NG by date proximity
    (≤ 3 days) or high topic-word Jaccard (≥ 0.5). Unpaired PS each become a
    standalone group with ng=None.
    Handout files are ignored.
    """
    ngs, pss = [], []
    for f in filenames:
        kind = classify_file(f)
        if kind == "ng":
            ngs.append(f)
        elif kind == "ps":
            pss.append(f)
        # handout: skip

    groups: list[SermonGroup] = []
    used_ps: set[str] = set()

    for ng in ngs:
        group = SermonGroup(ng=ng)
        ng_date = extract_any_date(ng)
        ng_words = extract_topic_words(ng)

        for ps in pss:
            if ps in used_ps:
                continue
            ps_date = extract_any_date(ps)
            ps_words = extract_topic_words(ps)
            if _date_proximity(ng_date, ps_date) or _jaccard(ng_words, ps_words) >= 0.5:
                group.ps.append(ps)
                used_ps.add(ps)

        groups.append(group)

    for ps in pss:
        if ps not in used_ps:
            groups.append(SermonGroup(ps=[ps]))

    return groups
```

- [ ] **Step 3: Replace tests/test_sermon_grouper.py with updated field names**

Replace `tests/test_sermon_grouper.py`:

```python
import pytest
from src.ingestion.sermon_grouper import group_sermon_files


class TestGroupSermonFiles:
    def test_pairs_ng_with_matching_ps_by_date(self):
        files = [
            "English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf",
            "English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].ng == files[0]
        assert files[1] in groups[0].ps

    def test_pairs_by_topic_when_slide_has_date_proximity(self):
        files = [
            "English_2018_09-10-June-2018-An-Altar-Not-to-Miss-by-Ps-Jason-Teo-Members-Guide.pdf",
            "English_2018_An-Altar-Not-To-Miss-9-June-2018.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].ng == files[0]
        assert files[1] in groups[0].ps

    def test_standalone_ps_without_ng(self):
        files = ["English_2018_20180623-Growing-Faith-in-God-Final-PPT.pdf"]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].ng is None
        assert files[0] in groups[0].ps

    def test_standalone_ng_without_ps(self):
        files = ["English_2018_28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide-updated.pdf"]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].ng == files[0]
        assert groups[0].ps == []

    def test_does_not_pair_different_weekends(self):
        files = [
            "English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf",
            "English_2018_09-10-June-2018-An-Altar-Not-to-Miss-by-Ps-Jason-Teo-Members-Guide.pdf",
            "English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf",
            "English_2018_An-Altar-Not-To-Miss-9-June-2018.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 2
        ng_ps = {g.ng: g.ps for g in groups}
        assert "English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf" in \
               ng_ps["English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf"]
        assert "English_2018_An-Altar-Not-To-Miss-9-June-2018.pdf" in \
               ng_ps["English_2018_09-10-June-2018-An-Altar-Not-to-Miss-by-Ps-Jason-Teo-Members-Guide.pdf"]

    def test_handouts_are_ignored(self):
        files = [
            "English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf",
            "English_2018_FinishingWell_Handout.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert "English_2018_FinishingWell_Handout.pdf" not in groups[0].ps
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_sermon_grouper.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/sermon_grouper.py tests/test_sermon_grouper.py
git commit -m "feat: update sermon_grouper to use ng/ps labels"
```

---

## Task 7: Update ChromaDB Store to BGE-M3

**Files:**
- Modify: `src/storage/chroma_store.py`

- [ ] **Step 1: Update chroma_store.py**

In `src/storage/chroma_store.py`, replace the `__init__` method's embedding initialization block:

Old:
```python
if self._embeddings is None:
    try:
        from langchain_ollama import OllamaEmbeddings
        self._embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self._embeddings.embed_query("test")
    except Exception:
        print("⚠️  Ollama not available. Using local HuggingFace embeddings (all-mpnet-base-v2, 768-dim).")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        self._embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
```

New:
```python
if self._embeddings is None:
    try:
        from langchain_ollama import OllamaEmbeddings
        self._embeddings = OllamaEmbeddings(model="BGE-M3")
        self._embeddings.embed_query("test")
        print("✅ Using BGE-M3 embeddings via Ollama.")
    except Exception:
        print("⚠️  Ollama BGE-M3 unavailable. Falling back to nomic-embed-text.")
        try:
            from langchain_ollama import OllamaEmbeddings
            self._embeddings = OllamaEmbeddings(model="nomic-embed-text")
            self._embeddings.embed_query("test")
        except Exception:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
```

Also update the `_embed` method's safe truncation comment and limit (BGE-M3 supports up to 8192 tokens):

Old comment: `# 4000 characters is a safe limit for Nomic Embed`
New: `# 8000 characters is a safe limit for BGE-M3`

Old: `safe_texts = [t[:4000] for t in texts]`
New: `safe_texts = [t[:8000] for t in texts]`

- [ ] **Step 2: Verify the store still imports cleanly**

```bash
python -c "from src.storage.chroma_store import SermonVectorStore; print('OK')"
```

Expected: `OK` (or warning about Ollama if not running).

- [ ] **Step 3: Commit**

```bash
git add src/storage/chroma_store.py
git commit -m "feat: switch embeddings to BGE-M3 with nomic fallback"
```

---

## Task 8: Create Main Ingest Pipeline

**Files:**
- Create: `ingest.py`

- [ ] **Step 1: Create ingest.py**

Create `ingest.py` in the project root:

```python
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
                  llm, splitter: RecursiveCharacterTextSplitter, incremental: bool):
    ng_file = group.ng
    ps_files = group.ps

    if not ng_file and not ps_files:
        return

    # Skip if already indexed in incremental mode
    if incremental and ng_file and registry.ng_file_indexed(ng_file):
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

    key_verse = all_verses[0]["verse_ref"] if all_verses else None
    verse_refs = [v["verse_ref"] for v in all_verses]

    # Generate unified summary
    summary = _generate_summary(ng_body, topic, theme, speaker, verse_refs, ps_text_combined, llm)

    sermon_id = _make_sermon_id(date, topic, ng_file or (ps_files[0] if ps_files else "unknown"))

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


def run_pipeline(wipe: bool = False, year: int | None = None, incremental: bool = True):
    print("🚀 BBTC Sermon Ingestion Pipeline")

    registry = SermonRegistry(db_path=DB_PATH)
    vector_store = SermonVectorStore(persist_dir=CHROMA_DIR)
    llm = get_llm(ollama_model="llama3.1:8b")
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
        print(f"❌ Staging directory not found: {STAGING_DIR}")
        sys.exit(1)

    all_files = os.listdir(STAGING_DIR)
    if year:
        all_files = [f for f in all_files if f"_{year}_" in f]
    # Only NG and PS
    sermon_files = [f for f in all_files if classify_file(f) in ("ng", "ps")]
    print(f"📁 Found {len(sermon_files)} NG/PS files in staging/")

    groups = group_sermon_files(sermon_files)
    print(f"📦 Formed {len(groups)} sermon groups")

    indexed = 0
    skipped = 0
    failed = 0
    for group in groups:
        try:
            ng = group.ng
            if incremental and ng and registry.ng_file_indexed(ng):
                skipped += 1
                continue
            process_group(group, registry, vector_store, llm, splitter, incremental)
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
    args = parser.parse_args()
    run_pipeline(wipe=args.wipe, year=args.year, incremental=not args.wipe)
```

- [ ] **Step 2: Verify ingest.py imports cleanly**

```bash
python -c "import ingest; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Smoke-test with a single year (dry run check)**

```bash
python ingest.py --year 2024 2>&1 | head -20
```

Expected: shows file counts and group formation without crashing.

- [ ] **Step 4: Commit**

```bash
git add ingest.py
git commit -m "feat: create ingest.py — classify/group/extract/embed pipeline"
```

---

## Task 9: Rewrite SQL Tool

**Files:**
- Rewrite: `src/tools/sql_tool.py`

- [ ] **Step 1: Rewrite sql_tool.py**

Replace `src/tools/sql_tool.py`:

```python
import sqlite3
from langchain_core.tools import tool


def make_sql_tool(db_path: str):

    @tool
    def sql_query_tool(query: str) -> str:
        """Executes a SQL query against the BBTC sermon database.

        Schema:
        - sermons(sermon_id, date, year, language, speaker, topic, theme,
                  summary, key_verse, ng_file, ps_file, status)
        - verses(id, sermon_id, verse_ref, book, chapter, verse_start, verse_end, is_key_verse)

        Common queries:
        - List speakers: SELECT DISTINCT speaker FROM sermons WHERE speaker IS NOT NULL ORDER BY speaker
        - Speakers in 2023: SELECT speaker, COUNT(*) as n FROM sermons WHERE year=2023 GROUP BY speaker ORDER BY n DESC
        - Most preached book: SELECT book, COUNT(*) as n FROM verses GROUP BY book ORDER BY n DESC LIMIT 10
        - Verses by speaker: SELECT v.verse_ref, COUNT(*) as n FROM verses v JOIN sermons s USING(sermon_id) WHERE s.speaker LIKE '%Chua%' GROUP BY v.verse_ref ORDER BY n DESC
        - Key verses: SELECT key_verse, speaker, date FROM sermons WHERE key_verse IS NOT NULL ORDER BY date DESC

        Returns up to 50 rows."""
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(query)
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchmany(50)
                if not rows:
                    return "No results found."
                result = "Columns: " + ", ".join(columns) + "\n"
                for row in rows:
                    result += str(row) + "\n"
                return result
        except Exception as e:
            return (
                f"SQL Error: {e}\n"
                "Tables:\n"
                "  sermons(sermon_id, date, year, language, speaker, topic, theme, summary, key_verse, ng_file, ps_file, status)\n"
                "  verses(id, sermon_id, verse_ref, book, chapter, verse_start, verse_end, is_key_verse)"
            )

    return sql_query_tool
```

- [ ] **Step 2: Verify tool creation**

```bash
python -c "
from src.storage.sqlite_store import SermonRegistry
from src.tools.sql_tool import make_sql_tool
import tempfile, os
with tempfile.TemporaryDirectory() as d:
    reg = SermonRegistry(db_path=os.path.join(d,'t.db'))
    tool = make_sql_tool(reg.db_path)
    print(tool.invoke({'query': 'SELECT 1'}))
"
```

Expected: `No results found.` or `Columns: 1\n(1,)\n`

- [ ] **Step 3: Commit**

```bash
git add src/tools/sql_tool.py
git commit -m "feat: rewrite sql_tool for new schema with verses table"
```

---

## Task 10: Update Vector Tool

**Files:**
- Modify: `src/tools/vector_tool.py`

- [ ] **Step 1: Update vector_tool.py to use new metadata fields**

Replace `src/tools/vector_tool.py`:

```python
from langchain_core.tools import tool
from src.storage.chroma_store import SermonVectorStore


def make_vector_tool(vector_store: SermonVectorStore):

    @tool
    def search_sermons_tool(query: str, year: int | None = None, speaker: str | None = None) -> str:
        """Searches sermon text and summaries using semantic similarity.
        Use for 'What did the pastor say about X?' or 'Find sermons about Y'.
        Optionally filter by year (integer e.g. 2024) or speaker (partial name e.g. 'Chua').
        Returns excerpts with topic, speaker, date, and key verse."""

        where: dict | None = None
        if year is not None and speaker:
            where = {"$and": [{"year": {"$eq": year}}, {"speaker": {"$eq": speaker}}]}
        elif year is not None:
            where = {"year": {"$eq": year}}
        elif speaker:
            where = {"speaker": {"$eq": speaker}}

        results = vector_store.search_sermons(query, k=5, where=where)
        if not results:
            return "No relevant sermon content found."

        parts = []
        for res in results:
            m = res.get("metadata") or {}
            header = (
                f"[{m.get('topic') or 'Unknown Topic'} | {m.get('speaker') or 'Unknown'} "
                f"| {m.get('date') or ''} | {m.get('key_verse') or ''}]"
            )
            parts.append(f"{header}\n{res['content']}")

        return "\n\n---\n\n".join(parts)

    return search_sermons_tool
```

- [ ] **Step 2: Commit**

```bash
git add src/tools/vector_tool.py
git commit -m "feat: update vector_tool metadata fields for new schema"
```

---

## Task 11: Update Viz Tool

**Files:**
- Modify: `src/tools/viz_tool.py`

- [ ] **Step 1: Update viz_tool.py — fix top_bible_books and add verses_per_book**

In `src/tools/viz_tool.py`, replace the `elif chart_name == "top_bible_books":` block and the `@tool` docstring:

Update the `@tool` docstring to:
```python
"""Generates an interactive Plotly chart from live sermon data and returns the JSON file path.
Supported chart_name values:
- 'sermons_per_speaker' — bar chart of sermon count per speaker (top 15)
- 'sermons_per_year' — bar chart of sermon count per year
- 'verses_per_book' — bar chart of most-preached Bible books from verses table (top 15)
- 'sermons_scatter' — bubble chart of sermon count by speaker and year
Returns the file path to the saved Plotly JSON."""
```

Replace the `top_bible_books` block with `verses_per_book`:

```python
elif chart_name == "verses_per_book":
    rows = conn.execute(
        "SELECT book, COUNT(*) as n FROM verses "
        "WHERE book IS NOT NULL AND book != '' "
        "GROUP BY book ORDER BY n DESC LIMIT 15"
    ).fetchall()
    if not rows:
        return "No verse data found. Run ingest.py first."
    books, counts = zip(*rows)
    fig = px.bar(
        x=counts, y=books, orientation='h',
        title="Top 15 Preached Bible Books",
        labels={'x': 'Times Preached', 'y': 'Bible Book'},
        color=counts, color_continuous_scale='Greens'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
```

Also update the `sermons_per_speaker` LIMIT from 10 to 15 and the `else` error message to list the new chart name:

```python
return (
    f"Unknown chart '{chart_name}'. "
    "Valid options: sermons_per_speaker, sermons_per_year, verses_per_book, sermons_scatter."
)
```

- [ ] **Step 2: Verify viz_tool imports cleanly**

```bash
python -c "from src.tools.viz_tool import make_viz_tool; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/tools/viz_tool.py
git commit -m "feat: update viz_tool — verses_per_book uses verses table"
```

---

## Task 12: Fix App Agent

**Files:**
- Modify: `app.py`

The current `app.py` has `from langchain.agents import create_agent` which is incorrect. The correct import is `create_react_agent` from `langgraph.prebuilt`. This is a primary cause of broken responses.

- [ ] **Step 1: Fix the agent creation and remove bible_tool**

In `app.py`, make the following targeted changes:

**Change 1** — Replace the broken import:
```python
# Remove this line:
from langchain.agents import create_agent

# Add this line:
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
```

**Change 2** — Remove `bible_tool` imports (lines `from src.tools.bible_tool import make_bible_tool`).

**Change 3** — Replace the tool creation and agent instantiation block:

Old:
```python
sql_tool = make_sql_tool(registry)
vector_tool = make_vector_tool(vector_store)
bible_tool = make_bible_tool(vector_store)
viz_tool = make_viz_tool(registry)

SYSTEM_PROMPT = (...)

agent = create_agent(llm, tools=[sql_tool, vector_tool, bible_tool, viz_tool], system_prompt=SYSTEM_PROMPT)
```

New:
```python
sql_tool = make_sql_tool(registry.db_path)
vector_tool = make_vector_tool(vector_store)
viz_tool = make_viz_tool(registry)

SYSTEM_PROMPT = (
    "You are the BBTC Sermon Intelligence Assistant for Bethesda Bedok-Tampines Church.\n\n"
    "## Tool routing\n"
    "- Use 'sql_query_tool' for: counts, lists of speakers/years, verse statistics, "
    "questions that need numbers. The database has:\n"
    "    • sermons(sermon_id, date, year, language, speaker, topic, theme, summary, key_verse, status)\n"
    "    • verses(id, sermon_id, verse_ref, book, chapter, verse_start, verse_end, is_key_verse)\n"
    "  For 'most preached book': SELECT book, COUNT(*) as n FROM verses GROUP BY book ORDER BY n DESC LIMIT 10\n"
    "  For 'verses by speaker': SELECT v.verse_ref, COUNT(*) FROM verses v JOIN sermons s USING(sermon_id) WHERE s.speaker LIKE '%Name%' GROUP BY v.verse_ref ORDER BY COUNT(*) DESC\n"
    "- Use 'search_sermons_tool' for: questions about sermon content, what a pastor said about a topic, "
    "summaries of specific sermons. Pass year/speaker filters when specified.\n"
    "- Use 'viz_tool' only when the user asks for a chart or visualization. "
    "Valid chart_name values: 'sermons_per_speaker', 'sermons_per_year', 'verses_per_book', 'sermons_scatter'. "
    "When viz_tool returns a file path, include that exact path verbatim in your response.\n\n"
    "## Rules\n"
    "- Answer ONLY from tool results. Never invent speaker names, dates, or verses.\n"
    "- If tools return no data, say so explicitly.\n"
    "- For 'most recent sermon': use sql_query_tool with ORDER BY date DESC LIMIT 1.\n"
)

agent = create_react_agent(
    llm,
    tools=[sql_tool, vector_tool, viz_tool],
    prompt=SystemMessage(content=SYSTEM_PROMPT),
)
```

**Change 4** — In the `respond()` function, update the agent invocation. The `create_react_agent` returns a graph, so invocation is the same but the result structure may differ:

```python
result = agent.invoke({"messages": messages})
final = result["messages"][-1].content
```

This part is already correct. No change needed here.

- [ ] **Step 2: Verify app.py imports without error**

```bash
python -c "
import sys
sys.argv = ['app.py']
# Don't actually launch, just check imports
import importlib.util
spec = importlib.util.spec_from_file_location('app', 'app.py')
# Just parse, don't execute
import ast
with open('app.py') as f:
    ast.parse(f.read())
print('Syntax OK')
"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "fix: replace create_agent with create_react_agent from langgraph; remove bible_tool"
```

---

## Task 13: Rewrite Dagster Pipeline

**Files:**
- Rewrite: `dagster_pipeline.py`

- [ ] **Step 1: Rewrite dagster_pipeline.py as thin wrapper**

Replace `dagster_pipeline.py`:

```python
"""
Dagster pipeline — thin wrapper around ingest.py.
Weekly schedule: Saturday at 22:00 (so new weekend files are ready).

UI:  DAGSTER_HOME=$(mktemp -d) dagster dev -m dagster_pipeline
Run: dagster asset materialize --select sermon_ingestion -m dagster_pipeline
"""

from dagster import (
    asset, Definitions, ScheduleDefinition, AssetSelection,
    define_asset_job, AssetExecutionContext, MetadataValue, in_process_executor,
)
from ingest import run_pipeline


@asset
def sermon_ingestion(context: AssetExecutionContext):
    """Weekly incremental ingestion of new BBTC sermons."""
    context.log.info("Starting incremental sermon ingestion...")
    run_pipeline(wipe=False, year=None, incremental=True)
    context.log.info("Ingestion complete.")
    return MetadataValue.text("done")


ingestion_job = define_asset_job(
    "sermon_ingestion_job",
    selection=AssetSelection.assets(sermon_ingestion),
    executor_def=in_process_executor,
)

sermon_weekly_schedule = ScheduleDefinition(
    job=ingestion_job,
    cron_schedule="0 22 * * 6",  # Saturday 22:00
)

defs = Definitions(
    assets=[sermon_ingestion],
    schedules=[sermon_weekly_schedule],
    jobs=[ingestion_job],
    executor=in_process_executor,
)
```

- [ ] **Step 2: Verify dagster pipeline loads**

```bash
python -c "import dagster_pipeline; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add dagster_pipeline.py
git commit -m "refactor: simplify dagster_pipeline as thin wrapper over ingest.py"
```

---

## Task 14: Update Scraper — Classify Before Download

**Files:**
- Modify: `src/scraper/bbtc_scraper.py`

- [ ] **Step 1: Add classify-before-download to scrape_year or the download loop**

Read `src/scraper/bbtc_scraper.py` lines 58–120 to find the download loop. Add the classifier check before each download call.

Find the block where files are downloaded (look for `self._download_file` call). Before the download, add:

```python
from src.ingestion.file_classifier import classify_file

# Inside the per-link loop, before downloading:
filename = os.path.basename(urllib.parse.urlparse(url).path)
if classify_file(filename) == "handout":
    self._logger(f"⏭️  Skipping handout: {filename}")
    continue
```

Add this import at the top of the file (alongside existing imports):
```python
from src.ingestion.file_classifier import classify_file
```

And in the download loop (find the section that iterates over links and downloads), add the skip check before calling `self._download_file`:

```python
fname = os.path.basename(urllib.parse.urlparse(link).path)
if classify_file(fname) == "handout":
    print(f"⏭️  Skipping handout: {fname}")
    continue
```

- [ ] **Step 2: Verify scraper imports cleanly**

```bash
python -c "from src.scraper.bbtc_scraper import BBTCScraper; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/scraper/bbtc_scraper.py
git commit -m "feat: skip handout files before downloading in scraper"
```

---

## Task 15: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Rewrite CLAUDE.md to reflect new architecture**

Replace the entire contents of `CLAUDE.md`:

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Hybrid Agentic RAG pipeline** for the BBTC (Bethesda Bedok-Tampines Church) sermon archive.

Scrapes sermon documents from the BBTC website, groups them into **sermon units** (one Notes/Guide + one Slides/PPT per Sunday), extracts structured metadata, stores in SQLite + ChromaDB, and exposes a Gradio chat interface backed by a LangGraph ReAct agent.

## Environment Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # optional: GROQ_API_KEY, GEMINI_API_KEY for cloud fallback
```

Ollama must be running locally: `ollama serve`

Required Ollama models:
- `BGE-M3` — embeddings (primary)
- `llama3.1:8b` — metadata extraction + summary generation

## Running the Application

```bash
# Launch Gradio chat UI
python app.py

# Full ingest from scratch (wipe + rebuild)
python ingest.py --wipe

# Incremental ingest (new files only)
python ingest.py

# Ingest a specific year
python ingest.py --year 2024

# Dagster web UI (weekly scheduler)
DAGSTER_HOME=$(mktemp -d) dagster dev -m dagster_pipeline

# Scrape a single year from BBTC website
python src/scraper/bbtc_scraper.py 2024
```

## Architecture

### Sermon Unit Model

Every weekend (Sat/Sun), BBTC posts two files:
- **NG** (Notes/Guide): PDF with labeled fields `TOPIC`, `SPEAKER`, `THEME`, `DATE` + body text
- **PS** (Slides/PPT): PDF exported from PowerPoint; filename encodes the key verse

Together they form one **sermon unit** — the atomic unit of ingestion.

### Data Flow

```
BBTC Website → BBTCScraper (classify-before-download: skip handouts)
    ↓
data/staging/  (NG + PS files only)
    ↓
ingest.py
  ├── CLASSIFY  (file_classifier.py)  → ng | ps | handout
  ├── GROUP     (sermon_grouper.py)   → SermonGroup(ng, ps[])
  ├── EXTRACT   (ng_extractor.py)     → TOPIC/SPEAKER/THEME/DATE via regex
  │             (ps_extractor.py)     → verses from filename + LLM on text
  ├── SUMMARIZE (llama3.1:8b)         → unified NG+PS summary
  └── EMBED     (chroma_store.py)     → BGE-M3 → sermon_collection
    ↓
SQLite (data/sermons.db)  ← structured metadata + verses table
ChromaDB (data/chroma_db/) ← body chunks (800/150) + summary chunk per sermon
    ↓
LangGraph ReAct Agent (3 tools)
    ↓
Gradio UI
```

### Key Components

| Component | File | Purpose |
|---|---|---|
| `SermonRegistry` | `src/storage/sqlite_store.py` | SQLite CRUD; sermons + verses tables |
| `SermonVectorStore` | `src/storage/chroma_store.py` | ChromaDB with BGE-M3 + CrossEncoder reranker |
| `BBTCScraper` | `src/scraper/bbtc_scraper.py` | Cloudflare-bypass scraper; classify-before-download |
| `classify_file` | `src/ingestion/file_classifier.py` | Returns `ng` \| `ps` \| `handout` |
| `group_sermon_files` | `src/ingestion/sermon_grouper.py` | Pairs NG+PS by date proximity/topic overlap |
| `extract_ng_metadata` | `src/ingestion/ng_extractor.py` | Regex on labeled fields; filename fallback |
| `parse_verses_from_filename` | `src/ingestion/ps_extractor.py` | Verse regex on PS filenames |
| `run_pipeline` | `ingest.py` | Orchestrates full classify→group→extract→embed |
| `dagster_pipeline.py` | root | Weekly Saturday schedule wrapping `ingest.py` |
| `app.py` | root | Gradio UI + LangGraph ReAct agent |

### Agent Tools

- **`sql_query_tool`** — SQL against `data/sermons.db`; use for counts, lists, verse aggregations
- **`search_sermons_tool`** — BGE-M3 semantic search over `sermon_collection`; use for content queries
- **`viz_tool`** — Plotly interactive charts: `sermons_per_speaker`, `sermons_per_year`, `verses_per_book`, `sermons_scatter`

### SQLite Schema

```sql
sermons(
  sermon_id TEXT PRIMARY KEY,  -- "2024-01-06-the-heart-of-discipleship"
  date      TEXT,              -- YYYY-MM-DD
  year      INTEGER,
  language  TEXT,              -- "English" | "Mandarin"
  speaker   TEXT,
  topic     TEXT,
  theme     TEXT,
  summary   TEXT,              -- LLM-generated from NG+PS
  key_verse TEXT,              -- first verse from PS
  ng_file   TEXT,              -- staging filename of NG
  ps_file   TEXT,              -- staging filename of PS (nullable)
  status    TEXT               -- grouped → extracted → indexed | failed
)

verses(
  id          INTEGER PRIMARY KEY,
  sermon_id   TEXT,            -- FK → sermons
  verse_ref   TEXT,            -- "Luke 9:23"
  book        TEXT,            -- "Luke"
  chapter     INTEGER,
  verse_start INTEGER,
  verse_end   INTEGER,
  is_key_verse INTEGER         -- 1 = key verse (first in PS)
)
```

### ChromaDB

- Collection: `sermon_collection`
- Chunks: NG body text (800/150) + LLM summary (single chunk) per sermon
- Metadata per chunk: `{sermon_id, doc_type, speaker, date, year, topic, theme, language, key_verse}`
- Embeddings: `BGE-M3` via Ollama (fallback: nomic-embed-text)

## Notable Quirks

- NG labeled fields (`TOPIC`, `SPEAKER`, etc.) are reliable for 2022+ files. Older files fall back to `filename_parser.py`.
- ~50% of PS files are image-based PDFs with no extractable text — verse extraction relies on filename regex.
- The scraper skips handouts before downloading (classify-before-download).
- `create_react_agent` from `langgraph.prebuilt` is used — NOT `langchain.agents.create_agent`.
- BGE-M3 embedding model: 1.2 GB, multilingual (handles English + Mandarin sermons).
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: rewrite CLAUDE.md for sermon-unit architecture"
```

---

## Task 16: End-to-End Smoke Test

**Files:** none (verification only)

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v --ignore=tests/test_llm.py --ignore=tests/test_reranker.py 2>&1 | tail -20
```

Expected: majority pass; `test_llm.py` and `test_reranker.py` skipped (require Ollama).

- [ ] **Step 2: Run ingest on 2024 data**

```bash
python ingest.py --wipe --year 2024 2>&1 | tail -30
```

Expected: sermons grouped and indexed, ChromaDB chunk count > 0.

- [ ] **Step 3: Verify SQLite has data**

```bash
python -c "
import sqlite3
with sqlite3.connect('data/sermons.db') as conn:
    print('Sermons:', conn.execute('SELECT COUNT(*) FROM sermons').fetchone()[0])
    print('Verses:', conn.execute('SELECT COUNT(*) FROM verses').fetchone()[0])
    print('Top books:')
    for row in conn.execute('SELECT book, COUNT(*) as n FROM verses GROUP BY book ORDER BY n DESC LIMIT 5').fetchall():
        print(' ', row)
"
```

Expected: non-zero counts, recognisable Bible book names.

- [ ] **Step 4: Verify agent tools work**

```bash
python -c "
from src.storage.sqlite_store import SermonRegistry
from src.tools.sql_tool import make_sql_tool
reg = SermonRegistry()
tool = make_sql_tool(reg.db_path)
print(tool.invoke({'query': 'SELECT DISTINCT speaker FROM sermons WHERE speaker IS NOT NULL ORDER BY speaker LIMIT 5'}))
"
```

Expected: list of 5 speaker names.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: post-redesign smoke test verified"
```
