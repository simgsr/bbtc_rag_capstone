# Sermon File Grouping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treat each sermon weekend as one logical unit — metadata from the cell guide, sermon content from the PPT/slide file — instead of two disconnected records.

**Architecture:** A file classifier labels each scraped file as `cell_guide`, `sermon_slides`, or `other`. A filename parser extracts speaker, date, and topic directly from the cell guide filename (no LLM needed). A grouper pairs each cell guide with its slide file by date proximity and topic word overlap. The Dagster pipeline is updated to process sermon groups: the cell guide becomes the canonical SQLite record and both files' text is vectorised under the same `sermon_id`. A one-time backfill script applies these rules to all existing DB records.

**Tech Stack:** Python 3.12+, SQLite (sqlite3), ChromaDB, pytest, existing `src/storage/normalize_speaker.py` + `src/ingestion/speaker_from_filename.py`.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/ingestion/filename_parser.py` | Parse date, topic, speaker from BBTC filenames |
| Create | `src/ingestion/file_classifier.py` | Label each file: cell_guide / sermon_slides / other |
| Create | `src/ingestion/sermon_grouper.py` | Pair cell guide + slide files into sermon groups |
| Create | `tests/test_filename_parser.py` | Unit tests for filename parser |
| Create | `tests/test_file_classifier.py` | Unit tests for classifier |
| Create | `tests/test_sermon_grouper.py` | Unit tests for grouper |
| Create | `backfill_metadata.py` | One-time script: fix speaker/date/topic for all existing records |
| Modify | `src/storage/sqlite_store.py` | Add `topic TEXT` column, handle in `insert_sermon` |
| Modify | `dagster_pipeline.py` | Replace per-file loop with sermon-group loop |

---

## Task 1: `src/ingestion/filename_parser.py`

**Files:**
- Create: `src/ingestion/filename_parser.py`
- Test: `tests/test_filename_parser.py`

BBTC uses two filename conventions:

**Convention 1 – long hyphenated** (mostly 2016 onwards):
`English_2018_28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide-updated.pdf`

**Convention 2 – CamelCase with ISO date** (mostly 2015):
`English_2015_FearOrFaith_eLVM_2015-12-19_20_MessageSummary_MembersGuide.pdf`

- [ ] **Step 1: Write failing tests**

Create `tests/test_filename_parser.py`:

```python
import pytest
from src.ingestion.filename_parser import parse_cell_guide_filename, extract_any_date, extract_topic_words


class TestParseCellGuideFilename:
    def test_long_hyphenated_with_by_elder(self):
        r = parse_cell_guide_filename(
            "English_2018_28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide-updated.pdf"
        )
        assert r["date"] == "2018-07-28"
        assert r["topic"] == "Know Your Enemy"
        assert r["speaker"] == "Ps Edric Sng"

    def test_long_hyphenated_without_by(self):
        r = parse_cell_guide_filename(
            "English_2018_06-07-July-2018-Effective-Prayer-Part-5-SP-Daniel-Foo-Members-Guide.pdf"
        )
        assert r["date"] == "2018-07-06"
        assert r["topic"] == "Effective Prayer Part 5"
        assert r["speaker"] == "SP Daniel Foo"

    def test_long_hyphenated_ps_speaker(self):
        r = parse_cell_guide_filename(
            "English_2018_01-02-Dec-2018-The-WOW-Factor-by-Ps-Andrew-Tan-Members-guide.pdf"
        )
        assert r["date"] == "2018-12-01"
        assert r["topic"] == "The WOW Factor"
        assert r["speaker"] == "Ps Andrew Tan"

    def test_camelcase_abbreviation_elvm(self):
        r = parse_cell_guide_filename(
            "English_2015_FearOrFaith_eLVM_2015-12-19_20_MessageSummary_MembersGuide.pdf"
        )
        assert r["date"] == "2015-12-19"
        assert r["topic"] == "Fear or Faith"
        assert r["speaker"] == "Elder Lok Vi Ming"

    def test_camelcase_full_name(self):
        r = parse_cell_guide_filename(
            "English_2015_ChooseWisely_PsAndrewTan_2015-12-05_06_MessageSummary_MembersGuide.pdf"
        )
        assert r["date"] == "2015-12-05"
        assert r["topic"] == "Choose Wisely"
        assert r["speaker"] == "Ps Andrew Tan"

    def test_guest_speaker_normalized(self):
        r = parse_cell_guide_filename(
            "English_2015_Pursuit-of-Gods-Presence-by-Rev-David-Ravenhill-members_guide.pdf"
        )
        assert r["speaker"] == "Guest Speaker"

    def test_two_digit_year(self):
        r = parse_cell_guide_filename(
            "English_2015_25-26-July-15-This-Life-the-Next-by-Ps-Chew-Weng-Chee_Members-Guide.pdf"
        )
        assert r["date"] == "2015-07-25"


class TestExtractAnyDate:
    def test_iso_date_in_camelcase_filename(self):
        assert extract_any_date("English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf") == "2018-06-02"

    def test_single_day_month_year(self):
        assert extract_any_date("English_2018_An-Altar-Not-To-Miss-9-June-2018.pdf") == "2018-06-09"

    def test_compact_yyyymmdd(self):
        assert extract_any_date("English_2018_20180623-Growing-Faith-in-God-Final-PPT.pdf") == "2018-06-23"

    def test_no_date_returns_none(self):
        assert extract_any_date("English_2018_SomeSermonNoDate.pdf") is None


class TestExtractTopicWords:
    def test_returns_content_words(self):
        words = extract_topic_words(
            "English_2018_28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide-updated.pdf"
        )
        assert "know" in words
        assert "enemy" in words
        # stop words and metadata terms removed
        assert "members" not in words
        assert "guide" not in words
        assert "elder" not in words

    def test_camelcase_split(self):
        words = extract_topic_words(
            "English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf"
        )
        assert "finishing" in words
        assert "well" in words
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source .venv/bin/activate
pytest tests/test_filename_parser.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError` for `src.ingestion.filename_parser`.

- [ ] **Step 3: Create `src/ingestion/filename_parser.py`**

```python
"""
Parse speaker, date, and topic from BBTC sermon filenames.

Two conventions exist:
  Conv 1 (long hyphenated):  28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide.pdf
  Conv 2 (CamelCase + ISO):  FearOrFaith_eLVM_2015-12-19_20_MessageSummary_MembersGuide.pdf
"""

import re
from src.ingestion.speaker_from_filename import speaker_from_filename
from src.storage.normalize_speaker import normalize_speaker

_MONTHS = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
    'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
    'may': 5, 'jun': 6, 'june': 6,
    'jul': 7, 'july': 7, 'aug': 8, 'august': 8,
    'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
    'nov': 11, 'november': 11, 'dec': 12, 'december': 12,
}

# Removes the cell-guide suffix and everything after it
_MARKER_RE = re.compile(
    r'[-_](?:message[-_]?summary[-_]?)?'
    r'(?:members?(?:27)?|leaders?|cell)[-_](?:guide|copy|guide[-_]updated).*',
    re.IGNORECASE,
)

# Speaker title words that signal the start of a speaker segment
_TITLE_RE = re.compile(r'\b(SP|DSP|Ps|Pastor|Elder|Dr|Rev)\b', re.IGNORECASE)

# CamelCase splitter: insert space before each uppercase-followed-by-lower run
_CAMEL_RE = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')

# Stop words excluded from topic word sets
_STOP = {
    'the', 'and', 'for', 'with', 'by', 'our', 'your', 'you', 'its',
    'not', 'part', 'from', 'this', 'that', 'are', 'was', 'how',
    'who', 'why', 'what', 'when', 'members', 'guide', 'copy', 'leaders',
    'message', 'summary', 'handout', 'english', 'mandarin', 'ppt',
    'notes', 'updated', 'final', 'church', 'bbtc',
}


def _strip(filename: str) -> str:
    """Remove file extension and language/year prefix."""
    s = re.sub(r'\.(pdf|pptx?|docx?)$', '', filename, flags=re.IGNORECASE)
    return re.sub(r'^(English|Mandarin)_\d{4}_', '', s)


def _camel_to_words(s: str) -> str:
    return _CAMEL_RE.sub(' ', s)


def _parse_leading_date(s: str) -> tuple[str | None, str]:
    """
    Parse a leading date token like '28-29-Jul-2018-' or '28-Jul-15-'.
    Returns (ISO-date-string-or-None, remainder-after-date).
    """
    m = re.match(
        r'^(\d{1,2})(?:-(\d{1,2}))?'
        r'[-_]?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may'
        r'|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?'
        r'|nov(?:ember)?|dec(?:ember)?)[a-z]*'
        r'[-_]?(\d{2,4})?[-_]?',
        s.lower(),
    )
    if not m:
        return None, s

    day = int(m.group(1))
    month = _MONTHS[m.group(3)[:3]]
    year_raw = m.group(4)
    if year_raw is None:
        year = None
    else:
        year = int(year_raw)
        if year < 100:
            year += 2000

    date_str = f"{year}-{month:02d}-{day:02d}" if year else None
    return date_str, s[m.end():]


def parse_cell_guide_filename(filename: str) -> dict:
    """
    Extract speaker, date, topic from a BBTC cell guide filename.
    Returns dict with keys: speaker, date, topic (any may be None).
    """
    core = _MARKER_RE.sub('', _strip(filename)).strip('-_')

    # ── Convention 1: leading digit → date-led hyphenated ────────────────────
    if re.match(r'^\d{1,2}[-_]', core):
        date_str, after = _parse_leading_date(core)
        after = after.strip('-_ ')

        by_parts = re.split(r'-by-', after, maxsplit=1, flags=re.IGNORECASE)
        if len(by_parts) == 2:
            topic_raw, speaker_raw = by_parts
            topic = topic_raw.replace('-', ' ').strip().title()
            # Preserve known all-caps acronyms (WOW, MAP, etc.)
            topic = re.sub(r'\b([A-Z]{2,})\b', lambda m: m.group(0), topic)
            speaker = normalize_speaker(speaker_raw.replace('-', ' ').strip())
        else:
            # No "-by-": speaker starts at first title word
            title_m = _TITLE_RE.search(after)
            if title_m:
                topic_raw = after[:title_m.start()].strip('-_ ')
                speaker_raw = after[title_m.start():].replace('-', ' ').strip()
                topic = topic_raw.replace('-', ' ').strip().title()
                speaker = normalize_speaker(speaker_raw)
            else:
                topic = after.replace('-', ' ').strip().title() or None
                speaker = speaker_from_filename(filename)

        return {"speaker": speaker, "date": date_str, "topic": topic or None}

    # ── Convention 2: CamelCase with ISO date ────────────────────────────────
    iso_m = re.search(r'(\d{4})-(\d{2})-(\d{2})', core)
    if iso_m:
        date_str = iso_m.group(0)
        before = core[:core.index(date_str)].rstrip('_-')
        parts = before.rsplit('_', 1)

        if len(parts) == 2:
            topic_camel, speaker_seg = parts
            topic = _camel_to_words(topic_camel).strip()
            # speaker_seg is an abbreviation: eLVM, PsAndrewTan, etc.
            speaker = speaker_from_filename(f"dummy_{speaker_seg}_dummy.pdf")
            if not speaker:
                speaker = normalize_speaker(speaker_seg.replace('-', ' '))
        else:
            topic = _camel_to_words(before).strip()
            speaker = speaker_from_filename(filename)

        return {"speaker": speaker, "date": date_str, "topic": topic or None}

    # Fallback
    return {
        "speaker": speaker_from_filename(filename),
        "date": None,
        "topic": None,
    }


def extract_any_date(filename: str) -> str | None:
    """Return the first recognisable date from any BBTC filename, or None."""
    s = _strip(filename)

    # ISO: 2018-06-02
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', s)
    if m:
        return m.group(0)

    # Compact YYYYMMDD: 20180623
    m = re.search(r'(20\d{2})(\d{2})(\d{2})', s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # Leading date token: 28-29-Jul-2018 or 9-June-2018
    date_str, _ = _parse_leading_date(s)
    return date_str


def extract_topic_words(filename: str) -> set[str]:
    """Return lowercase content words from the filename, for similarity matching."""
    s = _strip(filename)
    # Remove date-like patterns
    s = re.sub(r'\d{4}[-_]\d{2}[-_]\d{2}', ' ', s)
    s = re.sub(r'(20\d{2})(\d{2})(\d{2})', ' ', s)
    s = re.sub(r'\d{1,2}[-_]?\d{1,2}[-_]?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-_]?\d{2,4}',
               ' ', s, flags=re.IGNORECASE)
    # Remove speaker abbreviation patterns
    s = re.sub(r'\b(SP|DSP|Ps|eLVM|eLKG|eGHC|PSL|pCSL|DF)\b', ' ', s, flags=re.IGNORECASE)
    # Split camelCase then on non-alpha
    words = re.split(r'[^a-zA-Z]+', _CAMEL_RE.sub(' ', s))
    return {w.lower() for w in words if len(w) >= 3 and w.lower() not in _STOP}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_filename_parser.py -v
```
Expected: all tests pass. If any assertion fails, fix the regex in `_parse_leading_date` or `parse_cell_guide_filename` for that specific pattern.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/filename_parser.py tests/test_filename_parser.py
git commit -m "feat: add filename_parser — extract speaker/date/topic from BBTC filenames"
```

---

## Task 2: `src/ingestion/file_classifier.py`

**Files:**
- Create: `src/ingestion/file_classifier.py`
- Test: `tests/test_file_classifier.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_file_classifier.py`:

```python
import pytest
from src.ingestion.file_classifier import classify_file


class TestClassifyFile:
    # Cell guide variants
    def test_members_guide_hyphenated(self):
        assert classify_file("English_2018_28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide-updated.pdf") == "cell_guide"

    def test_members27_guide(self):
        assert classify_file("English_2018_10-11-Nov-2018-Stewards-by-Ps-Hakan-Gabrielsson-members27-guide.pdf") == "cell_guide"

    def test_leaders_guide(self):
        assert classify_file("English_2018_15-16-Dec-2018-And-the-Bleeding-Stopped-by-Elder-Chua-Seng-Lee-Leaders-Guide.pdf") == "cell_guide"

    def test_members_copy(self):
        assert classify_file("English_2018_12-13-May-2018-A-Tale-of-4-Mothers-by-Gary-and-Joanna-Koh-Members-Copy.pdf") == "cell_guide"

    def test_camelcase_members_guide(self):
        assert classify_file("English_2015_FearOrFaith_eLVM_2015-12-19_20_MessageSummary_MembersGuide.pdf") == "cell_guide"

    def test_message_summary_members(self):
        assert classify_file("English_2015_ChooseWisely_PsAndrewTan_2015-12-05_06_MessageSummary_MembersGuide.pdf") == "cell_guide"

    # Sermon slides variants
    def test_pptx_extension(self):
        assert classify_file("English_2020_CHURCH-IS-FAMILY-Edric-Sng-12-Feb-2020-website.pptx") == "sermon_slides"

    def test_ppt_keyword(self):
        assert classify_file("English_2018_20180623-Growing-Faith-in-God-Final-PPT.pdf") == "sermon_slides"

    def test_camelcase_abbreviated_pdf(self):
        assert classify_file("English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf") == "sermon_slides"

    def test_camelcase_with_elvm(self):
        assert classify_file("English_2018_WhyTheCross_eLVM_2018-03-24_25.pdf") == "sermon_slides"

    # Other variants
    def test_handout(self):
        assert classify_file("English_2018_EffectivePrayer-1-Principles_Handout.pdf") == "other"

    def test_visual_summary(self):
        assert classify_file("English_2018_VisualSummary_EP5_BlessingsCurses.pdf") == "other"

    def test_visual_summary_hyphenated(self):
        assert classify_file("English_2018_Visual-Summary_EffectivePrayer-6.pdf") == "other"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_file_classifier.py -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/ingestion/file_classifier.py`**

```python
"""Classify BBTC sermon files by role: cell_guide, sermon_slides, or other."""

import re

_CELL_GUIDE_RE = re.compile(
    r'(?:members?(?:27)?|leaders?|cell)[-_]?(?:guide|copy|guide[-_]updated)'
    r'|MembersGuide|MessageSummary.*Members',
    re.IGNORECASE,
)

_OTHER_RE = re.compile(
    r'[-_](handout|visual[-_]?summary)[-_.]|handout\.',
    re.IGNORECASE,
)


def classify_file(filename: str) -> str:
    """
    Returns:
        "cell_guide"    — Members/Leaders/Cell Guide or MessageSummary+Members
        "sermon_slides" — PPT deck, .pptx, or primary sermon PDF
        "other"         — handout, visual summary, or supplementary
    """
    if _CELL_GUIDE_RE.search(filename):
        return "cell_guide"
    if _OTHER_RE.search(filename):
        return "other"
    return "sermon_slides"
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_file_classifier.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/file_classifier.py tests/test_file_classifier.py
git commit -m "feat: add file_classifier — label BBTC files as cell_guide/sermon_slides/other"
```

---

## Task 3: `src/ingestion/sermon_grouper.py`

**Files:**
- Create: `src/ingestion/sermon_grouper.py`
- Test: `tests/test_sermon_grouper.py`

A `SermonGroup` holds one optional cell guide and zero or more paired slide files. Pairing is by date proximity (≤ 3 days) OR high topic-word overlap (Jaccard ≥ 0.5). Date proximity alone is strong enough for sermons on the same weekend; topic overlap handles slide files with no date in their name.

- [ ] **Step 1: Write failing tests**

Create `tests/test_sermon_grouper.py`:

```python
import pytest
from src.ingestion.sermon_grouper import group_sermon_files


class TestGroupSermonFiles:
    def test_pairs_cell_guide_with_matching_slide_by_date(self):
        files = [
            "English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf",
            "English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].cell_guide == files[0]
        assert files[1] in groups[0].slides

    def test_pairs_by_topic_when_slide_has_no_date(self):
        files = [
            "English_2018_09-10-June-2018-An-Altar-Not-to-Miss-by-Ps-Jason-Teo-Members-Guide.pdf",
            "English_2018_An-Altar-Not-To-Miss-9-June-2018.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].cell_guide == files[0]
        assert files[1] in groups[0].slides

    def test_standalone_slide_without_cell_guide(self):
        files = ["English_2018_20180623-Growing-Faith-in-God-Final-PPT.pdf"]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].cell_guide is None
        assert files[0] in groups[0].slides

    def test_standalone_cell_guide_without_slides(self):
        files = ["English_2018_28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide-updated.pdf"]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].cell_guide == files[0]
        assert groups[0].slides == []

    def test_does_not_pair_different_weekends(self):
        files = [
            "English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf",
            "English_2018_09-10-June-2018-An-Altar-Not-to-Miss-by-Ps-Jason-Teo-Members-Guide.pdf",
            "English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf",
            "English_2018_An-Altar-Not-To-Miss-9-June-2018.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 2
        cg_slides = {g.cell_guide: g.slides for g in groups}
        assert "English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf" in \
               cg_slides["English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf"]
        assert "English_2018_An-Altar-Not-To-Miss-9-June-2018.pdf" in \
               cg_slides["English_2018_09-10-June-2018-An-Altar-Not-to-Miss-by-Ps-Jason-Teo-Members-Guide.pdf"]

    def test_handouts_go_to_other(self):
        files = [
            "English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf",
            "English_2018_FinishingWell_Handout.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert "English_2018_FinishingWell_Handout.pdf" in groups[0].other
        assert "English_2018_FinishingWell_Handout.pdf" not in groups[0].slides
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_sermon_grouper.py -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/ingestion/sermon_grouper.py`**

```python
"""Group BBTC sermon files into (cell_guide, slides, other) sermon groups."""

from dataclasses import dataclass, field
from datetime import datetime
from src.ingestion.file_classifier import classify_file
from src.ingestion.filename_parser import extract_any_date, extract_topic_words


@dataclass
class SermonGroup:
    cell_guide: str | None = None
    slides: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)


def _date_proximity(d1: str | None, d2: str | None, tolerance: int = 3) -> bool:
    if not d1 or not d2:
        return False
    fmt = "%Y-%m-%d"
    return abs((datetime.strptime(d1, fmt) - datetime.strptime(d2, fmt)).days) <= tolerance


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def group_sermon_files(filenames: list[str]) -> list[SermonGroup]:
    """
    Group filenames into SermonGroups.
    Each cell guide becomes one group. Slides are paired to their cell guide
    by date proximity (≤ 3 days) or high topic-word Jaccard (≥ 0.5).
    Unpaired slides each become a standalone group.
    """
    cell_guides, slides, others = [], [], []

    for f in filenames:
        kind = classify_file(f)
        if kind == "cell_guide":
            cell_guides.append(f)
        elif kind == "sermon_slides":
            slides.append(f)
        else:
            others.append(f)

    groups: list[SermonGroup] = []
    used_slides: set[str] = set()

    for cg in cell_guides:
        group = SermonGroup(cell_guide=cg)
        cg_date = extract_any_date(cg)
        cg_words = extract_topic_words(cg)

        for slide in slides:
            if slide in used_slides:
                continue
            slide_date = extract_any_date(slide)
            slide_words = extract_topic_words(slide)

            near = _date_proximity(cg_date, slide_date)
            similar = _jaccard(cg_words, slide_words) >= 0.5

            if near or similar:
                group.slides.append(slide)
                used_slides.add(slide)

        # Attach handouts/other files that topic-match this cell guide
        for o in others:
            o_words = extract_topic_words(o)
            if _jaccard(cg_words, o_words) >= 0.4:
                group.other.append(o)

        groups.append(group)

    # Standalone slides (no matching cell guide)
    for slide in slides:
        if slide not in used_slides:
            groups.append(SermonGroup(slides=[slide]))

    return groups
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_sermon_grouper.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/sermon_grouper.py tests/test_sermon_grouper.py
git commit -m "feat: add sermon_grouper — pair cell guides with PPT slides by date/topic"
```

---

## Task 4: Add `topic` column to SQLite

**Files:**
- Modify: `src/storage/sqlite_store.py:12-30` (schema), `:56-65` (insert_sermon)

- [ ] **Step 1: Write failing test**

Add to `tests/test_filename_parser.py` (or a new file `tests/test_sqlite_store.py`):

```python
import os, tempfile, pytest
from src.storage.sqlite_store import SermonRegistry


def test_topic_column_exists():
    with tempfile.TemporaryDirectory() as d:
        reg = SermonRegistry(db_path=os.path.join(d, "t.db"))
        import sqlite3
        with sqlite3.connect(reg.db_path) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(sermons)").fetchall()]
        assert "topic" in cols


def test_insert_sermon_stores_topic():
    with tempfile.TemporaryDirectory() as d:
        reg = SermonRegistry(db_path=os.path.join(d, "t.db"))
        reg.insert_sermon({
            "sermon_id": "test-001",
            "filename": "test.pdf",
            "url": "http://example.com/test.pdf",
            "speaker": "SP Daniel Foo",
            "date": "2018-07-28",
            "topic": "Know Your Enemy",
            "status": "indexed",
        })
        import sqlite3
        with sqlite3.connect(reg.db_path) as conn:
            row = conn.execute(
                "SELECT topic FROM sermons WHERE sermon_id = 'test-001'"
            ).fetchone()
        assert row[0] == "Know Your Enemy"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_sqlite_store.py -v
```
Expected: `test_topic_column_exists` fails — column doesn't exist yet.

- [ ] **Step 3: Add `topic` column to `src/storage/sqlite_store.py`**

In `_init_db`, add `topic TEXT` to the `CREATE TABLE` statement and add a migration for existing databases:

```python
def _init_db(self):
    with sqlite3.connect(self.db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sermons (
                sermon_id TEXT PRIMARY KEY,
                filename TEXT,
                url TEXT UNIQUE,
                speaker TEXT,
                date TEXT,
                series TEXT,
                bible_book TEXT,
                primary_verse TEXT,
                topic TEXT,
                language TEXT,
                file_type TEXT,
                year INTEGER,
                status TEXT,
                date_scraped TEXT
            )
        """)
        # Migrate existing databases that predate the topic column
        existing = [r[1] for r in conn.execute("PRAGMA table_info(sermons)").fetchall()]
        if "topic" not in existing:
            conn.execute("ALTER TABLE sermons ADD COLUMN topic TEXT")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bible_versions (
                version_id TEXT PRIMARY KEY,
                filename TEXT,
                status TEXT,
                date_indexed TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sermon_intelligence (
                sermon_id TEXT PRIMARY KEY,
                speaker TEXT,
                primary_verse TEXT,
                verses_used TEXT,
                summary TEXT,
                FOREIGN KEY (sermon_id) REFERENCES sermons (sermon_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON sermons(url)")
```

The `insert_sermon` method already uses dynamic `record.keys()` / values, so it handles `topic` automatically — no further change needed there.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_sqlite_store.py -v
```
Expected: both pass.

- [ ] **Step 5: Confirm migration works on live DB**

```bash
source .venv/bin/activate
python -c "from src.storage.sqlite_store import SermonRegistry; r = SermonRegistry(); print('topic column OK')"
```
Expected: prints `topic column OK` without error.

- [ ] **Step 6: Commit**

```bash
git add src/storage/sqlite_store.py tests/test_sqlite_store.py
git commit -m "feat: add topic column to sermons table with backward-compatible migration"
```

---

## Task 5: `backfill_metadata.py` — fix all existing records

**Files:**
- Create: `backfill_metadata.py`

This script re-parses speaker, date, and topic from the filename for every cell guide in the DB, and applies `speaker_from_filename()` to every sermon slide. It does **not** re-vectorise (re-vectorisation can be triggered by resetting statuses and re-running the Dagster pipeline if needed).

- [ ] **Step 1: Create `backfill_metadata.py`**

```python
"""
One-time backfill: re-extract speaker, date, topic from filenames for all
existing records in data/sermons.db.

Run from project root:
    python backfill_metadata.py [--dry-run]
"""

import sys
import sqlite3
from src.ingestion.file_classifier import classify_file
from src.ingestion.filename_parser import parse_cell_guide_filename
from src.ingestion.speaker_from_filename import speaker_from_filename
from src.storage.normalize_speaker import normalize_speaker
from src.storage.sqlite_store import SermonRegistry

DB_PATH = "data/sermons.db"
DRY_RUN = "--dry-run" in sys.argv


def main():
    # Ensure topic column exists
    SermonRegistry(DB_PATH)

    changed = 0
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT sermon_id, filename, speaker, date, topic FROM sermons"
        ).fetchall()

        for sermon_id, filename, cur_speaker, cur_date, cur_topic in rows:
            kind = classify_file(filename)
            updates: dict = {}

            if kind == "cell_guide":
                parsed = parse_cell_guide_filename(filename)
                if parsed.get("speaker") and parsed["speaker"] != cur_speaker:
                    updates["speaker"] = parsed["speaker"]
                if parsed.get("date") and parsed["date"] != cur_date:
                    updates["date"] = parsed["date"]
                if parsed.get("topic") and parsed["topic"] != cur_topic:
                    updates["topic"] = parsed["topic"]

            elif kind == "sermon_slides":
                new_sp = speaker_from_filename(filename)
                if new_sp and new_sp != cur_speaker:
                    updates["speaker"] = new_sp

            if updates:
                changed += 1
                label = f"[{kind}] {filename}"
                for k, v in updates.items():
                    print(f"  {k}: {cur_speaker if k == 'speaker' else '?'!r} → {v!r}  ({label})")
                if not DRY_RUN:
                    set_clause = ", ".join(f"{k} = ?" for k in updates)
                    conn.execute(
                        f"UPDATE sermons SET {set_clause} WHERE sermon_id = ?",
                        (*updates.values(), sermon_id),
                    )

        if not DRY_RUN:
            conn.commit()

    mode = "DRY RUN — " if DRY_RUN else ""
    print(f"\n{mode}{changed} record(s) updated.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run to review changes**

```bash
source .venv/bin/activate
python backfill_metadata.py --dry-run 2>&1 | head -60
```
Review the output. Confirm the speaker/date/topic changes look correct. Pay attention to any unexpected replacements.

- [ ] **Step 3: Apply the backfill**

```bash
python backfill_metadata.py
```
Expected: prints a count of updated records.

- [ ] **Step 4: Verify speaker roster after backfill**

```bash
python - <<'EOF'
import sqlite3
with sqlite3.connect("data/sermons.db") as conn:
    rows = conn.execute(
        "SELECT speaker, COUNT(*) n FROM sermons GROUP BY speaker ORDER BY n DESC"
    ).fetchall()
print(f"{len(rows)} distinct speakers:")
for sp, n in rows:
    print(f"  {n:>4}  {sp}")
EOF
```
Expected: `Name` count drops significantly; main BBTC pastors show correct canonical names.

- [ ] **Step 5: Commit**

```bash
git add backfill_metadata.py
git commit -m "feat: add backfill_metadata script — re-extract speaker/date/topic from filenames"
```

---

## Task 6: Update `dagster_pipeline.py` — process sermon groups

**Files:**
- Modify: `dagster_pipeline.py:67-161`

Replace the flat per-file processing loop with a group-aware loop. For each group:
- Cell guide present → parse metadata from filename; vectorise both cell guide + slide content under the cell guide's `sermon_id`.
- No cell guide → fall back to LLM extraction on the slide file (existing behaviour).

- [ ] **Step 1: Read current loop** (`dagster_pipeline.py:83-145`) to understand the existing structure, then apply the diff below.

- [ ] **Step 2: Replace the ingestion loop in `dagster_pipeline.py`**

Replace the block from `# 2. Process newly 'extracted' sermons` to the end of the inner loop with:

```python
    # 2. Group files and process as sermon units
    from src.ingestion.file_classifier import classify_file
    from src.ingestion.filename_parser import parse_cell_guide_filename
    from src.ingestion.sermon_grouper import group_sermon_files
    from src.ingestion.speaker_from_filename import speaker_from_filename
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    sermons = registry.get_all_sermons()
    pending = [s for s in sermons if s['status'] in ('extracted', 'processed')]
    context.log.info(f"📋 {len(pending)} sermon files pending indexing.")

    # Index pending files by filename for fast lookup
    by_filename = {s['filename']: s for s in pending}
    pending_filenames = list(by_filename.keys())

    sermon_groups = group_sermon_files(pending_filenames)
    context.log.info(f"📦 {len(sermon_groups)} sermon groups formed.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    processed_count = 0

    def _load_text(sermon: dict) -> str | None:
        txt_name = os.path.splitext(sermon['filename'])[0] + ".txt"
        txt_path = os.path.join("data/sermons", txt_name)
        if not os.path.exists(txt_path):
            context.log.warning(f"⚠️ Text file not found: {txt_name}")
            return None
        with open(txt_path, encoding="utf-8") as f:
            return f.read()

    def _vectorise(sermon_id: str, text: str, meta: dict):
        chunks = splitter.split_text(text) or [text[:500]]
        ids = [f"{sermon_id}_{i}" for i in range(len(chunks))]
        vector_store.upsert_sermon_chunks(chunks, [meta] * len(chunks), ids)

    for group in sermon_groups:
        # ── Cell-guide-led group ──────────────────────────────────────────────
        if group.cell_guide:
            cg_sermon = by_filename[group.cell_guide]
            sermon_id = cg_sermon['sermon_id']

            # Metadata from filename (reliable — no LLM needed)
            parsed = parse_cell_guide_filename(group.cell_guide)
            speaker = parsed.get("speaker") or speaker_from_filename(group.cell_guide)
            date    = parsed.get("date") or cg_sermon.get("date")
            topic   = parsed.get("topic")

            context.log.info(f"📖 [{sermon_id}] {topic or group.cell_guide} — {speaker}")

            update_record = {
                "sermon_id":     sermon_id,
                "filename":      cg_sermon['filename'],
                "url":           cg_sermon['url'],
                "speaker":       speaker,
                "date":          date,
                "topic":         topic,
                "series":        cg_sermon.get("series"),
                "bible_book":    cg_sermon.get("bible_book"),
                "primary_verse": cg_sermon.get("primary_verse"),
                "status":        "indexed",
            }
            registry.insert_sermon(update_record)

            # Vectorise cell guide content
            cg_text = _load_text(cg_sermon)
            if cg_text:
                _vectorise(sermon_id, cg_text, update_record)

            # Vectorise paired slide content under the same sermon_id
            for slide_filename in group.slides:
                slide_sermon = by_filename.get(slide_filename)
                if not slide_sermon:
                    continue
                context.log.info(f"  📎 Adding slides: {slide_filename}")
                slide_text = _load_text(slide_sermon)
                if slide_text:
                    _vectorise(sermon_id, slide_text, update_record)
                # Mark slide as indexed (it's been absorbed into the cell guide group)
                registry.mark_processed(slide_sermon['url'], status="indexed")

            # LLM intelligence (summary + verses) from combined text
            combined = "\n\n".join(filter(None, [cg_text, *(
                _load_text(by_filename[s]) for s in group.slides if s in by_filename
            )]))
            if combined:
                intel = extractor.extract(combined[:2000])
                registry.insert_intelligence({
                    "sermon_id":     sermon_id,
                    "speaker":       speaker,
                    "primary_verse": intel.get("primary_verse"),
                    "verses_used":   intel.get("verses_used"),
                    "summary":       intel.get("summary"),
                })

            processed_count += 1

        # ── Standalone slides (no matching cell guide) ────────────────────────
        else:
            for slide_filename in group.slides:
                slide_sermon = by_filename.get(slide_filename)
                if not slide_sermon:
                    continue
                sermon_id = slide_sermon['sermon_id']
                slide_text = _load_text(slide_sermon)
                if not slide_text:
                    continue

                context.log.info(f"📄 Standalone: {slide_filename}")
                meta = extractor.extract(slide_text[:2000])
                speaker = (
                    speaker_from_filename(slide_filename)
                    or meta.get("speaker")
                )
                update_record = {
                    "sermon_id":     sermon_id,
                    "filename":      slide_sermon['filename'],
                    "url":           slide_sermon['url'],
                    "speaker":       speaker,
                    "date":          meta.get("date") or slide_sermon.get("date"),
                    "series":        meta.get("series"),
                    "bible_book":    meta.get("bible_book"),
                    "primary_verse": meta.get("primary_verse"),
                    "status":        "indexed",
                }
                registry.insert_sermon(update_record)
                _vectorise(sermon_id, slide_text, update_record)
                registry.insert_intelligence({
                    "sermon_id":     sermon_id,
                    "speaker":       speaker,
                    "primary_verse": meta.get("primary_verse"),
                    "verses_used":   meta.get("verses_used"),
                    "summary":       meta.get("summary"),
                })
                processed_count += 1
```

- [ ] **Step 3: Remove the old imports** that are now inside the function body (move `from langchain_text_splitters import RecursiveCharacterTextSplitter` from inside the loop to the top of the file or the function imports block).

- [ ] **Step 4: Run the pipeline on a small test**

```bash
source .venv/bin/activate
# Reset one known cell-guide record to 'extracted' to test pipeline
python - <<'EOF'
import sqlite3
with sqlite3.connect("data/sermons.db") as conn:
    conn.execute("""
        UPDATE sermons SET status = 'extracted'
        WHERE filename = '2018_28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide-updated.pdf'
           OR filename LIKE '%Know-Your-Enemy%'
        LIMIT 2
    """)
    print("Reset", conn.execute("SELECT changes()").fetchone()[0], "rows")
EOF
```

Then trigger the Dagster asset locally:

```bash
DAGSTER_HOME=$(mktemp -d) dagster asset materialize --select sermon_ingestion_summary -m dagster_pipeline 2>&1 | tail -30
```
Expected: the "Know Your Enemy" cell guide appears with `speaker = "Ps Edric Sng"` and `date = "2018-07-28"`.

- [ ] **Step 5: Commit**

```bash
git add dagster_pipeline.py
git commit -m "feat: rework ingestion pipeline to process sermon groups — cell guide + PPT as one unit"
```

---

## Self-Review

**Spec coverage:**
- ✅ Cell guide metadata (speaker, date, topic) extracted from filename — Task 1
- ✅ PPT/slide files classified separately — Task 2
- ✅ Cell guide + PPT grouped as one sermon — Task 3
- ✅ `topic` column in SQLite — Task 4
- ✅ Existing data fixed — Task 5
- ✅ Pipeline updated — Task 6

**Placeholder scan:** No TBDs or "similar to Task N" references found.

**Type consistency:**
- `SermonGroup.cell_guide` is `str | None` throughout — consistent.
- `extract_any_date` returns `str | None` — used that way in `sermon_grouper.py`.
- `parse_cell_guide_filename` returns `dict` with optional keys — consumed with `.get()` everywhere — consistent.

**Known edge cases handled:**
- Two-digit years (`July-15` → 2015) in `_parse_leading_date`
- `Members27` URL-encoded apostrophe in `_CELL_GUIDE_RE`
- Slide files already `status = "indexed"` are excluded from `pending` — no double-processing
- If `group.slides` contains a filename not in `by_filename` (e.g. it's not pending), it's skipped safely
