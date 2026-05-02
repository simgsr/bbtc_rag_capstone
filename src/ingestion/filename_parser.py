"""
Parse speaker, date, and topic from BBTC sermon filenames.

Two conventions exist:
  Conv 1 (long hyphenated):  28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide.pdf
  Conv 2 (CamelCase + ISO):  FearOrFaith_eLVM_2015-12-19_20_MessageSummary_MembersGuide.pdf
"""

import re
from src.ingestion.speaker_from_filename import speaker_from_filename
from src.storage.normalize_speaker import normalize_speaker, normalize_speaker_strict

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
    r'(?:members?(?:27)?|leaders?|cell)[-_]?(?:guide|copy|guide[-_]updated).*',
    re.IGNORECASE,
)

# Speaker title words that signal the start of a speaker segment
# Use (?<![A-Za-z]) instead of \b so that _DSP and _Ps after underscores also match
_TITLE_RE = re.compile(r'(?<![A-Za-z])(SP|DSP|Ps|Pastor|Elder|Dr|Rev)(?![A-Za-z])', re.IGNORECASE)

# CamelCase splitter
_CAMEL_RE = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')

_STOP = {
    'the', 'and', 'for', 'with', 'by', 'our', 'your', 'you', 'its',
    'not', 'part', 'from', 'this', 'that', 'are', 'was', 'how',
    'who', 'why', 'what', 'when', 'members', 'guide', 'copy', 'leaders',
    'message', 'summary', 'handout', 'english', 'mandarin', 'ppt',
    'notes', 'updated', 'final', 'church', 'bbtc',
    # Title words — should not appear as topic content words
    'elder', 'pastor', 'rev', 'reverend', 'dr',
}

# Minor words kept lowercase in title (unless first word)
_MINOR = {'of', 'or', 'the', 'a', 'an', 'in', 'to', 'and', 'for', 'on', 'at'}


def _smart_title(text: str) -> str:
    """Title-case text, preserving ALL-CAPS tokens and lowercasing minor words."""
    words = text.split()
    out = []
    for i, w in enumerate(words):
        if w.isupper() and len(w) > 1:
            # Preserve initialisms / acronyms like WOW, NDP
            out.append(w)
        elif i > 0 and w.lower() in _MINOR:
            # Minor connector words stay lowercase (unless first word)
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:].lower() if w[:1].isalpha() else w)
    return ' '.join(out)


def _strip(filename: str) -> str:
    s = re.sub(r'\.(pdf|pptx?|docx?)$', '', filename, flags=re.IGNORECASE)
    return re.sub(r'^(English|Mandarin)_\d{4}_', '', s)


def _camel_to_words(s: str) -> str:
    return _CAMEL_RE.sub(' ', s)


def _parse_leading_date(s: str) -> tuple[str | None, str]:
    m = re.match(
        r'^(\d{1,2})(?:-(\d{1,2}))?'
        r'[-_]?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may'
        r'|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?'
        r'|nov(?:ember)?|dec(?:ember)?)[a-z]*'
        r'[-_]?(20\d{2}|[12]\d)?[-_]?',
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


def _search_date_month_year(s: str) -> str | None:
    """Search mid-string for a dd-Month-yyyy pattern (single day, not just leading)."""
    m = re.search(
        r'\b([0-2]?\d|3[01])(?:[-_]\d{1,2})?[-_]?'
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may'
        r'|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?'
        r'|nov(?:ember)?|dec(?:ember)?)[a-z]*'
        r'[-_]?(20\d{2}|[12]\d)\b',
        s, flags=re.IGNORECASE,
    )
    if not m:
        return None
    day = int(m.group(1))
    month = _MONTHS[m.group(2).lower()[:3]]
    year = int(m.group(3))
    if year < 100:
        year += 2000
    return f"{year}-{month:02d}-{day:02d}"


def parse_cell_guide_filename(filename: str) -> dict:
    """
    Extract speaker, date, topic from a BBTC cell guide filename.
    Returns dict with keys: speaker, date, topic (any may be None).
    """
    core = _MARKER_RE.sub('', _strip(filename)).strip('-_')

    # Convention 1: leading digit → date-led hyphenated
    if re.match(r'^\d{1,2}[-_]', core):
        date_str, after = _parse_leading_date(core)
        after = after.strip('-_ ')

        last_by = after.lower().rfind('-by-')
        if last_by >= 0:
            topic_raw = after[:last_by]
            speaker_raw = after[last_by + 4:]
            speaker_raw = re.sub(r'[_\-](?:notes?|handout|slides?|summary|ppt|v\d[\w.]*).*$', '', speaker_raw, flags=re.IGNORECASE)
            # Strip trailing date suffixes like -2016-03-05_06
            speaker_raw = re.sub(r'[-_]\d{4}[-_]\d{2}[-_]\d{2}.*$', '', speaker_raw)
            topic = _smart_title(topic_raw.replace('-', ' ').strip())
            speaker = normalize_speaker(speaker_raw.replace('-', ' ').strip()) or speaker_from_filename(filename)
        else:
            title_m = _TITLE_RE.search(after)
            if title_m:
                topic_raw = after[:title_m.start()].strip('-_ ')
                speaker_raw = after[title_m.start():].replace('-', ' ').strip()
                topic = _smart_title(topic_raw.replace('-', ' ').strip())
                speaker = normalize_speaker(speaker_raw)
            else:
                # Last resort: only accept known canonical speakers to avoid returning topic words
                name_m = re.search(r'(?:^|[-_])([A-Z][a-z]+(?:[-][A-Z][a-z]+)+)$', after)
                if name_m:
                    candidate = name_m.group(1).replace('-', ' ')
                    topic_raw = after[:name_m.start()].strip('-_ ')
                    speaker = normalize_speaker_strict(candidate)
                    topic = _smart_title(topic_raw.replace('-', ' ').strip()) or None
                else:
                    topic = _smart_title(after.replace('-', ' ').strip()) or None
                    speaker = speaker_from_filename(filename)

        if not date_str:
            date_str = extract_any_date(filename)

        return {"speaker": speaker, "date": date_str, "topic": topic or None}

    # Hyphenated with "by-Speaker" but no leading date (e.g. Pursuit-of-Gods-Presence-by-Rev-David-Ravenhill)
    last_by = core.lower().rfind('-by-')
    if last_by >= 0:
        topic_raw = core[:last_by]
        speaker_raw = core[last_by + 4:]
        # Strip trailing date suffixes like -2016-03-05_06
        speaker_raw = re.sub(r'[-_]\d{4}[-_]\d{2}[-_]\d{2}.*$', '', speaker_raw)
        topic = _smart_title(topic_raw.replace('-', ' ').strip())
        speaker = normalize_speaker(speaker_raw.replace('-', ' ').strip())
        return {"speaker": speaker, "date": None, "topic": topic or None}

    # Convention 2: CamelCase with ISO date
    iso_m = re.search(r'(\d{4})-(\d{2})-(\d{2})', core)
    if iso_m:
        date_str = iso_m.group(0)
        before = core[:core.index(date_str)].rstrip('_-')
        parts = before.rsplit('_', 1)

        if len(parts) == 2:
            topic_camel, speaker_seg = parts
            topic = _smart_title(_camel_to_words(topic_camel).strip())
            speaker = speaker_from_filename(f"dummy_{speaker_seg}_dummy.pdf")
            if not speaker:
                speaker = normalize_speaker(speaker_seg.replace('-', ' '))
        else:
            topic = _smart_title(_camel_to_words(before).strip())
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

    # ISO date: YYYY-MM-DD
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', s)
    if m:
        return m.group(0)

    # Compact: YYYYMMDD (with valid ranges)
    m = re.search(r'\b(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b', s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # Named month (mid-string search, e.g. "9-June-2018")
    result = _search_date_month_year(s)
    if result:
        return result

    return None


def extract_topic_words(filename: str) -> set[str]:
    """Return lowercase content words from the filename, for similarity matching."""
    s = _strip(filename)
    s = re.sub(r'\d{4}[-_]\d{2}[-_]\d{2}', ' ', s)
    s = re.sub(r'(20\d{2})(\d{2})(\d{2})', ' ', s)
    s = re.sub(
        r'\d{1,2}[-_]?\d{1,2}[-_]?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-_]?\d{2,4}',
        ' ', s, flags=re.IGNORECASE,
    )
    s = re.sub(r'\b(SP|DSP|Ps|eLVM|eLKG|eGHC|PSL|pCSL|DF)\b', ' ', s, flags=re.IGNORECASE)
    words = re.split(r'[^a-zA-Z]+', _CAMEL_RE.sub(' ', s))
    return {w.lower() for w in words if len(w) >= 3 and w.lower() not in _STOP}
