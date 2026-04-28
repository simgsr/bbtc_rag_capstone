"""Extract verse references from BBTC Slides/PPT (PS) filenames and text."""

import re
import os

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

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
_BOOK_PATTERN = "|".join(sorted(_BOOKS.keys(), key=len, reverse=True))

# Matches: LUKE-9V23, LUKE-10V1-3, JOHN-11, HEBREWS
_VERSE_RE = re.compile(
    rf'(?<![A-Za-z])({_BOOK_PATTERN})'
    r'(?:-(\d{1,3})(?:V(\d{1,3})(?:-(\d{1,3}))?)?)?'
    r'(?![A-Za-z])',
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
    Return list of verse dicts parsed from the filename.
    Each dict: {verse_ref, book, chapter, verse_start, verse_end, is_key_verse}.
    First match is the key verse (is_key_verse=1).
    """
    core = _strip_prefix(filename)
    # Remove 8-digit date stamps and version suffixes to reduce false matches
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
    if not _FITZ_AVAILABLE:
        return ""
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
        verse_pattern = re.compile(r'^[A-Z][a-z]+ \d+:\d+', re.IGNORECASE)
        return [l for l in lines if verse_pattern.match(l)]
    except Exception:
        return []
