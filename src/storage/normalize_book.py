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
    "1 john": "1 John", "1john": "1 John", "1jn": "1 John", "1 jn": "1 John",
    "2 john": "2 John", "2john": "2 John", "2jn": "2 John", "2 jn": "2 John",
    "3 john": "3 John", "3john": "3 John", "3jn": "3 John", "3 jn": "3 John",
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


# Ambiguous unnumbered books that normalize_book() cannot resolve alone.
# key → (book1, book2, max_ch_book1, max_ch_book2). book1 is the default.
BOOK_DISAMBIGUATION: dict[str, tuple] = {
    "samuel":      ("1 Samuel",      "2 Samuel",      31, 24),
    "kings":       ("1 Kings",       "2 Kings",       22, 25),
    "chronicles":  ("1 Chronicles",  "2 Chronicles",  29, 36),
    "corinthians": ("1 Corinthians", "2 Corinthians", 16, 13),
    "timothy":     ("1 Timothy",     "2 Timothy",     6,  4),
    "peter":       ("1 Peter",       "2 Peter",       5,  3),
}


def disambiguate_book(raw: str, chapter) -> str | None:
    """Resolve an ambiguous unnumbered book name using chapter number.

    Returns the canonical book name, or None if raw is not a known ambiguous key.
    """
    if not raw:
        return None
    key = raw.strip().lower()
    if key not in BOOK_DISAMBIGUATION:
        return None
    book1, book2, max1, max2 = BOOK_DISAMBIGUATION[key]
    if chapter is None:
        return book1
    ch = int(chapter)
    if ch > max1 and ch > max2:
        return book1   # invalid chapter — use default
    if ch > max1:
        return book2   # exceeds book1's max → must be book2
    if ch > max2:
        return book1   # exceeds book2's max → must be book1
    return book1       # ambiguous overlap — default to book1
