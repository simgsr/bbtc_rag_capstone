import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re

BIBLE_BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth",
    "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah",
    "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah",
    "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah",
    "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
    "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
    "Ephesians", "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy",
    "2 Timothy", "Titus", "Philemon", "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John",
    "3 John", "Jude", "Revelation"
]

# Longest-first so "1 Samuel" wins over "Samuel"
_BOOK_PAT = re.compile(
    r'(?<![A-Za-z])(' + '|'.join(re.escape(b) for b in sorted(BIBLE_BOOKS, key=len, reverse=True)) + r')(?![A-Za-z])',
    re.IGNORECASE,
)
_CHAP_WORD_RE = re.compile(r'(?:Chapter|Chap\.?)\s+(\d+)', re.IGNORECASE)


def _canonical_book(raw: str) -> str | None:
    for b in BIBLE_BOOKS:
        if b.lower() == raw.lower():
            return b
    return None


class BibleEpubParser:
    def __init__(self, filepath, version_name):
        self.filepath = filepath
        self.version_name = version_name
        self.book_obj = epub.read_epub(filepath, {"ignore_ncx": True})
        self._file_to_book = self._map_toc()

    def _map_toc(self):
        mapping = {}

        def _norm(t):
            t = t.lower()
            for p in ("1st ", "2nd ", "3rd "):
                t = t.replace(p, p[0] + " ")
            return t.strip()

        def _walk(links):
            for link in links:
                if isinstance(link, (list, tuple)):
                    _walk(link)
                elif hasattr(link, 'title') and hasattr(link, 'href'):
                    title = _norm(str(link.title))
                    fname = link.href.split('#')[0]
                    for book in BIBLE_BOOKS:
                        if title == book.lower() or title.startswith(book.lower() + " "):
                            mapping[fname] = book
                            break

        _walk(self.book_obj.toc)
        return mapping

    def parse(self):
        verses_dict: dict = {}

        current_book = "Unknown"
        current_chapter = 0
        current_v_num: int | None = None
        current_v_parts: list[str] = []

        def flush():
            nonlocal current_v_num, current_v_parts
            if current_v_num is None or current_book == "Unknown" or current_chapter == 0:
                current_v_num = None
                current_v_parts = []
                return
            text = re.sub(r'\s+', ' ', " ".join(current_v_parts)).strip()
            # Strip leading verse number artifact ("7 " or "7:")
            text = re.sub(rf'^{re.escape(str(current_v_num))}[:\s]+', '', text).strip()
            if text and len(text) >= 3:
                ref_id = f"{current_book[:3].upper()}_{current_chapter:03}_{current_v_num:03}"
                reference = f"{current_book} {current_chapter}:{current_v_num}"
                if ref_id in verses_dict:
                    verses_dict[ref_id]['text'] += " " + text
                else:
                    verses_dict[ref_id] = {
                        "book": current_book, "chapter": current_chapter,
                        "verse": current_v_num, "text": text,
                        "version": self.version_name,
                        "ref_id": ref_id, "reference": reference,
                    }
            current_v_num = None
            current_v_parts = []

        for item in self.book_obj.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            fname = item.get_name()
            if fname in self._file_to_book:
                flush()
                current_book = self._file_to_book[fname]
                current_chapter = 0

            soup = BeautifulSoup(item.get_content(), 'html.parser')

            # Only walk <p> and heading elements — NOT div/span at this level.
            # Nested divs/spans share text with their parent <p>, causing triple-hits.
            for para in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4']):
                raw = para.get_text(' ', strip=True)
                if not raw:
                    continue

                # ── Determine if this paragraph has inline verse numbers ─────────
                sups = para.find_all(['sup', 'big'])
                # Also check for <b> tags that might contain the first verse marker like "1:1"
                bs = para.find_all('b')
                
                # A paragraph might start with "ChapterNum:VerseNum" e.g. "2:1"
                # We check the very beginning of the paragraph text.
                chapter_verse_match = re.match(r'^(\d+)\s*:\s*(\d+)', raw)
                if chapter_verse_match:
                    new_ch = int(chapter_verse_match.group(1))
                    new_v = int(chapter_verse_match.group(2))
                    if new_ch != current_chapter:
                        flush()
                        current_chapter = new_ch
                        # We don't continue here because this paragraph contains the verse text.

                sup_verse_nums: list[int] = []
                for s in sups:
                    t = s.get_text().strip()
                    if t.isdigit() and 1 <= int(t) <= 200:
                        sup_verse_nums.append(int(t))
                
                for b in bs:
                    # Look for "2:1" inside a <b> tag
                    bm = re.search(r'(\d+)\s*:\s*(\d+)', b.get_text())
                    if bm:
                        new_ch = int(bm.group(1))
                        if new_ch != current_chapter:
                            flush()
                            current_chapter = new_ch
                        sup_verse_nums.append(int(bm.group(2)))

                # Only check for chapter/book headers on paragraphs WITHOUT verse markers.
                # This prevents false matches on words like "mark", "Titus", "Ruth" in verse text.
                if not sup_verse_nums and not chapter_verse_match:
                    bm = _BOOK_PAT.search(raw)
                    if bm:
                        cand = _canonical_book(bm.group(1))
                        if cand:
                            after = raw[bm.end():].strip()
                            cm = re.match(r'^(\d+)', after)
                            if cm:
                                flush()
                                current_book = cand
                                current_chapter = int(cm.group(1))
                            elif not after or not after[0].isdigit():
                                flush()
                                current_book = cand
                        continue

                    cw = _CHAP_WORD_RE.search(raw)
                    if cw:
                        flush()
                        current_chapter = int(cw.group(1))
                        continue

                if current_book == "Unknown" or current_chapter == 0:
                    continue

                if not sup_verse_nums and not chapter_verse_match:
                    # No verse-number markers — plain continuation text
                    if current_v_num is not None:
                        current_v_parts.append(raw)
                    continue

                # Reconstruct verse texts by splitting on verse boundaries
                tokens: list[tuple[str, int | str]] = []   # ("V", num) or ("T", text)
                
                # We need to be careful with descendants. 
                # For ESV, a chapter start looks like: <b><big>2</big>:1</b>
                for child in para.descendants:
                    if child.name in ('sup', 'big', 'b'):
                        t = child.get_text().strip()
                        # Match "1:1" or just "1"
                        m = re.search(r'(?:(\d+)\s*:\s*)?(\d+)', t)
                        if m:
                            v_num = int(m.group(2))
                            if 1 <= v_num <= 200:
                                # If it's a "Chapter:Verse" marker, update chapter
                                if m.group(1):
                                    ch_num = int(m.group(1))
                                    if ch_num != current_chapter:
                                        # Note: we don't flush here yet, we'll flush when we process the 'V' token
                                        current_chapter = ch_num
                                
                                # Only add 'V' token if this tag doesn't contain other tags 
                                # (to avoid double counting if it's <b><sup>2</sup></b>)
                                if not child.find(['sup', 'big', 'b']):
                                    tokens.append(('V', v_num))
                    elif child.name is None:  # NavigableString
                        t = str(child).strip()
                        if t:
                            # Skip strings that are purely verse markers we already handled
                            if child.parent and child.parent.name in ('sup', 'big', 'b'):
                                # If the parent is a tag we handled, and this string is just part of it, skip.
                                # But be careful: if <b>2:1</b> contains the text "2:1", we skip it.
                                if re.fullmatch(r'(?:\d+\s*:\s*)?\d+\s*', t):
                                    continue
                            tokens.append(('T', t))

                for tok_type, tok_val in tokens:
                    if tok_type == 'V':
                        flush()
                        current_v_num = int(tok_val)
                        current_v_parts = []
                    else:
                        if current_v_num is not None:
                            current_v_parts.append(str(tok_val))

        flush()
        return list(verses_dict.values())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        parser = BibleEpubParser(sys.argv[1], "TEST")
        results = parser.parse()
        print(f"Parsed {len(results)} verses.")
        sample = sorted([v for v in results if v['book'] == '2 Timothy' and v['chapter'] == 1],
                        key=lambda x: x['verse'])
        print(f"2 Timothy ch1 ({len(sample)} verses):")
        for v in sample:
            print(f"  {v['reference']}: {v['text'][:80]}")
