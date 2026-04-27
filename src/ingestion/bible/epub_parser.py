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

class BibleEpubParser:
    def __init__(self, filepath, version_name):
        self.filepath = filepath
        self.version_name = version_name
        self.book_obj = epub.read_epub(filepath)
        self._file_to_book = self._map_toc()
        
    def _map_toc(self):
        mapping = {}
        def normalize(t):
            t = t.lower().replace("1st ", "1 ").replace("2nd ", "2 ").replace("3rd ", "3 ")
            return t.strip()

        def process_toc(links):
            for link in links:
                if isinstance(link, (list, tuple)):
                    process_toc(link)
                elif hasattr(link, 'title') and hasattr(link, 'href'):
                    title = normalize(str(link.title))
                    filename = link.href.split('#')[0]
                    for book in BIBLE_BOOKS:
                        if title == book.lower() or title.startswith(book.lower() + " "):
                            mapping[filename] = book
        process_toc(self.book_obj.toc)
        return mapping

    def parse(self):
        self.verses_dict = {} 
        self.current_v_num = None
        self.current_v_text = []
        self.current_book = "Unknown"
        self.current_chapter = 0
        
        chapter_regex = re.compile(r"^\s*(\d+)[:]\d+", re.IGNORECASE)
        chap_word_regex = re.compile(r"(?:Chapter|Chap\.)\s*(\d+)", re.IGNORECASE)
        book_pattern = "|".join(BIBLE_BOOKS)
        book_chapter_regex = re.compile(rf"({book_pattern})\s*(\d+)", re.IGNORECASE)

        for item in self.book_obj.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                filename = item.get_name()
                if filename in self._file_to_book:
                    # If we found a new book in the file mapping, save previous verse
                    if self.current_v_num is not None:
                        self._add_verse(self.current_book, self.current_chapter, self.current_v_num, "".join(self.current_v_text))
                        self.current_v_num = None
                        self.current_v_text = []
                    self.current_book = self._file_to_book[filename]
                    self.current_chapter = 0
                
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                
                for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'div']):
                    p_text = p.get_text().strip()
                    if not p_text: continue

                    m1 = chap_word_regex.search(p_text)
                    m2 = book_chapter_regex.search(p_text)
                    m3 = chapter_regex.search(p_text)
                    
                    if m2:
                        self.current_book = m2.group(1)
                        self.current_chapter = int(m2.group(2))
                    elif m1:
                        self.current_chapter = int(m1.group(1))
                    elif m3:
                        self.current_chapter = int(m3.group(1))

                    markers = p.find_all(['sup', 'b', 'strong', 'span'])
                    if not markers:
                        # If no markers but we are tracking a verse, append this whole paragraph's text
                        if self.current_v_num is not None:
                            self.current_v_text.append(p_text)
                        continue
                    
                    # If markers exist, iterate through descendants to find verse numbers
                    for child in p.descendants:
                        if isinstance(child, str):
                            if self.current_v_num is not None:
                                self.current_v_text.append(child)
                        elif child.name in ['sup', 'b', 'strong', 'span']:
                            text = child.get_text().strip()
                            if ":" in text:
                                parts = text.split(":")
                                if parts[0].isdigit():
                                    self.current_chapter = int(parts[0])
                                text = parts[-1]
                            
                            if text.isdigit():
                                if self.current_v_num is not None:
                                    self._add_verse(self.current_book, self.current_chapter, self.current_v_num, "".join(self.current_v_text))
                                
                                self.current_v_num = int(text)
                                self.current_v_text = []
                    
        # Add final verse
        if self.current_v_num is not None:
            self._add_verse(self.current_book, self.current_chapter, self.current_v_num, "".join(self.current_v_text))

        return list(self.verses_dict.values())

    def _add_verse(self, book, chapter, verse, text):
        if book == "Unknown" or chapter == 0:
            return
            
        text = text.strip()
        if not text or len(text) < 3: 
            return
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(rf'^{verse}\s*', '', text) 
        text = re.sub(rf'^{chapter}:\s*', '', text)

        ref_id = f"{book[:3].upper()}_{chapter:03}_{verse:03}"
        
        if ref_id in self.verses_dict:
            self.verses_dict[ref_id]['text'] += " " + text
        else:
            self.verses_dict[ref_id] = {
                "book": book,
                "chapter": chapter,
                "verse": verse,
                "text": text,
                "version": self.version_name,
                "ref_id": ref_id,
                "reference": f"{book} {chapter}:{verse}"
            }



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        parser = BibleEpubParser(sys.argv[1], "TEST")
        results = parser.parse()
        print(f"Parsed {len(results)} verses. Sample:")
        if results:
            for i in range(min(10, len(results))):
                print(results[i])
