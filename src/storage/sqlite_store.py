import sqlite3, os
from src.storage.normalize_speaker import normalize_speaker
from src.storage.normalize_book import normalize_book, disambiguate_book, BOOK_MAP
from src.ingestion.ps_extractor import normalize_verse_ref

_BIBLE_BOOKS = [
    ("Genesis", "OT", 1), ("Exodus", "OT", 2), ("Leviticus", "OT", 3),
    ("Numbers", "OT", 4), ("Deuteronomy", "OT", 5), ("Joshua", "OT", 6),
    ("Judges", "OT", 7), ("Ruth", "OT", 8), ("1 Samuel", "OT", 9),
    ("2 Samuel", "OT", 10), ("1 Kings", "OT", 11), ("2 Kings", "OT", 12),
    ("1 Chronicles", "OT", 13), ("2 Chronicles", "OT", 14), ("Ezra", "OT", 15),
    ("Nehemiah", "OT", 16), ("Esther", "OT", 17), ("Job", "OT", 18),
    ("Psalms", "OT", 19), ("Proverbs", "OT", 20), ("Ecclesiastes", "OT", 21),
    ("Song of Songs", "OT", 22), ("Isaiah", "OT", 23), ("Jeremiah", "OT", 24),
    ("Lamentations", "OT", 25), ("Ezekiel", "OT", 26), ("Daniel", "OT", 27),
    ("Hosea", "OT", 28), ("Joel", "OT", 29), ("Amos", "OT", 30),
    ("Obadiah", "OT", 31), ("Jonah", "OT", 32), ("Micah", "OT", 33),
    ("Nahum", "OT", 34), ("Habakkuk", "OT", 35), ("Zephaniah", "OT", 36),
    ("Haggai", "OT", 37), ("Zechariah", "OT", 38), ("Malachi", "OT", 39),
    ("Matthew", "NT", 40), ("Mark", "NT", 41), ("Luke", "NT", 42),
    ("John", "NT", 43), ("Acts", "NT", 44), ("Romans", "NT", 45),
    ("1 Corinthians", "NT", 46), ("2 Corinthians", "NT", 47), ("Galatians", "NT", 48),
    ("Ephesians", "NT", 49), ("Philippians", "NT", 50), ("Colossians", "NT", 51),
    ("1 Thessalonians", "NT", 52), ("2 Thessalonians", "NT", 53), ("1 Timothy", "NT", 54),
    ("2 Timothy", "NT", 55), ("Titus", "NT", 56), ("Philemon", "NT", 57),
    ("Hebrews", "NT", 58), ("James", "NT", 59), ("1 Peter", "NT", 60),
    ("2 Peter", "NT", 61), ("1 John", "NT", 62), ("2 John", "NT", 63),
    ("3 John", "NT", 64), ("Jude", "NT", 65), ("Revelation", "NT", 66),
]


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
                    is_key_verse INTEGER DEFAULT 0,
                    UNIQUE(sermon_id, verse_ref)
                );
                CREATE TABLE IF NOT EXISTS bible_books (
                    book_name  TEXT PRIMARY KEY,
                    testament  TEXT,
                    book_order INTEGER
                );
                CREATE TABLE IF NOT EXISTS book_aliases (
                    alias     TEXT PRIMARY KEY,
                    canonical TEXT REFERENCES bible_books(book_name)
                );
                CREATE INDEX IF NOT EXISTS idx_verses_sermon ON verses(sermon_id);
                CREATE INDEX IF NOT EXISTS idx_sermons_year ON sermons(year);
                CREATE INDEX IF NOT EXISTS idx_sermons_speaker ON sermons(speaker);
            """)
        self._seed_reference_tables()

    def _seed_reference_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO bible_books(book_name, testament, book_order) VALUES (?,?,?)",
                _BIBLE_BOOKS,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO book_aliases(alias, canonical) VALUES (?,?)",
                [(alias, canonical) for alias, canonical in BOOK_MAP.items()],
            )

    def upsert_sermon(self, record: dict):
        if record.get("speaker"):
            record["speaker"] = normalize_speaker(record["speaker"])
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
        record = dict(record)
        canonical = normalize_book(record.get("book"))
        if canonical is None:
            canonical = disambiguate_book(record.get("book"), record.get("chapter"))
        if canonical is None:
            return
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
                "SELECT * FROM sermons WHERE status IS NULL OR status NOT IN ('indexed', 'failed')"
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

    def delete_verses(self, sermon_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM verses WHERE sermon_id = ?", (sermon_id,))

    def wipe(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                "DROP TABLE IF EXISTS verses; DROP TABLE IF EXISTS sermons;"
            )
        self._init_db()
