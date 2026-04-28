import sqlite3, os
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
                    is_key_verse INTEGER DEFAULT 0,
                    UNIQUE(sermon_id, verse_ref)
                );
                CREATE INDEX IF NOT EXISTS idx_verses_sermon ON verses(sermon_id);
                CREATE INDEX IF NOT EXISTS idx_sermons_year ON sermons(year);
                CREATE INDEX IF NOT EXISTS idx_sermons_speaker ON sermons(speaker);
            """)

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

    def wipe(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                "DROP TABLE IF EXISTS verses; DROP TABLE IF EXISTS sermons;"
            )
        self._init_db()
