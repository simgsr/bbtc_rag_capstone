import sqlite3
import os

DB_PATH = "data/sermons.db"

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    print(f"🔄 Migrating {DB_PATH} to add COLLATE NOCASE...")
    
    conn = sqlite3.connect(DB_PATH)
    try:
        # 1. Migrate sermons table
        print("  - Migrating sermons table...")
        conn.execute("ALTER TABLE sermons RENAME TO sermons_old")
        conn.execute("""
            CREATE TABLE sermons (
                sermon_id  TEXT PRIMARY KEY,
                date       TEXT,
                year       INTEGER,
                language   TEXT,
                speaker    TEXT COLLATE NOCASE,
                topic      TEXT COLLATE NOCASE,
                theme      TEXT COLLATE NOCASE,
                summary    TEXT,
                key_verse  TEXT,
                ng_file    TEXT,
                ps_file    TEXT,
                status     TEXT
            )
        """)
        conn.execute("""
            INSERT INTO sermons 
            SELECT * FROM sermons_old
        """)
        conn.execute("DROP TABLE sermons_old")

        # 2. Migrate verses table
        print("  - Migrating verses table...")
        conn.execute("ALTER TABLE verses RENAME TO verses_old")
        conn.execute("""
            CREATE TABLE verses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                sermon_id   TEXT REFERENCES sermons(sermon_id),
                verse_ref   TEXT COLLATE NOCASE,
                book        TEXT COLLATE NOCASE,
                chapter     INTEGER,
                verse_start INTEGER,
                verse_end   INTEGER,
                is_key_verse INTEGER DEFAULT 0,
                UNIQUE(sermon_id, verse_ref)
            )
        """)
        conn.execute("""
            INSERT INTO verses (id, sermon_id, verse_ref, book, chapter, verse_start, verse_end, is_key_verse)
            SELECT id, sermon_id, verse_ref, book, chapter, verse_start, verse_end, is_key_verse FROM verses_old
        """)
        conn.execute("DROP TABLE verses_old")

        # 3. Re-create indexes
        print("  - Re-creating indexes...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_verses_sermon ON verses(sermon_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sermons_year ON sermons(year)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sermons_speaker ON sermons(speaker)")

        conn.commit()
        print("✅ Migration successful!")
    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
