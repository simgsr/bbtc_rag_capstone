# deploy/backend/tests/conftest.py
import sqlite3
import os
import pytest


@pytest.fixture(scope="session")
def test_db(tmp_path_factory):
    """Create a temp SQLite DB with 6 test sermons."""
    db_path = str(tmp_path_factory.mktemp("data") / "sermons.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE sermons (
                sermon_id TEXT PRIMARY KEY, filename TEXT, url TEXT UNIQUE,
                speaker TEXT, date TEXT, series TEXT, bible_book TEXT,
                primary_verse TEXT, language TEXT, file_type TEXT,
                year INTEGER, status TEXT, date_scraped TEXT
            )
        """)
        rows = [
            ("s1", "a.pdf", "http://a", "Pastor A", "2022-01-01", "S1", "John", "John 3:16", "English", "pdf", 2022, "indexed", "2026-01-01"),
            ("s2", "b.pdf", "http://b", "Pastor A", "2022-06-01", "S1", "Romans", "Romans 8:28", "English", "pdf", 2022, "indexed", "2026-01-01"),
            ("s3", "c.pdf", "http://c", "Pastor B", "2023-03-01", "S2", "John", "John 1:1", "English", "pdf", 2023, "indexed", "2026-01-01"),
            ("s4", "d.pdf", "http://d", "Pastor B", "2023-09-01", "S2", "Psalms", "Psalm 23:1", "English", "pdf", 2023, "indexed", "2026-01-01"),
            ("s5", "e.pdf", "http://e", "Pastor A", "2024-01-01", "S3", "John", "John 14:6", "English", "pdf", 2024, "indexed", "2026-01-01"),
            ("s6", "f.pdf", "http://f", "Pastor C", "2024-05-01", "S3", "Genesis", "Gen 1:1", "English", "pdf", 2024, "indexed", "2026-01-01"),
        ]
        conn.executemany("INSERT INTO sermons VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    return db_path
