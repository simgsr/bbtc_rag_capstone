"""Regression tests for the agent's SQL tool (``src/tools/sql_tool.py``).

The tool executes LLM-generated SQL, so the read-only connection + explicit
statement guards are the load-bearing security boundary here. These tests pin
that behaviour: write statements and multi-statement injections must be
rejected without touching the DB, the row cap must produce a truncation notice,
and the speaker-suggestion fallback must fire when a query returns nothing.
"""
import sqlite3

from src.storage.sqlite_store import SermonRegistry
from src.tools.sql_tool import make_sql_tool


def _make_tool(tmp_path):
    db = tmp_path / "sermons.db"
    SermonRegistry(db_path=str(db))  # creates the real schema + reference tables
    rows = [
        ("2024-01-07-walk-by-faith", "2024-01-07", 2024, "English", "SP Chua Seng Lee", "Walk by Faith", "indexed"),
        ("2024-01-14-living-in-hope", "2024-01-14", 2024, "English", "SP Daniel Foo", "Living in Hope", "indexed"),
        ("2023-12-31-watch-n-pray", "2023-12-31", 2023, "English", "SP Daniel Foo", "Watch and Pray", "indexed"),
    ]
    with sqlite3.connect(str(db)) as conn:
        conn.executemany(
            "INSERT INTO sermons(sermon_id, date, year, language, speaker, topic, status) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )
    return make_sql_tool(str(db)), db


def test_select_returns_rows(tmp_path):
    tool, _ = _make_tool(tmp_path)
    out = tool.invoke({"query": "SELECT topic, speaker FROM sermons WHERE year=2024 ORDER BY topic"})
    assert "Walk by Faith" in out
    assert "SP Daniel Foo" in out


def test_write_statements_rejected_and_db_untouched(tmp_path):
    tool, db = _make_tool(tmp_path)
    for bad in (
        "DELETE FROM sermons",
        "UPDATE sermons SET topic='x'",
        "DROP TABLE sermons",
        "INSERT INTO sermons(sermon_id) VALUES ('x')",
    ):
        out = tool.invoke({"query": bad})
        assert "SQL Error" in out and "only read-only SELECT / WITH" in out, bad
    with sqlite3.connect(str(db)) as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM sermons").fetchone()[0]
    assert remaining == 3


def test_multi_statement_injection_rejected(tmp_path):
    tool, db = _make_tool(tmp_path)
    out = tool.invoke({"query": "SELECT * FROM sermons; DROP TABLE sermons"})
    assert "SQL Error" in out and "multiple statements" in out
    with sqlite3.connect(str(db)) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sermons'"
        ).fetchall()
    assert tables  # the DROP after the semicolon must never execute


def test_trailing_semicolon_is_single_statement(tmp_path):
    tool, _ = _make_tool(tmp_path)
    out = tool.invoke({"query": "SELECT COUNT(*) FROM sermons;"})
    assert "(3,)" in out


def test_with_cte_is_allowed(tmp_path):
    tool, _ = _make_tool(tmp_path)
    out = tool.invoke(
        {"query": "WITH t AS (SELECT speaker FROM sermons) SELECT DISTINCT speaker FROM t"}
    )
    assert "SP Daniel Foo" in out


def test_row_cap_truncation_notice(tmp_path):
    tool, db = _make_tool(tmp_path)
    with sqlite3.connect(str(db)) as conn:
        conn.executemany(
            "INSERT INTO sermons(sermon_id, status) VALUES (?, 'indexed')",
            [(f"2024-01-{i:02d}-bulk",) for i in range(20, 225)],  # +205 → 208 total
        )
    out = tool.invoke({"query": "SELECT sermon_id FROM sermons"})
    assert "truncated at 200" in out
    row_lines = [ln for ln in out.splitlines() if ln.startswith("(")]
    assert len(row_lines) == 200


def test_speaker_suggestion_fallback(tmp_path):
    tool, _ = _make_tool(tmp_path)
    out = tool.invoke({"query": "SELECT speaker FROM sermons WHERE speaker LIKE 'Chua'"})
    assert "Did you mean" in out
    assert "SP Chua Seng Lee" in out
