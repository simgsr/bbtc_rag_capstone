"""Regression tests for the agent's chart tool (``src/tools/viz_tool.py``).

``viz_tool`` is the one agent surface that interpolates an LLM-supplied argument
(``top_n``) into a SQL ``LIMIT``, so these tests pin the two layers of defense:
an injection-shaped ``top_n`` is rejected by the tool's argument schema before
the function body runs, and an int-parseable but out-of-range value is clamped
to [1, 100] in the body — either way it never reaches the DB un-clamped and the
tool never crashes. Charts must render from the SQLite data and the tool must
not open the DB read-write (verified via a read-only directory: a read-write
open would need to create journal files and fail).
"""
import os
import sqlite3

import pytest
from pydantic import ValidationError

from src.tools.viz_tool import make_viz_tool


class _FakeRegistry:
    def __init__(self, db_path):
        self.db_path = db_path


def _make_registry(tmp_path):
    db = tmp_path / "viz.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "CREATE TABLE sermons (sermon_id TEXT, date TEXT, year INTEGER, language TEXT, "
            "speaker TEXT, topic TEXT, theme TEXT, summary TEXT, key_verse TEXT, "
            "ng_file TEXT, ps_file TEXT, status TEXT)"
        )
        conn.execute(
            "CREATE TABLE verses (id INTEGER PRIMARY KEY, sermon_id TEXT, verse_ref TEXT, "
            "book TEXT, chapter INTEGER, verse_start INTEGER, verse_end INTEGER, is_key_verse INTEGER)"
        )
        conn.executemany(
            "INSERT INTO sermons(sermon_id, date, year, language, speaker, status) VALUES (?,?,?,?,?,?)",
            [
                ("2024-01-07-faith", "2024-01-07", 2024, "English", "SP Daniel Foo", "indexed"),
                ("2024-01-14-hope", "2024-01-14", 2024, "English", "SP Chua Seng Lee", "indexed"),
                ("2024-01-21-love", "2024-01-21", 2024, "English", "SP Daniel Foo", "indexed"),
            ],
        )
        conn.execute(
            "INSERT INTO verses(sermon_id, verse_ref, book, chapter, verse_start, verse_end, is_key_verse) "
            "VALUES (?,?,?,?,?,?,?)",
            ("2024-01-07-faith", "John 3:16", "John", 3, 16, 16, 1),
        )
    return _FakeRegistry(str(db))


def _chart_path(out: str) -> str | None:
    for token in out.split():
        if token.startswith("/tmp/bbtc_chart_"):
            return token
    return None


def _invoke_cleanup(tool, args, tmp_path):
    """Invoke the tool, assert a chart rendered, and clean up the temp chart file."""
    out = tool.invoke(args)
    assert "Chart generation error" not in out, out
    path = _chart_path(out)
    assert path is not None, out
    assert path.startswith("/tmp/bbtc_chart_")
    assert os.path.exists(path)
    try:
        os.remove(path)
    except OSError:
        pass
    return out


def test_generates_chart_with_default_top_n(tmp_path):
    tool = make_viz_tool(_make_registry(tmp_path))
    _invoke_cleanup(tool, {"chart_name": "sermons_per_speaker"}, tmp_path)


def test_top_n_accepts_int_string(tmp_path):
    tool = make_viz_tool(_make_registry(tmp_path))
    _invoke_cleanup(tool, {"chart_name": "sermons_per_speaker", "top_n": "2"}, tmp_path)


def test_top_n_injection_shaped_values_rejected_at_boundary(tmp_path):
    """Injection-shaped `top_n` is refused by the tool's argument schema BEFORE
    the function body runs — the string never reaches the ``LIMIT {top_n}``
    interpolation, so it can't become SQL."""
    tool = make_viz_tool(_make_registry(tmp_path))
    for bad in ("1 OR 1=1", "-5 UNION SELECT 1", "abc", None):
        with pytest.raises(ValidationError):
            tool.invoke({"chart_name": "sermons_per_speaker", "top_n": bad})


def test_top_n_out_of_range_int_is_clamped(tmp_path):
    """An int-parseable `top_n` outside [1, 100] reaches the function body and is
    clamped there (the second layer of defense, after schema validation) — the
    chart still renders, never a ValueError or raw SQL."""
    tool = make_viz_tool(_make_registry(tmp_path))
    _invoke_cleanup(tool, {"chart_name": "sermons_per_speaker", "top_n": "999"}, tmp_path)


def test_unknown_chart_returns_error(tmp_path):
    tool = make_viz_tool(_make_registry(tmp_path))
    out = tool.invoke({"chart_name": "not_a_chart"})
    assert "Unknown chart" in out
    assert "sermons_per_speaker" in out


def test_db_opened_read_only(tmp_path):
    """The tool opens the DB with mode=ro, so it must work (and write no
    journal) even when the DB directory is read-only. A read-write open would
    fail to create its journal/lock files on a r-x directory."""
    import stat
    reg = _make_registry(tmp_path)
    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)  # r-x: no writes allowed
    try:
        tool = make_viz_tool(reg)
        out = tool.invoke({"chart_name": "sermons_per_speaker"})
        path = _chart_path(out)
        assert path is not None, out
        assert os.path.exists(path)
        try:
            os.remove(path)  # charts live in /tmp, still writable
        except OSError:
            pass
    finally:
        os.chmod(tmp_path, stat.S_IRWXU)
