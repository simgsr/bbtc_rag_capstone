import os
import sqlite3
import pytest
from unittest.mock import MagicMock
from src.tools.matplotlib_tool import make_matplotlib_tool


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "sermons.db")
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE sermons (year INTEGER, speaker TEXT, bible_book TEXT)"
        )
        conn.executemany(
            "INSERT INTO sermons VALUES (?, ?, ?)",
            [
                (2022, "Pastor A", "Romans"),
                (2022, "Pastor A", "John"),
                (2023, "Pastor B", "Psalms"),
                (2023, "Pastor A", "Genesis"),
                (2024, "Pastor B", "Romans"),
            ],
        )
    return path


@pytest.fixture
def chart_tool(db_path):
    registry = MagicMock()
    registry.db_path = db_path
    return make_matplotlib_tool(registry)


def test_sermons_scatter_returns_png_path(chart_tool):
    result = chart_tool.invoke({"chart_name": "sermons_scatter"})
    assert result.endswith(".png"), f"Expected a PNG path, got: {result}"
    assert os.path.exists(result)


def test_sermons_scatter_file_is_nonempty(chart_tool):
    result = chart_tool.invoke({"chart_name": "sermons_scatter"})
    assert os.path.getsize(result) > 0


def test_sermons_scatter_empty_db_returns_message(tmp_path):
    path = str(tmp_path / "empty.db")
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE sermons (year INTEGER, speaker TEXT, bible_book TEXT)"
        )
    registry = MagicMock()
    registry.db_path = path
    tool = make_matplotlib_tool(registry)
    result = tool.invoke({"chart_name": "sermons_scatter"})
    assert "No sermon data" in result


def test_unknown_chart_name_returns_error(chart_tool):
    result = chart_tool.invoke({"chart_name": "unknown_chart"})
    assert "Unknown chart" in result
