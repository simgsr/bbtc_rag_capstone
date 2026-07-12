"""Pure helper functions for the Gradio UI (``app.py``).

Kept separate from ``app.py`` so they can be unit-tested without importing the
whole app / Gradio (see ``tests/test_ui_helpers.py``):

  * ``extract_chart_path`` — pull a ``viz_tool`` chart file path out of the
    agent's reply text and return the cleaned message + path.
  * ``fetch_archive_stats`` / ``render_stats_bar`` — read live sermon/speaker/
    year/language counts from SQLite and render the header stats pill bar.
"""
import re
import sqlite3


def extract_chart_path(response: str) -> tuple[str, str | None]:
    """Extract a chart file path from agent response text.
    Returns (cleaned_text, chart_path) or (original_text, None) if no path found.
    Supports .png (legacy) and .json (plotly).
    """
    match = re.search(r'/tmp/bbtc_chart_[a-f0-9]+\.(png|json)', response)
    if match is None:
        return response, None
    chart_path = match.group(0)
    cleaned = (response[:match.start()] + response[match.end():]).strip().rstrip(':').strip()
    if not cleaned:
        cleaned = "Here is the interactive chart:" if chart_path.endswith('.json') else "Here is the chart:"
    return cleaned, chart_path


def fetch_archive_stats(db_path: str) -> dict | None:
    """Fetch live archive counts from SQLite.
    Returns dict with keys: sermons, speakers, year_min, year_max, languages.
    Returns None if the DB is unavailable.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            sermon_count = conn.execute("SELECT COUNT(*) FROM sermons").fetchone()[0]
            speaker_count = conn.execute(
                "SELECT COUNT(DISTINCT speaker) FROM sermons "
                "WHERE speaker IS NOT NULL AND speaker != ''"
            ).fetchone()[0]
            year_row = conn.execute(
                "SELECT MIN(year), MAX(year) FROM sermons WHERE year IS NOT NULL"
            ).fetchone()
            lang_count = conn.execute(
                "SELECT COUNT(DISTINCT language) FROM sermons "
                "WHERE language IS NOT NULL AND language != ''"
            ).fetchone()[0]
            return {
                "sermons": sermon_count,
                "speakers": speaker_count,
                "year_min": year_row[0],
                "year_max": year_row[1],
                "languages": lang_count,
            }
    except Exception:
        return None


def render_stats_bar(stats: dict | None) -> str:
    """Render archive stats as an HTML pill bar for use in gr.HTML."""
    if stats is None:
        return "<div class='stats-bar'>📚 Archive stats unavailable</div>"
    year_range = (
        f"{stats['year_min']} – {stats['year_max']}"
        if stats["year_min"] is not None and stats["year_max"] is not None
        else "N/A"
    )
    return (
        f"<div class='stats-bar'>"
        f"📚 {stats['sermons']} sermons &nbsp;·&nbsp; "
        f"👤 {stats['speakers']} speakers &nbsp;·&nbsp; "
        f"📅 {year_range} &nbsp;·&nbsp; "
        f"🌐 {stats['languages']} languages"
        f"</div>"
    )
