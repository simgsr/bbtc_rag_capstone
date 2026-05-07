import sqlite3
from langchain_core.tools import tool


def make_sql_tool(db_path: str):

    @tool
    def sql_query_tool(query: str) -> str:
        """Executes a SQL query against the BBTC sermon database.

        Schema:
        - sermons(sermon_id, date, year, language, speaker, topic, theme,
                  summary, key_verse, ng_file, ps_file, status)
        - verses(id, sermon_id, verse_ref, book, chapter, verse_start, verse_end, is_key_verse)

        Common queries:
        - List speakers: SELECT DISTINCT speaker FROM sermons WHERE speaker IS NOT NULL ORDER BY speaker
        - Speakers in 2023: SELECT speaker, COUNT(*) as n FROM sermons WHERE year=2023 GROUP BY speaker ORDER BY n DESC
        - Most preached book: SELECT book, COUNT(*) as n FROM verses GROUP BY book ORDER BY n DESC LIMIT 10
        - Verses by speaker: SELECT v.verse_ref, COUNT(*) as n FROM verses v JOIN sermons s USING(sermon_id) WHERE s.speaker LIKE '%Chua%' GROUP BY v.verse_ref ORDER BY n DESC
        - Key verses: SELECT key_verse, speaker, date FROM sermons WHERE key_verse IS NOT NULL ORDER BY date DESC

        Returns up to 50 rows."""
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(query)
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchmany(50)
                if not rows:
                    # Fallback logic for speaker suggestions
                    import re
                    match = re.search(r"speaker\s*(?:LIKE|=)\s*['\"]%?([^'\"]+?)%?['\"]", query, re.IGNORECASE)
                    if match:
                        name = match.group(1)
                        # Try to find similar speakers
                        try:
                            alt_cursor = conn.execute(
                                "SELECT DISTINCT speaker FROM sermons WHERE speaker LIKE ? LIMIT 5",
                                (f"%{name}%",)
                            )
                            suggestions = [r[0] for r in alt_cursor.fetchall() if r[0]]
                            if suggestions:
                                return (
                                    f"No results found for '{name}'. "
                                    f"Did you mean one of these speakers: {', '.join(suggestions)}?"
                                )
                        except:
                            pass
                    return "No results found."
                result = "Columns: " + ", ".join(columns) + "\n"
                for row in rows:
                    result += str(row) + "\n"
                return result
        except Exception as e:
            return (
                f"SQL Error: {e}\n"
                "Tables:\n"
                "  sermons(sermon_id, date, year, language, speaker, topic, theme, summary, key_verse, ng_file, ps_file, status)\n"
                "  verses(id, sermon_id, verse_ref, book, chapter, verse_start, verse_end, is_key_verse)"
            )

    return sql_query_tool
