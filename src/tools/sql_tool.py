"""Agent tool: raw SQL over the sermon database (``data/sermons.db``).

``make_sql_tool(db_path)`` returns ``sql_query_tool``, the agent's structured-
query tool — used for counts, lists, date ranges, speaker stats, verse
aggregations, and gap/coverage analysis (anti-join against the ``bible_books``
reference table). The tool's docstring is the schema/example contract the LLM
reads, so keep it accurate: it documents every table (including ``bible_books`` /
``book_aliases``) and steers the model to anti-join rather than recall the
66-book canon from memory.

Results are capped at 200 rows; when a result hits exactly 200 the tool appends
an explicit truncation notice so the model never silently reasons over a partial
set. Errors return the schema so the model can self-correct and retry.
"""
import re
import sqlite3
from langchain_core.tools import tool

# Schema hint shown to the model on a rejected statement or a query error, so it
# can self-correct and retry (single source for the guard + error paths).
_SCHEMA_HINT = (
    "Tables:\n"
    "  sermons(sermon_id, date, year, language, speaker, topic, theme, summary, key_verse, ng_file, ps_file, status)\n"
    "  verses(id, sermon_id, verse_ref, book, chapter, verse_start, verse_end, is_key_verse)"
)


def make_sql_tool(db_path: str):

    @tool
    def sql_query_tool(query: str) -> str:
        """Executes a SQL query against the BBTC sermon database.

        Schema:
        - sermons(sermon_id, date, year, language, speaker, topic, theme,
                  summary, key_verse, ng_file, ps_file, status)
        - verses(id, sermon_id, verse_ref, book, chapter, verse_start, verse_end, is_key_verse)
        - bible_books(book_name, testament, book_order)  -- reference: all 66 canonical books (OT/NT), in canonical order
        - book_aliases(alias, canonical)                 -- reference: maps raw book spellings to canonical book_name

        Common queries:
        - List speakers: SELECT DISTINCT speaker FROM sermons WHERE speaker IS NOT NULL ORDER BY speaker
        - Speakers in 2023: SELECT speaker, COUNT(*) as n FROM sermons WHERE year=2023 GROUP BY speaker ORDER BY n DESC
        - Most preached book: SELECT book, COUNT(*) as n FROM verses GROUP BY book ORDER BY n DESC LIMIT 10
        - Verses by speaker: SELECT v.verse_ref, COUNT(*) as n FROM verses v JOIN sermons s USING(sermon_id) WHERE s.speaker LIKE '%Chua%' GROUP BY v.verse_ref ORDER BY n DESC
        - Key verses: SELECT key_verse, speaker, date FROM sermons WHERE key_verse IS NOT NULL ORDER BY date DESC
        - Gap analysis (books NEVER preached): SELECT book_name FROM bible_books WHERE book_name NOT IN (SELECT DISTINCT book FROM verses) ORDER BY book_order
        - Coverage by testament: SELECT b.testament, COUNT(DISTINCT v.book) as preached, COUNT(DISTINCT b.book_name) as total FROM bible_books b LEFT JOIN verses v ON b.book_name=v.book GROUP BY b.testament

        For "which X are missing / never used / not covered" questions, use an anti-join
        against the reference table (bible_books) with NOT IN / LEFT JOIN — do NOT try to
        recall the full 66-book list yourself.

        Returns up to 200 rows."""
        try:
            # Guard 1 — read-only statements only. Allowlisting the first keyword
            # before the statement reaches sqlite is defense in depth: the safety
            # must not rest solely on Python sqlite3's single-statement behaviour.
            stmt = query.strip()
            if stmt.endswith(";"):
                stmt = stmt[:-1]  # a trailing semicolon is still one statement
            first_kw = stmt.split(None, 1)[0].upper() if stmt else ""
            if first_kw not in ("SELECT", "WITH"):
                return (
                    f"SQL Error: only read-only SELECT / WITH statements are allowed "
                    f"(first keyword '{first_kw or '<empty>'}'). Retry with a SELECT.\n"
                    f"{_SCHEMA_HINT}"
                )
            # Guard 2 — a single statement per call. sqlite3.execute also enforces
            # this, but reject it explicitly so the model hears the reason and
            # splits its queries into separate calls.
            if ";" in stmt:
                return (
                    "SQL Error: multiple statements are not allowed — pass one "
                    f"query at a time.\n{_SCHEMA_HINT}"
                )
            # Open read-only. `query` is LLM-generated, and the LLM can be
            # prompt-injected via sermon content (stored PDF/summary text) or
            # simply hallucinate — a read-write connection would let a stray
            # `DROP TABLE` / `DELETE` / `UPDATE` destroy the archive. Every
            # documented tool query is a SELECT, so mode=ro breaks nothing and
            # turns a data-loss footgun into a harmless "attempt to write a
            # readonly database" error. uri=True is required for query-string
            # connection params.
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                # Extra belt: query_only refuses writes at the engine level even if
                # the mode=ro URI were ever dropped. (A WITH ... DELETE that slips
                # past Guard 1 is caught here.)
                conn.execute("PRAGMA query_only = ON")
                cursor = conn.execute(stmt)
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchmany(200)
                if not rows:
                    # Fallback logic for speaker suggestions
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
                        except sqlite3.Error:
                            pass  # suggestion lookup is best-effort; fall through to default
                    return "No results found."
                result = "Columns: " + ", ".join(columns) + "\n"
                for row in rows:
                    result += str(row) + "\n"
                if len(rows) == 200:
                    result += (
                        "[Note: results truncated at 200 rows — the full result set may be "
                        "larger. Use COUNT/GROUP BY or add filters for complete data.]\n"
                    )
                return result
        except Exception as e:
            return f"SQL Error: {e}\n{_SCHEMA_HINT}"

    return sql_query_tool
