import sqlite3
from langchain_core.tools import tool

def make_sql_tool(registry):
    db_path = registry.db_path

    @tool
    def sql_query_tool(query: str) -> str:
        """Executes a SQL query against the sermons SQLite database.
        Schema: sermons(sermon_id TEXT, filename TEXT, url TEXT, speaker TEXT,
        date TEXT YYYY-MM-DD, series TEXT, bible_book TEXT, primary_verse TEXT,
        language TEXT, file_type TEXT, year INTEGER, status TEXT, date_scraped TEXT).
        Returns up to 50 rows. Use COUNT(), GROUP BY, ORDER BY as needed."""
        print(f"DEBUG: Executing SQL: {query}")
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(query)
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                if not rows:
                    return "No results found."
                
                # Format as a string
                result = f"Columns: {', '.join(columns)}\n"
                for row in rows:
                    result += f"{row}\n"
                return result
        except Exception as e:
            schema = (
                "sermons("
                "sermon_id TEXT PRIMARY KEY, "
                "filename TEXT, "
                "url TEXT, "
                "speaker TEXT, "
                "date TEXT (YYYY-MM-DD), "
                "series TEXT, "
                "bible_book TEXT, "
                "primary_verse TEXT, "
                "language TEXT ('English'|'Mandarin'), "
                "file_type TEXT (pdf|pptx|docx), "
                "year INTEGER, "
                "status TEXT (extracted|indexed|failed), "
                "date_scraped TEXT"
                ")"
            )
            return f"SQL Error: {str(e)}. Full schema: {schema}"

    return sql_query_tool
