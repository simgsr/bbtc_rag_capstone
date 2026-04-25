import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sqlite3
import os
import uuid
from langchain_core.tools import tool


def make_matplotlib_tool(registry):
    db_path = registry.db_path

    @tool
    def matplotlib_tool(chart_name: str) -> str:
        """Generates a chart from live sermon data and returns the PNG file path.
        Supported chart_name values:
        - 'sermons_per_speaker' — bar chart of sermon count per speaker (top 10)
        - 'sermons_per_year' — bar chart of sermon count per year
        - 'top_bible_books' — bar chart of most-preached Bible books (top 10)
        Returns the file path to the saved PNG."""
        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            with sqlite3.connect(db_path) as conn:
                if chart_name == "sermons_per_speaker":
                    rows = conn.execute(
                        "SELECT speaker, COUNT(*) as n FROM sermons "
                        "WHERE speaker IS NOT NULL AND speaker != '' "
                        "GROUP BY speaker ORDER BY n DESC LIMIT 10"
                    ).fetchall()
                    if not rows:
                        plt.close(fig)
                        return "No sermon data found."
                    labels, values = zip(*rows)
                    ax.barh(labels, values, color="#3b82f6")
                    ax.set_xlabel("Number of Sermons")
                    ax.set_title("Top 10 Speakers by Sermon Count")
                    ax.invert_yaxis()

                elif chart_name == "sermons_per_year":
                    rows = conn.execute(
                        "SELECT year, COUNT(*) as n FROM sermons "
                        "WHERE year IS NOT NULL "
                        "GROUP BY year ORDER BY year"
                    ).fetchall()
                    if not rows:
                        plt.close(fig)
                        return "No sermon data found."
                    labels, values = zip(*rows)
                    ax.bar([str(y) for y in labels], values, color="#6366f1")
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Number of Sermons")
                    ax.set_title("Sermons per Year")

                elif chart_name == "top_bible_books":
                    rows = conn.execute(
                        "SELECT bible_book, COUNT(*) as n FROM sermons "
                        "WHERE bible_book IS NOT NULL AND bible_book != '' "
                        "GROUP BY bible_book ORDER BY n DESC LIMIT 10"
                    ).fetchall()
                    if not rows:
                        plt.close(fig)
                        return "No sermon data found."
                    labels, values = zip(*rows)
                    ax.barh(labels, values, color="#10b981")
                    ax.set_xlabel("Number of Sermons")
                    ax.set_title("Top 10 Preached Bible Books")
                    ax.invert_yaxis()

                else:
                    plt.close(fig)
                    return (
                        f"Unknown chart '{chart_name}'. "
                        "Valid options: sermons_per_speaker, sermons_per_year, top_bible_books."
                    )

        except Exception as e:
            plt.close(fig)
            return f"Chart generation error: {e}"

        plt.tight_layout()
        file_path = os.path.join("/tmp", f"bbtc_chart_{uuid.uuid4().hex[:8]}.png")
        try:
            fig.savefig(file_path)
        except Exception as e:
            plt.close(fig)
            return f"Chart save error: {e}"
        plt.close(fig)
        return file_path

    return matplotlib_tool
