import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import sqlite3
import os
import uuid
from langchain_core.tools import tool

pio.templates.default = "plotly_dark"

_FONT = "'Source Code Pro', 'Courier New', monospace"
_BG = "rgba(10, 15, 30, 0.0)"
_PLOT_BG = "rgba(15, 22, 42, 0.55)"
_GRID = "rgba(255, 255, 255, 0.06)"
_TEXT = "#cbd5e1"
_TITLE_COLOR = "#93c5fd"
_ACCENT = "#6366f1"


def _base_layout(title: str, left_margin: int = 60) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(family=_FONT, size=15, color=_TITLE_COLOR),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor=_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(family=_FONT, size=11, color=_TEXT),
        margin=dict(l=left_margin, r=24, t=56, b=56),
        width=720,
        height=520,
        coloraxis=dict(
            colorbar=dict(
                title=dict(text="Count", font=dict(family=_FONT, size=11, color=_TEXT)),
                tickfont=dict(family=_FONT, size=10, color=_TEXT),
                thickness=12,
                len=0.7,
                outlinewidth=0,
                bgcolor="rgba(0,0,0,0)",
            )
        ),
        xaxis=dict(
            tickfont=dict(family=_FONT, size=10, color=_TEXT),
            title_font=dict(family=_FONT, size=11, color=_TEXT),
            gridcolor=_GRID,
            zeroline=False,
            tickangle=0,
            automargin=True,
        ),
        yaxis=dict(
            tickfont=dict(family=_FONT, size=10, color=_TEXT),
            title_font=dict(family=_FONT, size=11, color=_TEXT),
            gridcolor=_GRID,
            zeroline=False,
            automargin=True,
        ),
    )


def make_viz_tool(registry):
    db_path = registry.db_path

    @tool
    def viz_tool(chart_name: str) -> str:
        """Generates an interactive Plotly chart from live sermon data and returns the JSON file path.
        Supported chart_name values:
        - 'sermons_per_speaker' — horizontal bar chart of sermon count per speaker (top 15)
        - 'sermons_per_year'    — bar chart of sermon count per year (2015–present)
        - 'verses_per_book'     — horizontal bar chart of most-preached Bible books (top 15)
        - 'sermons_scatter'     — bubble chart of sermon count by speaker and year
        Returns the file path to the saved Plotly JSON."""

        try:
            with sqlite3.connect(db_path) as conn:

                if chart_name == "sermons_per_speaker":
                    rows = conn.execute(
                        "SELECT speaker, COUNT(*) as n FROM sermons "
                        "WHERE speaker IS NOT NULL AND speaker != '' "
                        "GROUP BY speaker ORDER BY n DESC LIMIT 15"
                    ).fetchall()
                    if not rows:
                        return "No sermon data found."

                    speakers, counts = zip(*rows)
                    fig = go.Figure(go.Bar(
                        x=list(counts),
                        y=list(speakers),
                        orientation="h",
                        marker=dict(
                            color=list(counts),
                            colorscale="Blues",
                            showscale=True,
                            colorbar=dict(
                                title=dict(text="Sermons", font=dict(family=_FONT, size=11, color=_TEXT)),
                                tickfont=dict(family=_FONT, size=10, color=_TEXT),
                                thickness=12,
                                len=0.7,
                                outlinewidth=0,
                            ),
                            line=dict(width=0),
                        ),
                        hovertemplate="<b>%{y}</b><br>%{x} sermons<extra></extra>",
                    ))
                    layout = _base_layout("Top 15 Speakers by Sermon Count", left_margin=170)
                    layout["yaxis"]["categoryorder"] = "total ascending"
                    layout["xaxis"]["title"] = "Number of Sermons"
                    layout["yaxis"]["title"] = ""
                    fig.update_layout(**layout)

                elif chart_name == "sermons_per_year":
                    rows = conn.execute(
                        "SELECT year, COUNT(*) as n FROM sermons "
                        "WHERE year >= 2015 "
                        "GROUP BY year ORDER BY year"
                    ).fetchall()
                    if not rows:
                        return "No sermon data found."

                    years, counts = zip(*rows)
                    year_labels = [str(y) for y in years]
                    fig = go.Figure(go.Bar(
                        x=year_labels,
                        y=list(counts),
                        marker=dict(
                            color=list(counts),
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(
                                title=dict(text="Sermons", font=dict(family=_FONT, size=11, color=_TEXT)),
                                tickfont=dict(family=_FONT, size=10, color=_TEXT),
                                thickness=12,
                                len=0.7,
                                outlinewidth=0,
                            ),
                            line=dict(width=0),
                        ),
                        hovertemplate="<b>%{x}</b><br>%{y} sermons<extra></extra>",
                    ))
                    layout = _base_layout("Sermons per Year", left_margin=56)
                    layout["xaxis"]["title"] = "Year"
                    layout["xaxis"]["type"] = "category"
                    layout["xaxis"]["tickangle"] = -45
                    layout["yaxis"]["title"] = "Sermons"
                    fig.update_layout(**layout)

                elif chart_name == "verses_per_book":
                    rows = conn.execute(
                        "SELECT book, COUNT(*) as n FROM verses "
                        "WHERE book IS NOT NULL AND book != '' "
                        "GROUP BY book ORDER BY n DESC LIMIT 15"
                    ).fetchall()
                    if not rows:
                        return "No verse data found. Run ingest.py first."

                    books, counts = zip(*rows)
                    fig = go.Figure(go.Bar(
                        x=list(counts),
                        y=list(books),
                        orientation="h",
                        marker=dict(
                            color=list(counts),
                            colorscale="Teal",
                            showscale=True,
                            colorbar=dict(
                                title=dict(text="References", font=dict(family=_FONT, size=11, color=_TEXT)),
                                tickfont=dict(family=_FONT, size=10, color=_TEXT),
                                thickness=12,
                                len=0.7,
                                outlinewidth=0,
                            ),
                            line=dict(width=0),
                        ),
                        hovertemplate="<b>%{y}</b><br>%{x} references<extra></extra>",
                    ))
                    layout = _base_layout("Top 15 Preached Bible Books", left_margin=110)
                    layout["yaxis"]["categoryorder"] = "total ascending"
                    layout["xaxis"]["title"] = "Times Referenced"
                    layout["yaxis"]["title"] = ""
                    fig.update_layout(**layout)

                elif chart_name == "sermons_scatter":
                    rows = conn.execute(
                        "SELECT COALESCE(year, CAST(SUBSTR(date, 1, 4) AS INTEGER)) as yr, "
                        "speaker, COUNT(*) as n FROM sermons "
                        "WHERE (year >= 2015 OR (year IS NULL AND SUBSTR(date, 1, 4) >= '2015')) "
                        "AND speaker IS NOT NULL AND speaker != '' "
                        "GROUP BY yr, speaker ORDER BY yr"
                    ).fetchall()
                    if not rows:
                        return "No sermon data found."

                    years_str = [str(r[0]) for r in rows]
                    speakers = [r[1] for r in rows]
                    counts = [r[2] for r in rows]
                    all_years = sorted({str(r[0]) for r in rows})

                    fig = go.Figure(go.Scatter(
                        x=years_str,
                        y=speakers,
                        mode="markers",
                        marker=dict(
                            size=[max(8, c * 5) for c in counts],
                            sizemode="area",
                            sizeref=2.0 * max(counts) / (28 ** 2),
                            color=counts,
                            colorscale="Plasma",
                            showscale=True,
                            colorbar=dict(
                                title=dict(text="Sermons", font=dict(family=_FONT, size=11, color=_TEXT)),
                                tickfont=dict(family=_FONT, size=10, color=_TEXT),
                                thickness=12,
                                len=0.7,
                                outlinewidth=0,
                            ),
                            line=dict(width=0),
                            opacity=0.85,
                        ),
                        hovertemplate="<b>%{y}</b> · %{x}<br>%{marker.color} sermons<extra></extra>",
                    ))
                    layout = _base_layout("Sermon Activity by Speaker & Year", left_margin=170)
                    layout["xaxis"].update(
                        type="category",
                        tickmode="array",
                        tickvals=all_years,
                        tickangle=-45,
                        title="Year",
                    )
                    layout["yaxis"]["title"] = ""
                    layout["height"] = 640
                    layout["width"] = 720
                    fig.update_layout(**layout)

                else:
                    return (
                        f"Unknown chart '{chart_name}'. "
                        "Valid options: sermons_per_speaker, sermons_per_year, "
                        "verses_per_book, sermons_scatter."
                    )

            file_path = os.path.join("/tmp", f"bbtc_chart_{uuid.uuid4().hex[:8]}.json")
            fig.write_json(file_path)
            return file_path

        except Exception as e:
            return f"Chart generation error: {e}"

    return viz_tool
