# deploy/backend/api/charts.py
import sqlite3
import os
from fastapi import APIRouter

router = APIRouter()


def _db_path() -> str:
    return os.path.join(os.getenv("DATA_DIR", "data"), "sermons.db")


def _query(sql: str, params: tuple = ()) -> list[tuple]:
    with sqlite3.connect(_db_path()) as conn:
        return conn.execute(sql, params).fetchall()


@router.get("/api/stats")
def get_stats():
    total = _query("SELECT COUNT(*) FROM sermons")[0][0]
    speakers = _query(
        "SELECT COUNT(DISTINCT speaker) FROM sermons WHERE speaker IS NOT NULL AND speaker != ''"
    )[0][0]
    year_row = _query("SELECT MIN(year), MAX(year) FROM sermons WHERE year IS NOT NULL")[0]
    return {
        "total_sermons": total,
        "total_speakers": speakers,
        "year_min": year_row[0],
        "year_max": year_row[1],
    }


@router.get("/api/charts/by-year")
def by_year():
    rows = _query(
        "SELECT year, COUNT(*) FROM sermons WHERE year IS NOT NULL GROUP BY year ORDER BY year"
    )
    return [{"year": r[0], "count": r[1]} for r in rows]


@router.get("/api/charts/by-speaker")
def by_speaker():
    rows = _query(
        "SELECT speaker, COUNT(*) FROM sermons "
        "WHERE speaker IS NOT NULL AND speaker != '' "
        "GROUP BY speaker ORDER BY COUNT(*) DESC LIMIT 20"
    )
    return [{"speaker": r[0], "count": r[1]} for r in rows]


@router.get("/api/charts/by-verse")
def by_verse():
    rows = _query(
        "SELECT bible_book, COUNT(*) FROM sermons "
        "WHERE bible_book IS NOT NULL AND bible_book != '' "
        "GROUP BY bible_book ORDER BY COUNT(*) DESC LIMIT 20"
    )
    return [{"bible_book": r[0], "count": r[1]} for r in rows]


@router.get("/api/charts/scatter")
def scatter():
    rows = _query(
        "SELECT year, speaker, COUNT(*) FROM sermons "
        "WHERE year IS NOT NULL AND speaker IS NOT NULL AND speaker != '' "
        "GROUP BY year, speaker ORDER BY year"
    )
    return [{"year": r[0], "speaker": r[1], "count": r[2]} for r in rows]
