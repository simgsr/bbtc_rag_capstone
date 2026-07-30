"""Regression tests for scraper download-path hardening.

Guards the path-traversal fix in ``BBTCScraper._staging_filename``: a remote,
attacker-influenced URL must never resolve to a write outside ``data/staging``.
See the security audit — the original code took ``os.path.basename`` *before*
URL-decoding, so ``%2f``/``%2e%2e`` survived and let ``..`` escape staging.
"""
import os

from src.scraper.bbtc_scraper import BBTCScraper


def _staging_path(name: str, staging_dir: str = "data/staging") -> str:
    return os.path.normpath(os.path.join(staging_dir, name))


def test_encoded_traversal_is_reduced_to_basename():
    # The classic exploit: URL-encoded ../ climbing out of staging.
    url = "https://www.bbtc.com.sg/audio-sermons/..%2f..%2f..%2f..%2fingest.py"
    name = BBTCScraper._staging_filename(url, lang="English", year=2026)
    assert name == "English_2026_ingest.py"
    # And the resulting path stays inside staging.
    resolved = _staging_path(name)
    assert resolved.startswith(os.path.normpath("data/staging") + os.sep)
    assert ".." not in resolved.split(os.sep)


def test_literal_traversal_is_reduced_to_basename():
    url = "https://evil.example.com/../../../../etc/passwd.pdf"
    name = BBTCScraper._staging_filename(url, lang="Mandarin", year=2024)
    assert name == "Mandarin_2024_passwd.pdf"
    assert _staging_path(name).startswith(os.path.normpath("data/staging") + os.sep)


def test_backslash_separators_are_stripped():
    url = "https://www.bbtc.com.sg/x/..%5c..%5cwin.pdf"
    name = BBTCScraper._staging_filename(url, lang="English", year=2025)
    assert name == "English_2025_win.pdf"


def test_query_string_is_ignored():
    url = "https://www.bbtc.com.sg/sermons/notes.pdf?download=1&t=2"
    name = BBTCScraper._staging_filename(url, lang="English", year=2025)
    assert name == "English_2025_notes.pdf"


def test_dot_and_empty_names_are_rejected():
    for tail in ("..", ".", "", "..%2f..%2f"):
        url = f"https://www.bbtc.com.sg/{tail}"
        assert BBTCScraper._staging_filename(url, lang="English", year=2025) is None


def test_normal_filename_is_preserved():
    url = "https://www.bbtc.com.sg/audio-sermons/Members%27%20Guide%202024.pdf"
    name = BBTCScraper._staging_filename(url, lang="English", year=2024)
    assert name == "English_2024_Members' Guide 2024.pdf"
