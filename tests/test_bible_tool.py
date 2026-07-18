# tests/test_bible_tool.py
from unittest.mock import MagicMock
from src.tools.bible_tool import make_bible_tool, _normalize_ref


def _make_store(results):
    store = MagicMock()
    store.search_bible.return_value = results
    store.get_bible_versions.return_value = results
    return store


def _sample_results():
    return [
        {"content": "For God so loved the world.",
         "metadata": {"reference": "John 3:16", "version": "NIV"}},
        {"content": "「 神愛世人，甚至將他的獨生子賜給他們",
         "metadata": {"reference": "John 3:16", "version": "ChiUn"}},
    ]


def test_normalize_ref_canonicalises():
    assert _normalize_ref("1 john 1:9") == "1 John 1:9"
    assert _normalize_ref("John 3:16") == "John 3:16"
    assert _normalize_ref("john 3") == "John 3"
    assert _normalize_ref("not a ref") is None


def test_search_bible_no_version_filter_searches_all():
    store = _make_store(_sample_results())
    _, search = make_bible_tool(store)
    search.invoke({"query": "forgiveness", "k": 5})
    _, kwargs = store.search_bible.call_args
    assert kwargs.get("where") is None


def test_search_bible_version_filter_case_insensitive():
    store = _make_store(_sample_results())
    _, search = make_bible_tool(store)
    out = search.invoke({"query": "forgiveness", "k": 5, "version": "niv"})
    # Version is NOT pushed into a case-sensitive `where`; it's a case-insensitive
    # post-filter, so a lowercase 'niv' still matches stored 'NIV' and excludes ChiUn.
    _, kwargs = store.search_bible.call_args
    assert kwargs.get("where") is None
    assert kwargs.get("k") >= 40  # oversampled for the post-filter
    assert "NIV" in out
    assert "John 3:16" in out
    assert "ChiUn" not in out


def test_search_bible_result_includes_version_label():
    store = _make_store(_sample_results())
    _, search = make_bible_tool(store)
    out = search.invoke({"query": "forgiveness", "k": 5})
    assert "NIV" in out
    assert "John 3:16" in out


def test_get_bible_versions_returns_all_translations():
    store = _make_store(_sample_results())
    versions, _ = make_bible_tool(store)
    out = versions.invoke({"reference": "John 3:16"})
    assert "NIV" in out
    assert "ChiUn" in out


def test_get_bible_versions_unparseable_reference():
    store = _make_store([])
    versions, _ = make_bible_tool(store)
    out = versions.invoke({"reference": "nonsense"})
    assert "Could not parse" in out