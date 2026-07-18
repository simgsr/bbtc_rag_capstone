# tests/test_vector_tool.py
from unittest.mock import MagicMock
from src.tools.vector_tool import make_vector_tool


def _make_store(results):
    store = MagicMock()
    store.search_sermons.return_value = results
    # Speaker-filter path now fetches the whole collection (so a prolific speaker
    # on a rare topic isn't under-filled); it reads the count via counts().
    store.counts.return_value = {"sermon_collection": 589}
    return store


def _sample_results():
    return [
        {
            "content": "God so loved the world.",
            "metadata": {
                "sermon_id": "2024-03-10-grace",
                "topic": "Grace and Salvation",
                "speaker": "Pastor John",
                "date": "2024-03-10",
                "key_verse": "John 3:16",
            },
            "distance": 0.1,
        },
        {
            "content": "Faith is the substance of things hoped for.",
            "metadata": {
                "sermon_id": "2023-06-01-faith",
                "topic": "The Nature of Faith",
                "speaker": "Pastor Mary",
                "date": "2023-06-01",
                "key_verse": "Hebrews 11:1",
            },
            "distance": 0.2,
        },
        {
            # Same year (2024) as Pastor John but a different speaker — survives a
            # year-only `where` but MUST be removed by the speaker post-filter.
            "content": "Hope does not put us to shame.",
            "metadata": {
                "sermon_id": "2024-08-18-hope",
                "topic": "Living in Hope",
                "speaker": "Pastor Mary",
                "date": "2024-08-18",
                "key_verse": "Romans 5:5",
            },
            "distance": 0.15,
        },
    ]


def test_tool_returns_string():
    tool = make_vector_tool(_make_store(_sample_results()))
    result = tool.invoke({"query": "grace"})
    assert isinstance(result, str)


def test_tool_includes_speaker_in_output():
    tool = make_vector_tool(_make_store(_sample_results()))
    result = tool.invoke({"query": "grace"})
    assert "Pastor John" in result


def test_tool_includes_topic_in_output():
    tool = make_vector_tool(_make_store(_sample_results()))
    result = tool.invoke({"query": "grace"})
    assert "Grace and Salvation" in result


def test_tool_passes_year_filter():
    store = _make_store(_sample_results())
    tool = make_vector_tool(store)
    tool.invoke({"query": "grace", "year": 2024})
    store.search_sermons.assert_called_once()
    _, kwargs = store.search_sermons.call_args
    assert kwargs.get("where") == {"year": {"$eq": 2024}}


def test_tool_passes_speaker_filter():
    store = _make_store(_sample_results())
    tool = make_vector_tool(store)
    result = tool.invoke({"query": "grace", "speaker": "John"})
    _, kwargs = store.search_sermons.call_args
    # Speaker is NOT pushed into the Chroma `where` (exact-match only would miss
    # titled speakers); it's applied as a substring post-filter instead.
    assert kwargs.get("where") is None
    # The whole collection is fetched so a prolific speaker on a rare topic isn't
    # under-filled (fetch_k = collection count, not a fixed 4× oversample).
    assert kwargs.get("k") == 589
    # Only sermons whose speaker contains the needle survive the post-filter.
    assert "Pastor John" in result
    assert "Pastor Mary" not in result


def test_tool_no_results():
    tool = make_vector_tool(_make_store([]))
    result = tool.invoke({"query": "something obscure"})
    assert "No relevant" in result


def test_tool_default_no_filters():
    store = _make_store(_sample_results())
    tool = make_vector_tool(store)
    tool.invoke({"query": "grace"})
    _, kwargs = store.search_sermons.call_args
    assert kwargs.get("where") is None


def test_tool_passes_combined_year_and_speaker_filter():
    store = _make_store(_sample_results())
    tool = make_vector_tool(store)
    result = tool.invoke({"query": "grace", "year": 2024, "speaker": "Pastor John"})
    _, kwargs = store.search_sermons.call_args
    # Year goes into `where`; speaker is post-filtered, so it must NOT appear in `where`.
    assert kwargs.get("where") == {"year": {"$eq": 2024}}
    # The 2024 Pastor Mary row survives the year filter (the mock store ignores
    # `where` and returns all rows), so only the speaker post-filter can remove it.
    # If this assertion fails, the speaker post-filter is broken.
    assert "Pastor John" in result
    assert "Pastor Mary" not in result
    assert "Living in Hope" not in result
