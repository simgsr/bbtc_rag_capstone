# tests/test_vector_tool.py
from unittest.mock import MagicMock
from src.tools.vector_tool import make_vector_tool


def _make_store(results):
    store = MagicMock()
    store.search_sermons.return_value = results
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
    tool.invoke({"query": "grace", "speaker": "Pastor John"})
    _, kwargs = store.search_sermons.call_args
    assert kwargs.get("where") == {"speaker": {"$eq": "Pastor John"}}


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
    tool.invoke({"query": "grace", "year": 2024, "speaker": "Pastor John"})
    _, kwargs = store.search_sermons.call_args
    assert kwargs.get("where") == {
        "$and": [{"year": {"$eq": 2024}}, {"speaker": {"$eq": "Pastor John"}}]
    }
