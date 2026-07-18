# tests/test_title_chunk.py
from src.ingestion.title_chunk import build_sermon_title_text


def test_topic_leads_bare_others_labelled():
    out = build_sermon_title_text("Grace and Salvation", "Mercy", "Pastor John", "John 3:16", "2024-03-10")
    assert out == "Grace and Salvation | Theme: Mercy | Speaker: Pastor John | Key verse: John 3:16 | Date: 2024-03-10"


def test_falsy_fields_dropped():
    out = build_sermon_title_text("Discipleship", None, None, "Luke 9:23", None)
    assert out == "Discipleship | Key verse: Luke 9:23"


def test_empty_when_no_fields():
    assert build_sermon_title_text(None, None, None, None, None) == ""


def test_topic_only():
    assert build_sermon_title_text("Faith", None, None, None, None) == "Faith"