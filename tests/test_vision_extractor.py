"""Tests for the vision fallback (textless / image-only sermon PDFs).

Regression coverage for two bugs that silently lost data:

  * ``_parse_vision_response`` regexed the literal ``key_verse`` (underscore)
    while the metadata prompt tells the model to reply ``KEYVERSE:`` — the
    field never parsed, so ``ingest.py``'s key-verse feed was dead code.
  * ``extract_verses_from_images`` filtered with ``^[A-Z][a-z]+ \\d+:\\d+``,
    dropping numbered books ("1 Peter 4:8") and multi-word books ("Song of
    Songs 2:1"), and returned ``[]`` if the word "none" appeared anywhere in
    a reply that otherwise listed verses.

Verse extraction now routes through ``parse_verses_from_text`` (the same
parser used for PS filenames/text), and the "none" negative token is matched
per-line.
"""
from src.ingestion.vision_extractor import _parse_vision_response, extract_verses_from_images


class _StubContent:
    def __init__(self, text: str):
        self.content = text


class _StubVisionLLM:
    """Minimal llm stand-in: invoke() returns a fixed reply."""

    def __init__(self, reply: str):
        self._reply = reply

    def invoke(self, messages):
        return _StubContent(self._reply)


class _BoomLLM:
    def invoke(self, messages):
        raise RuntimeError("ollama down")


_METADATA_SAMPLE = (
    "TOPIC: A Call to Discipleship\n"
    "SPEAKER: SP Daniel Foo\n"
    "THEME: Cost of Following Christ\n"
    "DATE: 2017-02-26\n"
    "KEYVERSE: Mark 3:14-15\n"
    "SUMMARY: Jesus chose the twelve to be with him and send him out.\n"
)


# ── _parse_vision_response ───────────────────────────────────────────────────

def test_parse_vision_response_keyverse_without_underscore():
    # Regression: the prompt says `KEYVERSE:` but the parser used to search
    # `key_verse:` (with underscore) — the key_verse field was silently dropped.
    parsed = _parse_vision_response(_METADATA_SAMPLE)
    assert parsed["key_verse"] == "Mark 3:14-15"


def test_parse_vision_response_underscore_variants():
    # The tolerant matcher should accept every plausible spelling of the label.
    for label in ("KEY_VERSE:", "KEY VERSE:", "keyverse:"):
        reply = _METADATA_SAMPLE.replace("KEYVERSE:", label)
        assert _parse_vision_response(reply)["key_verse"] == "Mark 3:14-15", label


def test_parse_vision_response_all_fields():
    parsed = _parse_vision_response(_METADATA_SAMPLE)
    assert parsed["topic"] == "A Call to Discipleship"
    assert parsed["speaker"] == "SP Daniel Foo"
    assert parsed["theme"] == "Cost of Following Christ"
    assert parsed["date"] == "2017-02-26"


def test_parse_vision_response_none_becomes_none():
    parsed = _parse_vision_response(
        "TOPIC: Quiet Confidence\nSPEAKER: NONE\nTHEME: NONE\nDATE: NONE\n"
    )
    assert parsed["topic"] == "Quiet Confidence"
    assert parsed["speaker"] is None
    assert parsed["theme"] is None
    assert parsed["date"] is None


def test_parse_vision_response_summary_spans_lines():
    reply = "TOPIC: T\nSUMMARY: First line of the summary.\nStill the summary.\n"
    parsed = _parse_vision_response(reply)
    assert "First line" in parsed["summary"]
    assert "Still the summary" in parsed["summary"]


def test_parse_vision_response_unknown_labels_absent():
    parsed = _parse_vision_response("FOO: bar\nTOPIC: Known\n")
    assert parsed.get("topic") == "Known"
    assert "foo" not in parsed


def test_vision_llm_failure_degrades_to_empty():
    assert extract_verses_from_images(["png"], _BoomLLM()) == []


# ── extract_verses_from_images ───────────────────────────────────────────────

def test_extract_verses_single_word_book():
    assert extract_verses_from_images(["png"], _StubVisionLLM("Luke 9:23\n")) == ["Luke 9:23"]


def test_extract_verses_numbered_book():
    # Regression: the old `^[A-Z][a-z]+ \d+:\d+` filter dropped numbered books.
    out = extract_verses_from_images(
        ["png"], _StubVisionLLM("1 Peter 4:8\n2 Corinthians 5:17\n")
    )
    assert out == ["1 Peter 4:8", "2 Corinthians 5:17"]


def test_extract_verses_multi_word_book():
    # Regression: "Song of Songs" was dropped by the single-word filter too.
    assert extract_verses_from_images(["png"], _StubVisionLLM("Song of Songs 2:1\n")) == ["Song of Songs 2:1"]


def test_extract_verses_range():
    assert extract_verses_from_images(["png"], _StubVisionLLM("Mark 3:14-15\n")) == ["Mark 3:14-15"]


def test_extract_verses_dedupes():
    out = extract_verses_from_images(["png"], _StubVisionLLM("Luke 9:23\nLuke 9:23\n"))
    assert out == ["Luke 9:23"]


def test_extract_verses_book_only_mention_dropped():
    # A bare book name is too unreliable to store as a preached verse (it
    # collides with speaker names / common words) — no chapter, so skip.
    out = extract_verses_from_images(["png"], _StubVisionLLM("Hebrews\nLuke 9:23\n"))
    assert out == ["Luke 9:23"]


def test_extract_verses_none_reply_returns_empty():
    assert extract_verses_from_images(["png"], _StubVisionLLM("NONE\n")) == []
    assert extract_verses_from_images(["png"], _StubVisionLLM("none, nothing visible\n")) == []


def test_extract_verses_stray_none_keeps_verses():
    # Regression: the old `"NONE" in raw.upper()` check discarded any reply that
    # contained "none" anywhere, even one that also listed verses.
    reply = "No extractable slide text, but the verse is on screen.\nLuke 9:23\n"
    assert extract_verses_from_images(["png"], _StubVisionLLM(reply)) == ["Luke 9:23"]


def test_extract_verses_garbage_lines_ignored():
    out = extract_verses_from_images(
        ["png"], _StubVisionLLM("Slide title\n(no ref)\nRomans 8:28\n")
    )
    assert out == ["Romans 8:28"]
