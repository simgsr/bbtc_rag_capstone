"""Shared builder for the ``doc_type="metadata"`` sermon title chunk.

The ingest pipeline (``ingest.py``) and the backfill script
(``backfill_title_chunks.py``) both index a compact
``"Topic | Theme: … | Speaker: … | Key verse: … | Date: …"`` chunk per sermon so
topical/title queries retrieve the right sermon without depending on body-text
overlap. The exact format is a single fact that lives here — both callers go
through ``build_sermon_title_text`` so they can't drift (the backfill script
once hardcoded ``language`` and mismatched the ingest path; centralising the
string prevents a repeat).
"""


def build_sermon_title_text(
    topic: str | None,
    theme: str | None,
    speaker: str | None,
    key_verse: str | None,
    date: str | None,
) -> str:
    """Return the canonical ``doc_type="metadata"`` title text for a sermon.

    The topic leads bare (so a topic-only query matches its literal text); every
    other field is labelled. Falsy fields are dropped. Returns "" when no field
    is present (the caller should skip emitting an empty metadata chunk).
    """
    parts = [
        topic,
        f"Theme: {theme}" if theme else None,
        f"Speaker: {speaker}" if speaker else None,
        f"Key verse: {key_verse}" if key_verse else None,
        f"Date: {date}" if date else None,
    ]
    return " | ".join(str(x) for x in parts if x)