"""Vision-based extraction for textless / image-only sermon PDFs.

Renders PDF pages to images and asks a multimodal Ollama model (default
``gemma4:e4b``) to read them, recovering metadata and content that text
extraction can't. Used as a fallback in ``ingest.py`` when:

  * a Notes/Guide (NG) PDF yields no text or no topic (image-based PDFs), or
  * a sermon group has no NG file at all (PS-only groups), or
  * a Slides/PPT (PS) PDF is image-only, so no verse refs can be read from text.

The vision model is selected by ``OLLAMA_VISION_MODEL`` in ``.env`` (default
``gemma4:e4b`` — small, fast, and multimodal, so it can read slide images
without the MLX runtime). Set it empty to disable the vision fallback.
"""

import base64
import re

import fitz  # PyMuPDF

from src.ingestion.ps_extractor import parse_verses_from_text

# Fields the metadata prompt asks for, in the order the model is told to reply.
_FIELDS = ("topic", "speaker", "theme", "date", "key_verse", "summary")

_METADATA_PROMPT = (
    "Read these images of sermon slides from a church service and extract the "
    "following fields. Reply with EXACTLY these labels, one per line, using NONE "
    "when a field is not visible:\n"
    "TOPIC: <sermon title>\n"
    "SPEAKER: <speaker name>\n"
    "THEME: <theme>\n"
    "DATE: <service date>\n"
    "KEYVERSE: <main bible verse reference(s), e.g. \"Luke 9:23\" or \"Mark 3:14-15; John 3:3-5\">\n"
    "SUMMARY: <2-3 sentence summary of the sermon's main message>"
)

_VERSES_PROMPT = (
    "Read these images of sermon slides and list every Bible verse reference "
    "shown. Reply with one verse reference per line in \"Book Chapter:Verse\" "
    "format (e.g. \"Luke 9:23\"). If no verses are visible, reply with NONE."
)


def render_pdf_pages(filepath: str, max_pages: int = 4, dpi: int = 120) -> list[str]:
    """Render the first ``max_pages`` pages of a PDF to base64 PNG strings.

    Returns an empty list if the PDF can't be opened. ``dpi=120`` yields
    ~1000px-wide pages — enough for the vision model to read slide text.
    """
    try:
        doc = fitz.open(filepath)
    except Exception:
        return []
    images = []
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(dpi=dpi)
            images.append(base64.b64encode(pix.tobytes("png")).decode())
    finally:
        doc.close()
    return images


def _parse_vision_response(text: str) -> dict:
    """Parse a ``TOPIC: ...`` / ``SPEAKER: ...`` reply into a dict.

    ``NONE`` (case-insensitive) becomes ``None``; the ``summary`` field captures
    everything after its label to the end of the reply (it may span lines).
    Unknown/missing labels are simply absent from the result.
    """
    result: dict[str, str | None] = {}
    for field in _FIELDS:
        # Field labels are written without an underscore in the prompt
        # (`KEYVERSE:`), so the matcher must tolerate both `KEYVERSE:` and
        # `KEY_VERSE:` / `KEY VERSE:` — otherwise key_verse is silently dropped.
        label = re.sub(r"_", r"[_ ]?", field)
        m = re.search(rf'^\s*{label}\s*:\s*(.*)$', text, re.IGNORECASE | re.MULTILINE)
        if not m:
            continue
        value = m.group(1).strip()
        if field == "summary":
            # Everything after the SUMMARY label, joined back into one string.
            rest = text[m.end():].strip()
            value = (value + " " + rest).strip()
        if not value or value.upper() == "NONE":
            result[field] = None
        else:
            result[field] = value
    return result


def extract_from_images(images: list[str], llm) -> dict:
    """One vision call: read rendered slide images and return structured fields.

    Returns a dict with keys ``topic``, ``speaker``, ``theme``, ``date``,
    ``key_verse``, ``summary`` — any of which may be ``None``. Returns ``{}``
    on failure (caller degrades gracefully to the text-only path).
    """
    if not images or not llm:
        return {}
    from langchain_core.messages import HumanMessage
    content: list = [{"type": "text", "text": _METADATA_PROMPT}]
    content += [
        {"type": "image_url", "image_url": f"data:image/png;base64,{b}"} for b in images
    ]
    try:
        resp = llm.invoke([HumanMessage(content=content)])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        return _parse_vision_response(raw)
    except Exception as e:
        print(f"  ⚠️  Vision metadata extraction failed: {e}", flush=True)
        return {}


def extract_verses_from_images(images: list[str], llm) -> list[str]:
    """List Bible verse refs visible in rendered slide images.

    Returns normalized ref strings like ``["Luke 9:23", "Mark 3:14-15"]``, or
    ``[]`` when none are found / the call fails.
    """
    if not images or not llm:
        return []
    from langchain_core.messages import HumanMessage
    content: list = [{"type": "text", "text": _VERSES_PROMPT}]
    content += [
        {"type": "image_url", "image_url": f"data:image/png;base64,{b}"} for b in images
    ]
    try:
        resp = llm.invoke([HumanMessage(content=content)])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        # Parse through the same machinery as PS filenames/text so numbered books
        # ("1 Peter 4:8") and multi-word books ("Song of Songs 2:1") are handled
        # consistently — a naive `^[A-Z][a-z]+ \d+:\d+` filter dropped both.
        # Book-only mentions (no chapter) are dropped, mirroring ingest.py's rule
        # (a bare book name collides with speaker names and common words).
        refs, seen = [], set()
        for line in raw.strip().split("\n"):
            line = line.strip()
            # The prompt's negative token: "If no verses are visible, reply with
            # NONE" — skip exactly that (per-line, so a stray "none" elsewhere in
            # a real answer doesn't discard the verses it does list).
            if not line or re.match(r'^none[,.]?$', line, re.IGNORECASE):
                continue
            for v in parse_verses_from_text(line):
                if not v.get("chapter"):
                    continue
                ref = v["verse_ref"]
                if ref not in seen:
                    seen.add(ref)
                    refs.append(ref)
        return refs
    except Exception as e:
        print(f"  ⚠️  Vision verse extraction failed: {e}", flush=True)
        return []
