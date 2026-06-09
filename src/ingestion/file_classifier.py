"""Classify BBTC sermon files as ng, ps, or handout."""

import re

_NG_RE = re.compile(
    r'(?:members?(?:27s?)?|leaders?|cell)[-_]?(?:guide|copy|guide[-_]updated)'
    r'|MembersGuide|MessageSummary.*Members'
    r'|[-_]notes?[-_.]|[-_]notes?\.'
    r'|[-_]members?(?:27s?)?\.(?:pdf|docx?|pptx?)$',
    re.IGNORECASE,
)

_HANDOUT_RE = re.compile(
    r'[-_](handout|visual[-_]?summary)[-_.]|handout\.',
    re.IGNORECASE,
)


_SERMON_EXTENSIONS = ('.pdf', '.pptx', '.ppt', '.docx', '.doc')


def classify_file(filename: str) -> str:
    """
    Returns:
        "ng"      — Notes / Cell Guide / Members Guide / Members Copy
        "ps"      — PPT deck, slides PDF, or primary sermon PDF
        "handout" — Handout, visual summary, manifest JSON, or non-sermon file (skip)
    """
    if filename.startswith("_manifest_") and filename.endswith(".json"):
        return "handout"
    if not any(filename.lower().endswith(ext) for ext in _SERMON_EXTENSIONS):
        return "handout"
    if _NG_RE.search(filename):
        return "ng"
    if _HANDOUT_RE.search(filename):
        return "handout"
    return "ps"
