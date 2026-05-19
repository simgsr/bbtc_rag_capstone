"""Classify BBTC sermon files as ng, ps, or handout."""

import re

_NG_RE = re.compile(
    r'(?:members?(?:27)?|leaders?|cell)[-_]?(?:guide|copy|guide[-_]updated)'
    r'|MembersGuide|MessageSummary.*Members'
    r'|[-_]notes?[-_.]|[-_]notes?\.',
    re.IGNORECASE,
)

_HANDOUT_RE = re.compile(
    r'[-_](handout|visual[-_]?summary)[-_.]|handout\.',
    re.IGNORECASE,
)


def classify_file(filename: str) -> str:
    """
    Returns:
        "ng"      — Notes / Cell Guide / Members Guide / Members Copy
        "ps"      — PPT deck, slides PDF, or primary sermon PDF
        "handout" — Handout, visual summary, or scraper manifest (skip)
    """
    if filename.startswith("_manifest_") and filename.endswith(".json"):
        return "handout"
    if _NG_RE.search(filename):
        return "ng"
    if _HANDOUT_RE.search(filename):
        return "handout"
    return "ps"
