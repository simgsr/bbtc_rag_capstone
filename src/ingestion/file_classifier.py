"""Classify BBTC sermon files by role: cell_guide, sermon_slides, or other."""

import re

_CELL_GUIDE_RE = re.compile(
    r'(?:members?(?:27)?|leaders?|cell)[-_]?(?:guide|copy|guide[-_]updated)'
    r'|MembersGuide|MessageSummary.*Members',
    re.IGNORECASE,
)

_OTHER_RE = re.compile(
    r'[-_](handout|visual[-_]?summary)[-_.]|handout\.',
    re.IGNORECASE,
)


def classify_file(filename: str) -> str:
    """
    Returns:
        "cell_guide"    — Members/Leaders/Cell Guide or MessageSummary+Members
        "sermon_slides" — PPT deck, .pptx, or primary sermon PDF
        "other"         — handout, visual summary, or supplementary
    """
    if _CELL_GUIDE_RE.search(filename):
        return "cell_guide"
    if _OTHER_RE.search(filename):
        return "other"
    return "sermon_slides"
