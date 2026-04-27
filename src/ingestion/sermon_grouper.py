"""Group BBTC sermon files into (cell_guide, slides, other) sermon groups."""

from dataclasses import dataclass, field
from datetime import datetime
from src.ingestion.file_classifier import classify_file
from src.ingestion.filename_parser import extract_any_date, extract_topic_words


@dataclass
class SermonGroup:
    cell_guide: str | None = None
    slides: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)


def _date_proximity(d1: str | None, d2: str | None, tolerance: int = 3) -> bool:
    if not d1 or not d2:
        return False
    fmt = "%Y-%m-%d"
    return abs((datetime.strptime(d1, fmt) - datetime.strptime(d2, fmt)).days) <= tolerance


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def group_sermon_files(filenames: list[str]) -> list[SermonGroup]:
    """
    Group filenames into SermonGroups.
    Each cell guide becomes one group. Slides are paired to their cell guide
    by date proximity (≤ 3 days) or high topic-word Jaccard (≥ 0.5).
    Unpaired slides each become a standalone group.
    """
    cell_guides, slides, others = [], [], []

    for f in filenames:
        kind = classify_file(f)
        if kind == "cell_guide":
            cell_guides.append(f)
        elif kind == "sermon_slides":
            slides.append(f)
        else:
            others.append(f)

    groups: list[SermonGroup] = []
    used_slides: set[str] = set()

    for cg in cell_guides:
        group = SermonGroup(cell_guide=cg)
        cg_date = extract_any_date(cg)
        cg_words = extract_topic_words(cg)

        for slide in slides:
            if slide in used_slides:
                continue
            slide_date = extract_any_date(slide)
            slide_words = extract_topic_words(slide)

            near = _date_proximity(cg_date, slide_date)
            similar = _jaccard(cg_words, slide_words) >= 0.5

            if near or similar:
                group.slides.append(slide)
                used_slides.add(slide)

        # Attach other files (handouts etc.) that topic-match this cell guide
        for o in others:
            o_words = extract_topic_words(o)
            if _jaccard(cg_words, o_words) >= 0.4:
                group.other.append(o)

        groups.append(group)

    # Standalone slides (no matching cell guide)
    for slide in slides:
        if slide not in used_slides:
            groups.append(SermonGroup(slides=[slide]))

    return groups
