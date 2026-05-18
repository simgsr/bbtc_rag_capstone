"""Group BBTC sermon files into (ng, ps) sermon groups."""

import glob
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from src.ingestion.file_classifier import classify_file
from src.ingestion.filename_parser import extract_any_date, extract_topic_words


@dataclass
class SermonGroup:
    ng: str | None = None
    ps: list[str] = field(default_factory=list)
    page_date: str | None = None  # ISO date from the website page (most authoritative)


def _date_proximity(d1: str | None, d2: str | None, tolerance: int = 3) -> bool:
    if not d1 or not d2:
        return False
    fmt = "%Y-%m-%d"
    try:
        return abs((datetime.strptime(d1, fmt) - datetime.strptime(d2, fmt)).days) <= tolerance
    except ValueError:
        return False


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def group_sermon_files(filenames: list[str], staging_dir: str | None = None) -> list[SermonGroup]:
    """
    Group filenames into SermonGroups.

    If staging_dir is given, manifest files (_manifest_*.json) written by the
    scraper are read first — files listed in a manifest are paired exactly as
    the website had them, with no filename heuristics needed.

    Remaining files (no manifest, or legacy staging without manifests) are
    paired by date proximity (≤ 3 days) or topic-word Jaccard (≥ 0.5).
    Unpaired PS files become standalone groups (ng=None). Handouts are ignored.
    """
    groups: list[SermonGroup] = []
    manifested: set[str] = set()

    # --- Phase 1: manifest-based pairing (exact, from scraper) ---
    if staging_dir:
        for path in sorted(glob.glob(os.path.join(staging_dir, "_manifest_*.json"))):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue

            group = SermonGroup(page_date=data.get("date"))
            for fname in data.get("files", []):
                kind = classify_file(fname)
                if kind == "ng":
                    group.ng = fname
                elif kind == "ps":
                    group.ps.append(fname)
                manifested.add(fname)

            if group.ng or group.ps:
                groups.append(group)

    # --- Phase 2: fuzzy pairing for files not covered by any manifest ---
    remaining = [f for f in filenames if f not in manifested]

    ngs, pss = [], []
    for f in remaining:
        kind = classify_file(f)
        if kind == "ng":
            ngs.append(f)
        elif kind == "ps":
            pss.append(f)

    used_ps: set[str] = set()
    for ng in ngs:
        group = SermonGroup(ng=ng)
        ng_date = extract_any_date(ng)
        ng_words = extract_topic_words(ng)

        for ps in pss:
            if ps in used_ps:
                continue
            ps_date = extract_any_date(ps)
            ps_words = extract_topic_words(ps)
            if _date_proximity(ng_date, ps_date) or _jaccard(ng_words, ps_words) >= 0.5:
                group.ps.append(ps)
                used_ps.add(ps)

        groups.append(group)

    for ps in pss:
        if ps not in used_ps:
            groups.append(SermonGroup(ps=[ps]))

    return groups
