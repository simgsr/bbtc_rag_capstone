from src.ingestion.ps_extractor import parse_verses_from_filename, normalize_verse_ref


def test_luke_chapter_verse():
    verses = parse_verses_from_filename("English_2024_03-COST-TO-MENTOR-LUKE-9V23-20240615-compressed.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "Luke"
    assert verses[0]["chapter"] == 9
    assert verses[0]["verse_start"] == 23


def test_john_chapter_only():
    verses = parse_verses_from_filename("English_2024_04-HE-IS-OUR-HOPE-JOHN-11-20240330-PPT-2-compressed.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "John"
    assert verses[0]["chapter"] == 11


def test_hebrews_book_only():
    verses = parse_verses_from_filename("English_2024_Walking-in-Submission-HEBREWS-REVISITED-compressed.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "Hebrews"


def test_verse_range():
    verses = parse_verses_from_filename("English_2024_03-TOGETHER-AS-ONE-LUKE-10V1-3-20240127-PPT.pdf")
    assert len(verses) >= 1
    v = verses[0]
    assert v["book"] == "Luke"
    assert v["chapter"] == 10
    assert v["verse_start"] == 1
    assert v["verse_end"] == 3


def test_no_verse_returns_empty():
    verses = parse_verses_from_filename("English_2024_Some-Sermon-Without-Verse-compressed.pdf")
    assert verses == []


def test_normalize_verse_ref_basic():
    ref = normalize_verse_ref("Luke", 9, 23, None)
    assert ref == "Luke 9:23"


def test_normalize_verse_ref_range():
    ref = normalize_verse_ref("Luke", 10, 1, 3)
    assert ref == "Luke 10:1-3"


def test_normalize_verse_ref_chapter_only():
    ref = normalize_verse_ref("John", 11, None, None)
    assert ref == "John 11"


def test_numbered_prefix_1_samuel():
    verses = parse_verses_from_filename("English_2019_1-SAMUEL-9V1-10.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "1 Samuel"
    assert verses[0]["chapter"] == 9
    assert verses[0]["verse_start"] == 1
    assert verses[0]["verse_end"] == 10


def test_numbered_prefix_2_kings():
    verses = parse_verses_from_filename("English_2022_2-KINGS-4V1-7.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "2 Kings"
    assert verses[0]["chapter"] == 4


def test_numbered_prefix_1_corinthians():
    verses = parse_verses_from_filename("English_2020_LOVE-CHAPTER-1-CORINTHIANS-13V4-7.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "1 Corinthians"
    assert verses[0]["chapter"] == 13


def test_numbered_prefix_2_timothy():
    verses = parse_verses_from_filename("English_2021_EQUIP-2-TIMOTHY-3V16.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "2 Timothy"
    assert verses[0]["chapter"] == 3
    assert verses[0]["verse_start"] == 16


def test_unnumbered_book_still_works():
    verses = parse_verses_from_filename("English_2024_FAITH-HEBREWS-11V1.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "Hebrews"
    assert verses[0]["chapter"] == 11
    assert verses[0]["verse_start"] == 1


def test_colon_separator():
    verses = parse_verses_from_filename("English_2024_JOHN-3:16-compressed.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "John"
    assert verses[0]["chapter"] == 3
    assert verses[0]["verse_start"] == 16


def test_dash_separator_with_verse():
    verses = parse_verses_from_filename("English_2024_EXODUS-12-1-10-compressed.pdf")
    assert len(verses) >= 1
    assert verses[0]["book"] == "Exodus"
    assert verses[0]["chapter"] == 12
    assert verses[0]["verse_start"] == 1
    assert verses[0]["verse_end"] == 10


def test_multiple_verses_in_filename():
    verses = parse_verses_from_filename("English_2024_JOHN-3:16-ROMANS-8:28-compressed.pdf")
    assert len(verses) == 2
    assert verses[0]["verse_ref"] == "John 3:16"
    assert verses[1]["verse_ref"] == "Romans 8:28"
