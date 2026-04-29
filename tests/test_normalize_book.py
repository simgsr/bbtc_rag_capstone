import pytest
from src.storage.normalize_book import normalize_book


def test_canonical_name_passthrough():
    assert normalize_book("Matthew") == "Matthew"
    assert normalize_book("Revelation") == "Revelation"
    assert normalize_book("Psalms") == "Psalms"


def test_allcaps_variants():
    assert normalize_book("HEBREWS") == "Hebrews"
    assert normalize_book("ACTS") == "Acts"
    assert normalize_book("JOHN") == "John"
    assert normalize_book("MATTHEW") == "Matthew"
    assert normalize_book("ROMANS") == "Romans"
    assert normalize_book("ISAIAH") == "Isaiah"
    assert normalize_book("EXODUS") == "Exodus"
    assert normalize_book("DANIEL") == "Daniel"
    assert normalize_book("LUKE") == "Luke"
    assert normalize_book("MARK") == "Mark"
    assert normalize_book("PSALM") == "Psalms"
    assert normalize_book("PSALMS") == "Psalms"
    assert normalize_book("COLOSSIANS") == "Colossians"
    assert normalize_book("EPHESIANS") == "Ephesians"
    assert normalize_book("PHILIPPIANS") == "Philippians"
    assert normalize_book("JEREMIAH") == "Jeremiah"
    assert normalize_book("PROVERBS") == "Proverbs"
    assert normalize_book("GENESIS") == "Genesis"
    assert normalize_book("DEUTERONOMY") == "Deuteronomy"
    assert normalize_book("JOSHUA") == "Joshua"
    assert normalize_book("JUDGES") == "Judges"
    assert normalize_book("TITUS") == "Titus"
    assert normalize_book("REVELATION") == "Revelation"
    assert normalize_book("ECCLESIASTES") == "Ecclesiastes"
    assert normalize_book("HOSEA") == "Hosea"


def test_abbreviations():
    assert normalize_book("Lk") == "Luke"
    assert normalize_book("Heb") == "Hebrews"
    assert normalize_book("Rom") == "Romans"
    assert normalize_book("Rev") == "Revelation"
    assert normalize_book("Eph") == "Ephesians"
    assert normalize_book("Col") == "Colossians"
    assert normalize_book("Ps") == "Psalms"
    assert normalize_book("Psa") == "Psalms"
    assert normalize_book("Gen") == "Genesis"
    assert normalize_book("Isa") == "Isaiah"
    assert normalize_book("Jer") == "Jeremiah"
    assert normalize_book("Prov") == "Proverbs"
    assert normalize_book("Matt") == "Matthew"
    assert normalize_book("Jn") == "John"
    assert normalize_book("Act") == "Acts"
    assert normalize_book("Exo") == "Exodus"
    assert normalize_book("Ex") == "Exodus"
    assert normalize_book("Deu") == "Deuteronomy"
    assert normalize_book("Deut") == "Deuteronomy"
    assert normalize_book("Jos") == "Joshua"
    assert normalize_book("EPH") == "Ephesians"
    assert normalize_book("COL") == "Colossians"
    assert normalize_book("JER") == "Jeremiah"


def test_numbered_books():
    assert normalize_book("1 Samuel") == "1 Samuel"
    assert normalize_book("2 Samuel") == "2 Samuel"
    assert normalize_book("1Samuel") == "1 Samuel"
    assert normalize_book("1 Kings") == "1 Kings"
    assert normalize_book("2 Kings") == "2 Kings"
    assert normalize_book("1 Chronicles") == "1 Chronicles"
    assert normalize_book("2 Chronicles") == "2 Chronicles"
    assert normalize_book("1 Corinthians") == "1 Corinthians"
    assert normalize_book("2 Corinthians") == "2 Corinthians"
    assert normalize_book("1 Thessalonians") == "1 Thessalonians"
    assert normalize_book("2 Thessalonians") == "2 Thessalonians"
    assert normalize_book("1 Timothy") == "1 Timothy"
    assert normalize_book("2 Timothy") == "2 Timothy"
    assert normalize_book("1 Peter") == "1 Peter"
    assert normalize_book("2 Peter") == "2 Peter"
    assert normalize_book("1 John") == "1 John"
    assert normalize_book("2 John") == "2 John"
    assert normalize_book("3 John") == "3 John"


def test_revelations_variant():
    assert normalize_book("REVELATIONS") == "Revelation"
    assert normalize_book("revelations") == "Revelation"


def test_garbage_returns_none():
    assert normalize_book("Jericho") is None
    assert normalize_book("jericho") is None


def test_ambiguous_unnumbered_returns_none():
    assert normalize_book("Samuel") is None
    assert normalize_book("Kings") is None
    assert normalize_book("Chronicles") is None
    assert normalize_book("Corinthians") is None
    assert normalize_book("Timothy") is None
    assert normalize_book("Peter") is None


def test_empty_and_none_inputs():
    assert normalize_book("") is None
    assert normalize_book(None) is None
    assert normalize_book("   ") is None
