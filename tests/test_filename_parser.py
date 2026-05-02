import pytest
from src.ingestion.filename_parser import parse_cell_guide_filename, extract_any_date, extract_topic_words


class TestParseCellGuideFilename:
    def test_long_hyphenated_with_by_elder(self):
        r = parse_cell_guide_filename(
            "English_2018_28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide-updated.pdf"
        )
        assert r["date"] == "2018-07-28"
        assert r["topic"] == "Know Your Enemy"
        assert r["speaker"] == "Ps Edric Sng"

    def test_long_hyphenated_without_by(self):
        r = parse_cell_guide_filename(
            "English_2018_06-07-July-2018-Effective-Prayer-Part-5-SP-Daniel-Foo-Members-Guide.pdf"
        )
        assert r["date"] == "2018-07-06"
        assert r["topic"] == "Effective Prayer Part 5"
        assert r["speaker"] == "SP Daniel Foo"

    def test_long_hyphenated_ps_speaker(self):
        r = parse_cell_guide_filename(
            "English_2018_01-02-Dec-2018-The-WOW-Factor-by-Ps-Andrew-Tan-Members-guide.pdf"
        )
        assert r["date"] == "2018-12-01"
        assert r["topic"] == "The WOW Factor"
        assert r["speaker"] == "Ps Andrew Tan"

    def test_camelcase_abbreviation_elvm(self):
        r = parse_cell_guide_filename(
            "English_2015_FearOrFaith_eLVM_2015-12-19_20_MessageSummary_MembersGuide.pdf"
        )
        assert r["date"] == "2015-12-19"
        assert r["topic"] == "Fear or Faith"
        assert r["speaker"] == "Elder Lok Vi Ming"

    def test_camelcase_full_name(self):
        r = parse_cell_guide_filename(
            "English_2015_ChooseWisely_PsAndrewTan_2015-12-05_06_MessageSummary_MembersGuide.pdf"
        )
        assert r["date"] == "2015-12-05"
        assert r["topic"] == "Choose Wisely"
        assert r["speaker"] == "Ps Andrew Tan"

    def test_guest_speaker_normalized(self):
        r = parse_cell_guide_filename(
            "English_2015_Pursuit-of-Gods-Presence-by-Rev-David-Ravenhill-members_guide.pdf"
        )
        assert r["speaker"] == "Guest Speaker"

    def test_two_digit_year(self):
        r = parse_cell_guide_filename(
            "English_2015_25-26-July-15-This-Life-the-Next-by-Ps-Chew-Weng-Chee_Members-Guide.pdf"
        )
        assert r["date"] == "2015-07-25"

    def test_date_range_with_new_month_year(self):
        # This filename caused the 2001 bug because '01' in '01-Dec' was taken as year
        r = parse_cell_guide_filename(
            "English_2019_30-Nov-01-Dec-2019-T3-to-Maturity-by-E-Goh-Hock-Chye-Members-Guide.pdf"
        )
        # It now correctly skips '01' as a year and finds the full date later
        assert r["date"] == "2019-12-01"


class TestExtractAnyDate:
    def test_iso_date_in_camelcase_filename(self):
        assert extract_any_date("English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf") == "2018-06-02"

    def test_single_day_month_year(self):
        assert extract_any_date("English_2018_An-Altar-Not-To-Miss-9-June-2018.pdf") == "2018-06-09"

    def test_compact_yyyymmdd(self):
        assert extract_any_date("English_2018_20180623-Growing-Faith-in-God-Final-PPT.pdf") == "2018-06-23"

    def test_no_date_returns_none(self):
        assert extract_any_date("English_2018_SomeSermonNoDate.pdf") is None


class TestExtractTopicWords:
    def test_returns_content_words(self):
        words = extract_topic_words(
            "English_2018_28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide-updated.pdf"
        )
        assert "know" in words
        assert "enemy" in words
        assert "members" not in words
        assert "guide" not in words
        assert "elder" not in words

    def test_camelcase_split(self):
        words = extract_topic_words(
            "English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf"
        )
        assert "finishing" in words
        assert "well" in words
