import pytest
from src.ingestion.file_classifier import classify_file


class TestClassifyFile:
    def test_members_guide_hyphenated(self):
        assert classify_file("English_2018_28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide-updated.pdf") == "ng"

    def test_members27_guide(self):
        assert classify_file("English_2018_10-11-Nov-2018-Stewards-by-Ps-Hakan-members27-guide.pdf") == "ng"

    def test_leaders_guide(self):
        assert classify_file("English_2018_15-16-Dec-2018-And-the-Bleeding-Stopped-by-Elder-Chua-Seng-Lee-Leaders-Guide.pdf") == "ng"

    def test_members_copy(self):
        assert classify_file("English_2018_12-13-May-2018-A-Tale-of-4-Mothers-by-Gary-Koh-Members-Copy.pdf") == "ng"

    def test_camelcase_members_guide(self):
        assert classify_file("English_2015_FearOrFaith_eLVM_2015-12-19_20_MessageSummary_MembersGuide.pdf") == "ng"

    def test_notes_suffix(self):
        assert classify_file("English_2024_06-07-Jan-2024-The-Heart-of-Discipleship-by-SP-Chua-Seng-Lee-Members-Guide.pdf") == "ng"

    def test_pptx_extension(self):
        assert classify_file("English_2020_CHURCH-IS-FAMILY-Edric-Sng-12-Feb-2020-website.pptx") == "ps"

    def test_ppt_keyword(self):
        assert classify_file("English_2018_20180623-Growing-Faith-in-God-Final-PPT.pdf") == "ps"

    def test_compressed_slide(self):
        assert classify_file("English_2024_03-TOGETHER-AS-ONE-LUKE-10V1-3-20240127-PPT-FINAL-4-compressed.pdf") == "ps"

    def test_camelcase_abbreviated(self):
        assert classify_file("English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf") == "ps"

    def test_handout(self):
        assert classify_file("English_2018_EffectivePrayer-1-Principles_Handout.pdf") == "handout"

    def test_visual_summary(self):
        assert classify_file("English_2018_Visual-Summary_EffectivePrayer-6.pdf") == "handout"
