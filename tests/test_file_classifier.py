import pytest
from src.ingestion.file_classifier import classify_file


class TestClassifyFile:
    # Cell guide variants
    def test_members_guide_hyphenated(self):
        assert classify_file("English_2018_28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide-updated.pdf") == "cell_guide"

    def test_members27_guide(self):
        assert classify_file("English_2018_10-11-Nov-2018-Stewards-by-Ps-Hakan-Gabrielsson-members27-guide.pdf") == "cell_guide"

    def test_leaders_guide(self):
        assert classify_file("English_2018_15-16-Dec-2018-And-the-Bleeding-Stopped-by-Elder-Chua-Seng-Lee-Leaders-Guide.pdf") == "cell_guide"

    def test_members_copy(self):
        assert classify_file("English_2018_12-13-May-2018-A-Tale-of-4-Mothers-by-Gary-and-Joanna-Koh-Members-Copy.pdf") == "cell_guide"

    def test_camelcase_members_guide(self):
        assert classify_file("English_2015_FearOrFaith_eLVM_2015-12-19_20_MessageSummary_MembersGuide.pdf") == "cell_guide"

    def test_message_summary_members(self):
        assert classify_file("English_2015_ChooseWisely_PsAndrewTan_2015-12-05_06_MessageSummary_MembersGuide.pdf") == "cell_guide"

    # Sermon slides variants
    def test_pptx_extension(self):
        assert classify_file("English_2020_CHURCH-IS-FAMILY-Edric-Sng-12-Feb-2020-website.pptx") == "sermon_slides"

    def test_ppt_keyword(self):
        assert classify_file("English_2018_20180623-Growing-Faith-in-God-Final-PPT.pdf") == "sermon_slides"

    def test_camelcase_abbreviated_pdf(self):
        assert classify_file("English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf") == "sermon_slides"

    def test_camelcase_with_elvm(self):
        assert classify_file("English_2018_WhyTheCross_eLVM_2018-03-24_25.pdf") == "sermon_slides"

    # Other variants
    def test_handout(self):
        assert classify_file("English_2018_EffectivePrayer-1-Principles_Handout.pdf") == "other"

    def test_visual_summary(self):
        assert classify_file("English_2018_VisualSummary_EP5_BlessingsCurses.pdf") == "other"

    def test_visual_summary_hyphenated(self):
        assert classify_file("English_2018_Visual-Summary_EffectivePrayer-6.pdf") == "other"
