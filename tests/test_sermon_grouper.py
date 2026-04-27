import pytest
from src.ingestion.sermon_grouper import group_sermon_files


class TestGroupSermonFiles:
    def test_pairs_cell_guide_with_matching_slide_by_date(self):
        files = [
            "English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf",
            "English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].cell_guide == files[0]
        assert files[1] in groups[0].slides

    def test_pairs_by_topic_when_slide_has_date_proximity(self):
        files = [
            "English_2018_09-10-June-2018-An-Altar-Not-to-Miss-by-Ps-Jason-Teo-Members-Guide.pdf",
            "English_2018_An-Altar-Not-To-Miss-9-June-2018.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].cell_guide == files[0]
        assert files[1] in groups[0].slides

    def test_standalone_slide_without_cell_guide(self):
        files = ["English_2018_20180623-Growing-Faith-in-God-Final-PPT.pdf"]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].cell_guide is None
        assert files[0] in groups[0].slides

    def test_standalone_cell_guide_without_slides(self):
        files = ["English_2018_28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide-updated.pdf"]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].cell_guide == files[0]
        assert groups[0].slides == []

    def test_does_not_pair_different_weekends(self):
        files = [
            "English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf",
            "English_2018_09-10-June-2018-An-Altar-Not-to-Miss-by-Ps-Jason-Teo-Members-Guide.pdf",
            "English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf",
            "English_2018_An-Altar-Not-To-Miss-9-June-2018.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 2
        cg_slides = {g.cell_guide: g.slides for g in groups}
        assert "English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf" in \
               cg_slides["English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf"]
        assert "English_2018_An-Altar-Not-To-Miss-9-June-2018.pdf" in \
               cg_slides["English_2018_09-10-June-2018-An-Altar-Not-to-Miss-by-Ps-Jason-Teo-Members-Guide.pdf"]

    def test_handouts_go_to_other(self):
        files = [
            "English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf",
            "English_2018_FinishingWell_Handout.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert "English_2018_FinishingWell_Handout.pdf" in groups[0].other
        assert "English_2018_FinishingWell_Handout.pdf" not in groups[0].slides
