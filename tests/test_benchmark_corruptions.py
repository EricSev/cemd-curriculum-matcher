import unittest

import pandas as pd

from curriculum_matcher.benchmark_corruptions import build_corruption_assets


class BenchmarkCorruptionTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            [
                {
                    "product_name_raw": "StudySync",
                    "publisher_raw": "McGraw Hill",
                    "grade": "6",
                    "collection_grades": "6,7,8",
                    "expected_product_identifier": "A",
                },
                {
                    "product_name_raw": "Benchmark Advance",
                    "publisher_raw": "",
                    "grade": "K",
                    "collection_grades": "TK,K,1",
                    "expected_product_identifier": "B",
                },
            ]
        )

    def test_build_corruption_assets_includes_expected_slices(self):
        assets = build_corruption_assets(self.df)
        self.assertEqual(
            set(assets.keys()),
            {
                "clean_reference",
                "title_publisher_swapped",
                "title_in_publisher_title_blank",
                "publisher_in_title_publisher_blank",
                "grade_corrupted_or_missing",
            },
        )

    def test_title_publisher_swapped_moves_values(self):
        assets = build_corruption_assets(self.df)
        swapped = assets["title_publisher_swapped"]
        self.assertEqual(swapped.loc[0, "product_name_raw"], "McGraw Hill")
        self.assertEqual(swapped.loc[0, "publisher_raw"], "StudySync")
        self.assertEqual(swapped.loc[0, "original_product_name_raw"], "StudySync")
        self.assertEqual(swapped.loc[0, "original_publisher_raw"], "McGraw Hill")

    def test_title_in_publisher_blanks_title(self):
        assets = build_corruption_assets(self.df)
        corrupted = assets["title_in_publisher_title_blank"]
        self.assertEqual(corrupted.loc[0, "product_name_raw"], "")
        self.assertEqual(corrupted.loc[0, "publisher_raw"], "StudySync")

    def test_publisher_in_title_blanks_publisher(self):
        assets = build_corruption_assets(self.df)
        corrupted = assets["publisher_in_title_publisher_blank"]
        self.assertEqual(corrupted.loc[0, "product_name_raw"], "McGraw Hill")
        self.assertEqual(corrupted.loc[0, "publisher_raw"], "")

    def test_grade_corrupted_slice_changes_or_blanks_grade(self):
        assets = build_corruption_assets(self.df)
        corrupted = assets["grade_corrupted_or_missing"]
        self.assertEqual(corrupted.loc[0, "grade"], "1")
        self.assertEqual(corrupted.loc[1, "grade"], "8")
        self.assertEqual(corrupted.loc[1, "original_grade"], "K")


if __name__ == "__main__":
    unittest.main()
