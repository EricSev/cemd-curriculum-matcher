import unittest
from unittest import mock

import numpy as np
import rapidfuzz

from curriculum_matcher.app import EnhancedCurriculumMatcherV312


class MatcherCoreTests(unittest.TestCase):
    def setUp(self):
        self.matcher = EnhancedCurriculumMatcherV312(log_callback=lambda *_: None)

    def test_normalize_semantic_expands_common_aliases(self):
        normalized = self.matcher._normalize("ELA & Maths", for_semantic=True)
        self.assertEqual(normalized, "english language arts and mathematics")

    def test_publisher_canonical_handles_common_alias(self):
        self.assertEqual(
            self.matcher._publisher_canonical("McGraw-Hill Education"),
            "mcgraw hill",
        )

    def test_publisher_canonical_keeps_pearson_distinct_from_savvas(self):
        self.assertEqual(self.matcher._publisher_canonical("Pearson"), "pearson")

    def test_publisher_canonical_handles_first_pass_alias_variants(self):
        self.assertEqual(self.matcher._publisher_canonical("MGH"), "mcgraw hill")
        self.assertEqual(
            self.matcher._publisher_canonical("Savvas"),
            "savvas learning company",
        )
        self.assertEqual(
            self.matcher._publisher_canonical("NGL/Cengage"),
            "national geographic learning cengage",
        )
        self.assertEqual(
            self.matcher._publisher_canonical("Benchmark Education Co."),
            "benchmark education company",
        )
        self.assertEqual(
            self.matcher._publisher_canonical("Pearson/Prentice Hall"),
            "pearson education",
        )
        self.assertEqual(
            self.matcher._publisher_canonical("Holt, Rinehart and Winston"),
            "hmh",
        )

    def test_publisher_score_matches_safe_alias_variants(self):
        self.matcher._rapidfuzz = rapidfuzz
        self.assertEqual(
            self.matcher._publisher_score(
                "Benchmark Education Co.",
                "Benchmark Education Company",
            ),
            1.0,
        )
        self.assertEqual(
            self.matcher._publisher_score(
                "Pearson/Prentice Hall",
                "Pearson Education",
            ),
            1.0,
        )

    def test_normalize_product_title_expands_safe_acronyms(self):
        self.assertEqual(
            self.matcher._normalize_product_title("CAASPP", for_semantic=True),
            "california assessment of student performance and progress caaspp",
        )
        self.assertEqual(
            self.matcher._normalize_product_title("ELPAC", for_semantic=True),
            "english language proficiency assessments for california elpac",
        )
        self.assertEqual(
            self.matcher._normalize_product_title("NAEP", for_semantic=True),
            "national assessment of educational progress naep",
        )
        self.assertEqual(
            self.matcher._normalize_product_title("STAAR", for_semantic=True),
            "state of texas assessments of academic readiness staar",
        )
        self.assertEqual(
            self.matcher._normalize_product_title("TELPAS", for_semantic=True),
            "texas english language proficiency assessment system telpas",
        )
        self.assertEqual(
            self.matcher._normalize_product_title("DRA: K-5", for_semantic=True),
            "developmental reading assessment dra k 5",
        )
        self.assertEqual(
            self.matcher._normalize_product_title("TS GOLD", for_semantic=True),
            "teaching strategies gold ts gold",
        )
        self.assertEqual(
            self.matcher._normalize_product_title("Modern Chemistry GA Ed", for_semantic=True),
            "modern chemistry georgia edition",
        )
        self.assertEqual(
            self.matcher._normalize_product_title("EnVision Algebra 1, SC 1st Edition", for_semantic=True),
            "envision algebra 1 south carolina first edition",
        )

    def test_normalize_product_title_keeps_ambiguous_short_tokens_unexpanded(self):
        self.assertEqual(self.matcher._normalize_product_title("MAP"), "map")
        self.assertEqual(self.matcher._normalize_product_title("ACCESS"), "access")
        self.assertEqual(self.matcher._normalize_product_title("FAST"), "fast")

    def test_name_scores_benefit_from_safe_acronym_expansion(self):
        self.matcher._rapidfuzz = rapidfuzz
        input_text = "NAEP"
        candidate_text = "National Assessment of Educational Progress (NAEP)"
        raw_fuzz = (
            self.matcher._rapidfuzz.fuzz.token_set_ratio(
                self.matcher._normalize(input_text),
                self.matcher._normalize(candidate_text),
            )
            / 100.0
        )
        expanded_fuzz = (
            self.matcher._rapidfuzz.fuzz.token_set_ratio(
                self.matcher._normalize_product_title(input_text),
                self.matcher._normalize_product_title(candidate_text),
            )
            / 100.0
        )
        self.assertGreater(expanded_fuzz, raw_fuzz)

    def test_extract_first_year_finds_embedded_year(self):
        self.assertEqual(self.matcher._extract_first_year("Edition 2019-2021"), 2019)

    def test_parse_grade_range_handles_prek(self):
        self.assertEqual(self.matcher._parse_grade_range("Pre-K"), (0, 0))

    def test_grade_score_rewards_overlap(self):
        self.assertEqual(self.matcher._grade_score("6", "6-8"), 1.0)

    def test_grade_score_penalizes_distance(self):
        self.assertEqual(self.matcher._grade_score("3", "6-8"), 0.0)

    def test_build_char_ngram_tokens_uses_padded_character_windows(self):
        self.assertEqual(
            self.matcher._build_char_ngram_tokens("MAP"),
            ["__m", "_ma", "map", "ap_", "p__"],
        )

    def test_blend_stage1_scores_defaults_to_char_ngram_baseline(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            matcher = EnhancedCurriculumMatcherV312(log_callback=lambda *_: None)
        blended = matcher._blend_stage1_scores(
            np.array([0.2, 0.4]),
            np.array([0.8, 0.6]),
            np.array([0.9, 0.1]),
        )
        np.testing.assert_allclose(blended, np.array([0.675, 0.425]))

    def test_blend_stage1_scores_uses_direct_blend_when_explicitly_requested(self):
        with mock.patch.dict(
            "os.environ",
            {"CURRICULUM_MATCHER_RETRIEVAL_EXPERIMENT": "direct"},
            clear=False,
        ):
            matcher = EnhancedCurriculumMatcherV312(log_callback=lambda *_: None)
        blended = matcher._blend_stage1_scores(
            np.array([0.2, 0.4]),
            np.array([0.8, 0.6]),
            np.array([0.9, 0.1]),
        )
        np.testing.assert_allclose(blended, np.array([0.5, 0.5]))

    def test_cross_encoder_rerank_reorders_candidates_when_enabled(self):
        matcher = EnhancedCurriculumMatcherV312(
            log_callback=lambda *_: None,
            rerank_experiment="cross_encoder",
        )

        class FakeCrossEncoder:
            def predict(self, pairs, show_progress_bar=False):
                return np.array([0.1, 0.9])

        matcher.cross_encoder = FakeCrossEncoder()
        record = {
            "product_name_raw": "Benchmark Advance",
            "publisher_raw": "Benchmark Education",
            "grade": "4",
        }
        results = [
            {"catalog_id": "cand_a", "final_score": 0.8},
            {"catalog_id": "cand_b", "final_score": 0.7},
        ]
        candidate_lookup = {
            "cand_a": {
                "product_name": "Wrong Candidate",
                "series": "",
                "publisher": "Other",
                "publisher_prior": "",
                "intended_grades": "4",
                "copyright_year": "2020",
            },
            "cand_b": {
                "product_name": "Benchmark Advance",
                "series": "",
                "publisher": "Benchmark Education Company",
                "publisher_prior": "",
                "intended_grades": "4",
                "copyright_year": "2022",
            },
        }
        reranked = matcher._rerank_with_cross_encoder(record, results, candidate_lookup)
        self.assertEqual(
            [item["catalog_id"] for item in reranked],
            ["cand_b", "cand_a"],
        )
        self.assertGreater(
            reranked[0]["cross_encoder_score"],
            reranked[1]["cross_encoder_score"],
        )


if __name__ == "__main__":
    unittest.main()
