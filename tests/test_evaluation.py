import unittest

import pandas as pd

from curriculum_matcher.evaluation import (
    _available_rank_cutoffs,
    _confidence_band,
    _cross_slice_accuracy,
    _derive_assessment_slice,
    _derive_evidence_richness,
    _derive_placeholder_mapping,
    _derive_state_specific_risk,
    _derive_usage_ambiguity,
    _find_expected_rank,
    _group_ranking_metrics,
    _ndcg_at_k,
    _reciprocal_rank,
    resolve_expected_id_column,
)


class EvaluationHelpersTests(unittest.TestCase):
    def test_resolve_expected_id_column_prefers_explicit_match(self):
        df = pd.DataFrame(columns=["expected_product_identifier", "product_identifier"])
        self.assertEqual(
            resolve_expected_id_column(df, requested_column="product_identifier"),
            "product_identifier",
        )

    def test_resolve_expected_id_column_uses_fallback(self):
        df = pd.DataFrame(columns=["selection_identifier", "expected_product_identifier"])
        self.assertEqual(
            resolve_expected_id_column(df),
            "expected_product_identifier",
        )

    def test_find_expected_rank_returns_match_position(self):
        matches = [
            {"catalog_id": "A"},
            {"catalog_id": "B"},
            {"catalog_id": "C"},
        ]
        self.assertEqual(_find_expected_rank(matches, "B"), 2)
        self.assertIsNone(_find_expected_rank(matches, "Z"))

    def test_confidence_band_thresholds(self):
        self.assertEqual(_confidence_band(0.85), "high")
        self.assertEqual(_confidence_band(0.65), "medium")
        self.assertEqual(_confidence_band(0.35), "low")
        self.assertEqual(_confidence_band(0.10), "very_low")
        self.assertEqual(_confidence_band(None), "no_prediction")

    def test_derive_evidence_richness_marks_rich_with_publisher_and_variant(self):
        record = {
            "product_name_raw": "Benchmark Advance California Edition 2022",
            "publisher_raw": "Benchmark Education",
        }
        catalog_row = {"state_specific_version": "true"}
        self.assertEqual(_derive_evidence_richness(record, catalog_row), "rich")

    def test_derive_usage_ambiguity_treats_assessment_as_separate_regime(self):
        record = {
            "product_type_usage": "Assessment",
            "product_name_raw": "MAP",
        }
        self.assertEqual(_derive_usage_ambiguity(record), "assessment_naming_regime")

    def test_derive_state_specific_risk_prefers_expected_catalog_flag(self):
        record = {"state": "WA"}
        catalog_row = {"state_specific_version": "true"}
        self.assertEqual(
            _derive_state_specific_risk(record, catalog_row),
            "catalog_state_specific_expected",
        )

    def test_derive_placeholder_mapping_detects_unspecified_catalog(self):
        record = {"product_name_raw": "Journeys"}
        catalog_row = {"product_name": "Journeys: Unspecified", "series": "Journeys"}
        self.assertEqual(
            _derive_placeholder_mapping(record, catalog_row),
            "catalog_unspecified",
        )

    def test_derive_assessment_slice_prefers_state_specific_then_short_title(self):
        state_specific_record = {
            "product_type_usage": "Assessment",
            "product_name_raw": "STAAR",
            "publisher_raw": "",
        }
        self.assertEqual(
            _derive_assessment_slice(
                state_specific_record, {"state_specific_version": "true"}
            ),
            "assessment_state_specific_expected",
        )

        short_title_record = {
            "product_type_usage": "Assessment",
            "product_name_raw": "MAP",
            "publisher_raw": "",
        }
        self.assertEqual(
            _derive_assessment_slice(short_title_record, {"state_specific_version": "false"}),
            "assessment_short_or_acronym_title",
        )

    def test_cross_slice_accuracy_builds_combined_labels(self):
        df = pd.DataFrame(
            [
                {
                    "product_type_usage": "Assessment",
                    "assessment_slice": "assessment_short_or_acronym_title",
                    "top1_correct": False,
                    "top3_correct": True,
                    "top10_correct": True,
                    "reciprocal_rank": 0.5,
                    "ndcg_at_10": 0.6309,
                },
                {
                    "product_type_usage": "Assessment",
                    "assessment_slice": "assessment_short_or_acronym_title",
                    "top1_correct": True,
                    "top3_correct": True,
                    "top10_correct": True,
                    "reciprocal_rank": 1.0,
                    "ndcg_at_10": 1.0,
                },
            ]
        )

        metrics = _cross_slice_accuracy(df, "product_type_usage", "assessment_slice")
        self.assertIn(
            "Assessment | assessment_short_or_acronym_title",
            metrics,
        )
        self.assertEqual(
            metrics["Assessment | assessment_short_or_acronym_title"]["count"], 2
        )
        self.assertEqual(
            metrics["Assessment | assessment_short_or_acronym_title"]["top1_accuracy"],
            0.5,
        )
        self.assertEqual(
            metrics["Assessment | assessment_short_or_acronym_title"]["top10_recall"],
            1.0,
        )
        self.assertEqual(
            metrics["Assessment | assessment_short_or_acronym_title"]["mrr"],
            0.75,
        )

    def test_available_rank_cutoffs_uses_existing_rank_columns(self):
        df = pd.DataFrame(columns=["pred_rank_1_id", "pred_rank_3_id", "pred_rank_10_id"])
        self.assertEqual(_available_rank_cutoffs(df), [1, 3, 10])

    def test_group_ranking_metrics_includes_rank_metrics(self):
        df = pd.DataFrame(
            [
                {
                    "top1_correct": True,
                    "top3_correct": True,
                    "top10_correct": True,
                    "reciprocal_rank": 1.0,
                    "ndcg_at_10": 1.0,
                },
                {
                    "top1_correct": False,
                    "top3_correct": True,
                    "top10_correct": True,
                    "reciprocal_rank": 0.5,
                    "ndcg_at_10": 0.6309,
                },
            ]
        )
        metrics = _group_ranking_metrics(df)
        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["top1_accuracy"], 0.5)
        self.assertEqual(metrics["top3_recall"], 1.0)
        self.assertEqual(metrics["top10_recall"], 1.0)
        self.assertEqual(metrics["mrr"], 0.75)

    def test_rank_metric_helpers(self):
        self.assertEqual(_reciprocal_rank(1), 1.0)
        self.assertEqual(_reciprocal_rank(4), 0.25)
        self.assertEqual(_reciprocal_rank(None), 0.0)
        self.assertEqual(_ndcg_at_k(1, 10), 1.0)
        self.assertEqual(_ndcg_at_k(2, 10), 0.6309)
        self.assertEqual(_ndcg_at_k(11, 10), 0.0)


if __name__ == "__main__":
    unittest.main()
