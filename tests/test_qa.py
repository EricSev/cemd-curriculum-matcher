import unittest

from curriculum_matcher.qa import build_human_match_review


class QATests(unittest.TestCase):
    def test_build_human_match_review_flags_stronger_alternative(self):
        review = build_human_match_review(
            human_match_id="A",
            human_scores={
                "human_match_final_score": 0.31,
                "human_match_name_semantic": 0.42,
                "human_match_name_fuzzy": 0.33,
                "human_match_publisher_score": 0.2,
            },
            ai_result={
                "selected_strategy": "publisher_as_title",
                "used_fallback": True,
                "matches": [{"catalog_id": "B", "final_score": 0.59}],
            },
            row_context={
                "evidence_richness": "medium",
                "placeholder_mapping": "catalog_unspecified",
                "state_specific_risk": "catalog_state_specific_expected",
                "assessment_slice": "not_assessment",
                "product_type_usage": "Core Curriculum",
            },
        )
        self.assertTrue(review["human_match_challenge_flag"])
        self.assertEqual(review["human_match_review_priority"], "high")
        self.assertEqual(
            review["human_match_challenge_primary_reason"],
            "likely_ui_catalog_selection_ambiguity",
        )
        self.assertIn("ai_prefers_alternative", review["human_match_challenge_reasons"])
        self.assertIn(
            "repaired_input_changed_prediction", review["human_match_challenge_reasons"]
        )

    def test_build_human_match_review_flags_low_support_without_ai_alternative(self):
        review = build_human_match_review(
            human_match_id="A",
            human_scores={
                "human_match_final_score": 0.4,
                "human_match_name_semantic": 0.61,
                "human_match_name_fuzzy": 0.55,
                "human_match_publisher_score": 0.7,
            },
            ai_result={
                "selected_strategy": "primary",
                "used_fallback": False,
                "matches": [],
            },
            row_context={"evidence_richness": "medium", "product_type_usage": "Core Curriculum"},
        )
        self.assertFalse(review["human_match_challenge_flag"])
        self.assertIn("low_support_score", review["human_match_challenge_reasons"])
        self.assertEqual(review["human_match_review_priority"], "low")
        self.assertEqual(review["human_match_challenge_primary_reason"], "well_supported")

    def test_build_human_match_review_flags_weak_title_low_support_without_ai_alternative(self):
        review = build_human_match_review(
            human_match_id="A",
            human_scores={
                "human_match_final_score": 0.4,
                "human_match_name_semantic": 0.42,
                "human_match_name_fuzzy": 0.33,
                "human_match_publisher_score": 0.7,
            },
            ai_result={
                "selected_strategy": "primary",
                "used_fallback": False,
                "matches": [],
            },
            row_context={"evidence_richness": "sparse", "product_type_usage": "Supplemental"},
        )
        self.assertTrue(review["human_match_challenge_flag"])
        self.assertIn("weak_title_evidence", review["human_match_challenge_reasons"])
        self.assertEqual(
            review["human_match_challenge_primary_reason"],
            "likely_evidence_sparsity",
        )
        self.assertEqual(review["human_match_review_priority"], "medium")

    def test_build_human_match_review_marks_supported_case(self):
        review = build_human_match_review(
            human_match_id="A",
            human_scores={
                "human_match_final_score": 0.72,
                "human_match_name_semantic": 0.81,
                "human_match_name_fuzzy": 0.76,
                "human_match_publisher_score": 0.9,
            },
            ai_result={
                "selected_strategy": "primary",
                "used_fallback": False,
                "matches": [{"catalog_id": "A", "final_score": 0.74}],
            },
            row_context={"evidence_richness": "rich", "product_type_usage": "Core Curriculum"},
        )
        self.assertFalse(review["human_match_challenge_flag"])
        self.assertEqual(review["human_match_review_priority"], "low")
        self.assertEqual(review["human_match_challenge_primary_reason"], "well_supported")
        self.assertIn("well_supported", review["human_match_challenge_reasons"])

    def test_build_human_match_review_routes_assessment_cases_to_assessment_taxonomy(self):
        review = build_human_match_review(
            human_match_id="A",
            human_scores={
                "human_match_final_score": 0.28,
                "human_match_name_semantic": 0.40,
                "human_match_name_fuzzy": 0.35,
                "human_match_publisher_score": 0.1,
            },
            ai_result={
                "selected_strategy": "primary",
                "used_fallback": False,
                "matches": [{"catalog_id": "B", "final_score": 0.51}],
            },
            row_context={
                "product_type_usage": "Assessment",
                "assessment_slice": "assessment_short_or_acronym_title",
                "usage_ambiguity": "assessment_naming_regime",
                "evidence_richness": "sparse",
            },
        )
        self.assertTrue(review["human_match_challenge_flag"])
        self.assertEqual(
            review["human_match_challenge_primary_reason"],
            "likely_assessment_alias_or_subtype_ambiguity",
        )
        self.assertIn(
            "assessment_alias_or_subtype_risk",
            review["human_match_challenge_secondary_tags"],
        )


if __name__ == "__main__":
    unittest.main()
