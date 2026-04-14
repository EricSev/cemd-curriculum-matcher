import unittest

import pandas as pd

from curriculum_matcher.llm_rerank import (
    RERANK_RESPONSE_SCHEMA,
    build_candidate_shortlist,
    build_responses_api_body,
    build_prompt_record,
    extract_json_object,
    validate_rerank_response,
)


class LLMRerankTests(unittest.TestCase):
    def setUp(self):
        self.catalog_df = pd.DataFrame(
            [
                {
                    "product_identifier": "cand_a",
                    "product_name": "Wonders: Unspecified",
                    "series": "Wonders",
                    "publisher": "McGraw Hill Education",
                    "intended_grades": "K-5",
                    "copyright_year": "2022",
                    "product_type": "Core Curriculum",
                    "state_specific_version": "false",
                    "district_created": "false",
                    "embedded_assessment": "false",
                },
                {
                    "product_identifier": "cand_b",
                    "product_name": "Benchmark Advance California",
                    "series": "Benchmark Advance",
                    "publisher": "Benchmark Education Company",
                    "intended_grades": "K-6",
                    "copyright_year": "2022",
                    "product_type": "Core Curriculum",
                    "state_specific_version": "true",
                    "district_created": "false",
                    "embedded_assessment": "false",
                },
            ]
        )
        self.record_row = {
            "selection_identifier": "row_1",
            "state": "CA",
            "subject": "ELA",
            "grade": "4",
            "product_type_usage": "Core Curriculum",
            "product_name_raw": "Benchmark Workshop",
            "publisher_raw": "",
            "publisher_presence": "missing",
            "evidence_richness": "sparse",
            "usage_ambiguity": "implicit_usage",
            "state_specific_risk": "catalog_state_specific_expected",
            "placeholder_mapping": "catalog_unspecified",
            "assessment_slice": "not_assessment",
            "confidence_band": "low",
            "match_selected_strategy": "title_plus_publisher",
            "expected_match_id": "cand_b",
            "predicted_match_id": "cand_a",
            "top1_correct": False,
            "top3_correct": True,
            "pred_rank_1_id": "cand_a",
            "pred_rank_1_score": 0.51,
            "pred_rank_2_id": "cand_b",
            "pred_rank_2_score": 0.48,
            "pred_rank_3_id": "",
            "pred_rank_3_score": "",
        }

    def test_build_candidate_shortlist_uses_catalog_metadata(self):
        candidates = build_candidate_shortlist(self.record_row, self.catalog_df, topn_final=3)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].catalog_id, "cand_a")
        self.assertEqual(candidates[1].state_specific_version, "true")

    def test_build_prompt_record_contains_shortlist_and_prompts(self):
        prompt_record = build_prompt_record(self.record_row, self.catalog_df, topn_final=3)
        self.assertEqual(prompt_record["selection_identifier"], "row_1")
        self.assertEqual(prompt_record["candidate_ids"], ["cand_a", "cand_b"])
        self.assertIn("DISTRICT_ROW", prompt_record["user_prompt"])
        self.assertIn("SHORTLIST_CANDIDATES", prompt_record["user_prompt"])

    def test_extract_json_object_handles_fenced_json(self):
        response = extract_json_object(
            """```json
            {"decision":"select_candidate","selected_candidate_id":"cand_b","confidence":0.83}
            ```"""
        )
        self.assertEqual(response["selected_candidate_id"], "cand_b")

    def test_validate_rerank_response_normalizes_secondary_tags(self):
        validated = validate_rerank_response(
            {
                "decision": "select_candidate",
                "selected_candidate_id": "cand_b",
                "confidence": 0.83,
                "primary_reason": "likely_ui_catalog_selection_ambiguity",
                "secondary_tags": "catalog_unspecified|state_specific_variant_risk",
                "reasoning": "State-specific candidate is a better fit.",
            },
            candidate_ids=["cand_a", "cand_b"],
        )
        self.assertEqual(validated["selected_candidate_id"], "cand_b")
        self.assertEqual(
            validated["secondary_tags"],
            ["catalog_unspecified", "state_specific_variant_risk"],
        )

    def test_build_responses_api_body_uses_json_schema_format(self):
        prompt_record = build_prompt_record(self.record_row, self.catalog_df, topn_final=3)
        body = build_responses_api_body(
            prompt_record,
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        self.assertEqual(body["model"], "gpt-5.4-mini")
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertEqual(body["text"]["format"]["schema"], RERANK_RESPONSE_SCHEMA)
        self.assertEqual(body["reasoning"]["effort"], "low")


if __name__ == "__main__":
    unittest.main()
