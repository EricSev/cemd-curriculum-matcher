import unittest

from curriculum_matcher.match_layers import choose_match_strategy, should_try_repairs


class MatchLayersTests(unittest.TestCase):
    def test_should_try_repairs_for_missing_or_low_confidence(self):
        self.assertTrue(should_try_repairs([], 0.55))
        self.assertTrue(
            should_try_repairs([{"catalog_id": "A", "final_score": 0.42}], 0.55)
        )
        self.assertFalse(
            should_try_repairs([{"catalog_id": "A", "final_score": 0.72}], 0.55)
        )

    def test_choose_match_strategy_prefers_stronger_fallback(self):
        result = choose_match_strategy(
            "primary",
            [{"catalog_id": "A", "final_score": 0.41}],
            {
                "publisher_as_title": [{"catalog_id": "B", "final_score": 0.64}],
                "title_publisher_swapped": [{"catalog_id": "C", "final_score": 0.52}],
            },
            confidence_threshold=0.55,
            replacement_margin=0.03,
        )
        self.assertEqual(result["selected_strategy"], "publisher_as_title")
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["matches"][0]["catalog_id"], "B")

    def test_choose_match_strategy_keeps_good_primary(self):
        result = choose_match_strategy(
            "primary",
            [{"catalog_id": "A", "final_score": 0.78}],
            {"publisher_as_title": [{"catalog_id": "B", "final_score": 0.74}]},
            confidence_threshold=0.55,
            replacement_margin=0.03,
        )
        self.assertEqual(result["selected_strategy"], "primary")
        self.assertFalse(result["used_fallback"])

    def test_choose_match_strategy_supports_strategy_specific_margin(self):
        result = choose_match_strategy(
            "primary",
            [{"catalog_id": "A", "final_score": 0.50}],
            {"title_plus_publisher": [{"catalog_id": "B", "final_score": 0.55}]},
            confidence_threshold=0.55,
            replacement_margin=0.03,
            strategy_replacement_margins={"title_plus_publisher": 0.12},
        )
        self.assertEqual(result["selected_strategy"], "primary")
        self.assertFalse(result["used_fallback"])


if __name__ == "__main__":
    unittest.main()
