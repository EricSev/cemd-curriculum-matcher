from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from curriculum_matcher.app import MatcherApp
from curriculum_matcher.evaluation import resolve_expected_id_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate QA challenge behavior against human-selected matches."
    )
    parser.add_argument("--input-file", required=True, help="Input CSV containing product_identifier as the human-selected match.")
    parser.add_argument("--catalog-file", required=True, help="Catalog CSV file.")
    parser.add_argument("--profile", choices=["fast", "accurate"], default="fast")
    parser.add_argument("--expected-id-column", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _cache_key(row: pd.Series) -> tuple[Any, ...]:
    selection_id = _normalize_id(row.get("selection_identifier"))
    if selection_id:
        return ("selection_identifier", selection_id)

    return (
        "row_fallback",
        _normalize_id(row.get("product_name_raw")),
        _normalize_id(row.get("publisher_raw")),
        _normalize_id(row.get("subject")),
        _normalize_id(row.get("product_type_usage")),
        _normalize_id(row.get("grade")),
        _normalize_id(row.get("state")),
    )


def main() -> None:
    args = parse_args()

    input_df = pd.read_csv(args.input_file, encoding="latin-1")
    expected_id_column = resolve_expected_id_column(
        input_df, requested_column=args.expected_id_column
    )

    app = MatcherApp(headless=True, profile=args.profile)
    catalog_df = pd.read_csv(args.catalog_file, encoding="latin-1")
    app.matcher.load_models(profile=args.profile)
    app.matcher.prepare_catalog(catalog_df)

    rows = []
    ai_result_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    recall_k = 40 if args.profile == "fast" else 60
    for _, row in input_df.iterrows():
        key = _cache_key(row)
        ai_result = ai_result_cache.get(key)
        if ai_result is None:
            ai_result = app.matcher.match_record_with_repairs(
                row, topn_stage1=recall_k, topn_final=3
            )
            ai_result_cache[key] = ai_result
        qa = app._assess_human_match(row, ai_result)
        result = row.to_dict()
        result.update(qa)
        result["match_selected_strategy"] = ai_result["selected_strategy"]
        result["match_used_fallback"] = ai_result["used_fallback"]
        result["match_variant_scores"] = json.dumps(ai_result["variant_scores"], sort_keys=True)
        expected_id = _normalize_id(row.get(expected_id_column))
        human_id = _normalize_id(row.get("product_identifier"))
        result["qa_expected_incorrect"] = bool(expected_id and human_id and expected_id != human_id)
        rows.append(result)

    result_df = pd.DataFrame(rows)
    summary = {
        "row_count": int(len(result_df)),
        "unique_ai_rows_evaluated": int(len(ai_result_cache)),
        "challenge_rate": round(float(result_df["human_match_challenge_flag"].fillna(False).mean()), 4)
        if "human_match_challenge_flag" in result_df.columns
        else 0.0,
    }

    if "qa_expected_incorrect" in result_df.columns and not result_df.empty:
        incorrect = result_df["qa_expected_incorrect"].fillna(False)
        challenged = result_df["human_match_challenge_flag"].fillna(False)
        tp = int((incorrect & challenged).sum())
        fp = int((~incorrect & challenged).sum())
        fn = int((incorrect & ~challenged).sum())
        summary["qa_metrics"] = {
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
            "incorrect_row_rate": round(float(incorrect.mean()), 4),
        }

    if "human_match_challenge_primary_reason" in result_df.columns and not result_df.empty:
        summary["challenge_primary_reason_counts"] = (
            result_df["human_match_challenge_primary_reason"]
            .fillna("missing")
            .value_counts()
            .sort_index()
            .to_dict()
        )
    if "qa_policy_slice" in result_df.columns and not result_df.empty:
        summary["qa_policy_slice_counts"] = (
            result_df["qa_policy_slice"].fillna("missing").value_counts().sort_index().to_dict()
        )

    if args.output_csv:
        output_csv_path = Path(args.output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv_path, index=False)
    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
