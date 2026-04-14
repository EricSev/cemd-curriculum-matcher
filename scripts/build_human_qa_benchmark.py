from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a silver human-match QA benchmark from benchmark rows and evaluation outputs."
    )
    parser.add_argument("--benchmark-file", required=True, help="Benchmark CSV with expected ids.")
    parser.add_argument("--records-csv", required=True, help="Per-record evaluation CSV with pred_rank_* ids.")
    parser.add_argument("--output-csv", required=True, help="Output QA benchmark CSV.")
    parser.add_argument("--catalog-file", default=None, help="Optional catalog CSV for expected/predicted metadata.")
    parser.add_argument("--output-json", default=None, help="Optional benchmark summary JSON.")
    parser.add_argument("--max-negatives-per-row", type=int, default=2, help="Maximum synthetic incorrect human matches to emit per row.")
    return parser.parse_args()


def _normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _label_strength(row: dict) -> str:
    if row.get("qa_case_type") == "expected_match":
        return "gold_positive"
    if row.get("qa_case_type") == "pred_rank_1_incorrect":
        return "silver_strong_negative"
    return "silver_negative"


def _benchmark_slice(row) -> str:
    if str(row.get("product_type_usage", "")).strip() == "Assessment":
        return "assessment"
    if str(row.get("placeholder_mapping", "")).strip() == "catalog_unspecified":
        return "catalog_unspecified"
    if str(row.get("state_specific_risk", "")).strip() in {
        "catalog_state_specific_expected",
        "adoption_state_high_risk",
    }:
        return "state_specific_risk"
    if str(row.get("evidence_richness", "")).strip() in {
        "sparse",
        "policy_placeholder_no_info",
    }:
        return "sparse_evidence"
    if row.get("publisher_raw") is None or str(row.get("publisher_raw")).strip() == "":
        return "missing_publisher"
    if row.get("match_used_fallback"):
        return "fallback_selected"
    if row.get("adopted_curriculum_source_url") and str(row.get("adopted_curriculum_source_url")).strip():
        return "has_adopted_url"
    return "general"


def _policy_slice(row) -> str:
    if str(row.get("assessment_slice", "")).strip() not in {"", "not_assessment"}:
        return str(row.get("assessment_slice")).strip()
    if str(row.get("placeholder_mapping", "")).strip():
        return str(row.get("placeholder_mapping")).strip()
    if str(row.get("state_specific_risk", "")).strip():
        return str(row.get("state_specific_risk")).strip()
    if str(row.get("evidence_richness", "")).strip():
        return str(row.get("evidence_richness")).strip()
    return "unclassified"


def main() -> None:
    args = parse_args()
    benchmark_df = pd.read_csv(args.benchmark_file, encoding="latin-1")
    records_df = pd.read_csv(args.records_csv)
    catalog_lookup = None
    if args.catalog_file:
        catalog_df = pd.read_csv(args.catalog_file, encoding="latin-1")
        catalog_lookup = catalog_df[
            ["product_identifier", "product_name", "series", "publisher", "intended_grades"]
        ].copy()
        catalog_lookup["product_identifier"] = catalog_lookup["product_identifier"].astype(str)

    merged = benchmark_df.copy()
    for column in records_df.columns:
        if column not in merged.columns:
            merged[column] = records_df[column]

    rows = []
    for _, row in merged.iterrows():
        expected_id = _normalize_id(row.get("expected_match_id") or row.get("expected_product_identifier"))
        if not expected_id:
            continue

        positive = row.to_dict()
        positive["product_identifier"] = expected_id
        positive["qa_label"] = "correct_human_match"
        positive["qa_source"] = "expected_match"
        positive["qa_case_type"] = "expected_match"
        positive["qa_label_strength"] = _label_strength(positive)
        positive["qa_benchmark_slice"] = _benchmark_slice(positive)
        positive["qa_policy_slice"] = _policy_slice(positive)
        rows.append(positive)

        negative_count = 0
        for rank in range(1, 4):
            candidate_id = _normalize_id(row.get(f"pred_rank_{rank}_id"))
            if not candidate_id or candidate_id == expected_id:
                continue
            negative = row.to_dict()
            negative["product_identifier"] = candidate_id
            negative["qa_label"] = "incorrect_human_match"
            negative["qa_source"] = f"pred_rank_{rank}"
            negative["qa_case_type"] = f"pred_rank_{rank}_incorrect"
            negative["qa_label_strength"] = _label_strength(negative)
            negative["qa_benchmark_slice"] = _benchmark_slice(negative)
            negative["qa_policy_slice"] = _policy_slice(negative)
            rows.append(negative)
            negative_count += 1
            if negative_count >= args.max_negatives_per_row:
                break

    output_df = pd.DataFrame(rows)
    if catalog_lookup is not None and not output_df.empty:
        output_df["product_identifier"] = output_df["product_identifier"].map(_normalize_id)
        output_df = output_df.merge(
            catalog_lookup.add_prefix("human_catalog_").rename(
                columns={"human_catalog_product_identifier": "product_identifier"}
            ),
            on="product_identifier",
            how="left",
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    summary = {
        "row_count": int(len(output_df)),
        "label_counts": output_df["qa_label"].value_counts(dropna=False).sort_index().to_dict()
        if not output_df.empty
        else {},
        "case_type_counts": output_df["qa_case_type"].value_counts(dropna=False).sort_index().to_dict()
        if not output_df.empty
        else {},
        "label_strength_counts": output_df["qa_label_strength"].value_counts(dropna=False).sort_index().to_dict()
        if not output_df.empty
        else {},
        "slice_counts": output_df["qa_benchmark_slice"].value_counts(dropna=False).sort_index().to_dict()
        if not output_df.empty
        else {},
        "policy_slice_counts": output_df["qa_policy_slice"].value_counts(dropna=False).sort_index().to_dict()
        if not output_df.empty
        else {},
    }
    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
