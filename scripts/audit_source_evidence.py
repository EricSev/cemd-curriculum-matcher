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

from curriculum_matcher.evaluation import resolve_expected_id_column
from curriculum_matcher.url_evidence import audit_row_urls, normalize_evidence_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit adopted_curriculum_source_url and source_document_link evidence."
    )
    parser.add_argument("--input-file", required=True, help="Input CSV to audit.")
    parser.add_argument("--output-csv", default=None, help="Optional row-level audit CSV.")
    parser.add_argument("--output-json", default=None, help="Optional summary JSON.")
    parser.add_argument("--catalog-file", default=None, help="Optional catalog CSV for expected-id comparisons.")
    parser.add_argument("--expected-id-column", default=None, help="Optional expected id column override.")
    parser.add_argument("--fetch-live", action="store_true", help="Fetch live metadata and lightweight content.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for live audits.")
    return parser.parse_args()


def _normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def main() -> None:
    args = parse_args()

    input_df = pd.read_csv(args.input_file, encoding="latin-1")
    if args.max_rows:
        input_df = input_df.head(args.max_rows).copy()

    expected_lookup = None
    expected_id_column = None
    if args.catalog_file:
        expected_id_column = resolve_expected_id_column(
            input_df, requested_column=args.expected_id_column
        )
        catalog_df = pd.read_csv(args.catalog_file, encoding="latin-1")
        expected_lookup = (
            catalog_df[
                ["product_identifier", "product_name", "series", "publisher"]
            ]
            .copy()
            .rename(
                columns={
                    "product_identifier": "expected_match_id",
                    "product_name": "expected_catalog_product_name",
                    "series": "expected_catalog_series",
                    "publisher": "expected_catalog_publisher",
                }
            )
        )
        expected_lookup["expected_match_id"] = expected_lookup["expected_match_id"].astype(str)

    rows = []
    for _, row in input_df.iterrows():
        row_dict = row.to_dict()
        audits = audit_row_urls(row_dict, fetch_live=args.fetch_live)
        flattened = row_dict.copy()
        evidence_parts = []
        for audit in audits:
            prefix = audit.field_name
            flattened[f"{prefix}_url_type"] = audit.url_type
            flattened[f"{prefix}_domain"] = audit.domain
            flattened[f"{prefix}_document_filename"] = audit.document_filename
            flattened[f"{prefix}_fetch_status"] = audit.fetch_status
            flattened[f"{prefix}_http_status"] = audit.http_status
            flattened[f"{prefix}_content_type"] = audit.content_type
            flattened[f"{prefix}_final_url"] = audit.final_url
            flattened[f"{prefix}_page_title"] = audit.page_title
            flattened[f"{prefix}_visible_text_excerpt"] = audit.visible_text_excerpt
            evidence_parts.extend(
                [
                    audit.url,
                    audit.final_url,
                    audit.document_filename,
                    audit.page_title,
                    audit.visible_text_excerpt,
                ]
            )
        flattened["combined_evidence_text"] = normalize_evidence_text(*evidence_parts)
        rows.append(flattened)

    audit_df = pd.DataFrame(rows)

    if expected_lookup is not None and expected_id_column:
        audit_df["expected_match_id"] = audit_df[expected_id_column].map(_normalize_id)
        audit_df = audit_df.merge(expected_lookup, on="expected_match_id", how="left")
        audit_df["evidence_mentions_expected_product"] = audit_df.apply(
            lambda row: normalize_evidence_text(row.get("expected_catalog_product_name", ""))
            in row["combined_evidence_text"]
            if row.get("expected_catalog_product_name")
            else False,
            axis=1,
        )
        audit_df["evidence_mentions_expected_series"] = audit_df.apply(
            lambda row: normalize_evidence_text(row.get("expected_catalog_series", ""))
            in row["combined_evidence_text"]
            if row.get("expected_catalog_series")
            else False,
            axis=1,
        )
        audit_df["evidence_mentions_expected_publisher"] = audit_df.apply(
            lambda row: normalize_evidence_text(row.get("expected_catalog_publisher", ""))
            in row["combined_evidence_text"]
            if row.get("expected_catalog_publisher")
            else False,
            axis=1,
        )

    summary = {
        "row_count": int(len(audit_df)),
        "fetch_live": bool(args.fetch_live),
        "coverage": {
            "adopted_curriculum_source_url": round(
                float(
                    audit_df["adopted_curriculum_source_url"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                    .mean()
                ),
                4,
            )
            if "adopted_curriculum_source_url" in audit_df.columns
            else 0.0,
            "source_document_link": round(
                float(
                    audit_df["source_document_link"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                    .mean()
                ),
                4,
            )
            if "source_document_link" in audit_df.columns
            else 0.0,
        },
        "url_type_counts": {
            column: audit_df[column].value_counts(dropna=False).to_dict()
            for column in audit_df.columns
            if column.endswith("_url_type")
        },
        "fetch_status_counts": {
            column: audit_df[column].value_counts(dropna=False).to_dict()
            for column in audit_df.columns
            if column.endswith("_fetch_status")
        },
    }

    for key in [
        "evidence_mentions_expected_product",
        "evidence_mentions_expected_series",
        "evidence_mentions_expected_publisher",
    ]:
        if key in audit_df.columns:
            summary[key] = round(float(audit_df[key].mean()), 4)

    if args.output_csv:
        output_csv_path = Path(args.output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(output_csv_path, index=False)
    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
