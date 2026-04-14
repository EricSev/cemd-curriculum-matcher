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
        description="Run a bounded live source-evidence fetch pilot on adopted curriculum URLs."
    )
    parser.add_argument("--input-file", required=True, help="Input CSV to sample from.")
    parser.add_argument("--catalog-file", required=True, help="Catalog CSV for expected match metadata.")
    parser.add_argument("--output-csv", required=True, help="Row-level pilot output CSV.")
    parser.add_argument("--output-json", required=True, help="Summary JSON output.")
    parser.add_argument("--output-md", required=True, help="Markdown report output.")
    parser.add_argument("--expected-id-column", default=None)
    parser.add_argument("--max-per-type", type=int, default=5, help="Maximum rows to sample per URL type.")
    return parser.parse_args()


def _normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    input_df = pd.read_csv(args.input_file, encoding="latin-1")
    expected_id_column = resolve_expected_id_column(
        input_df, requested_column=args.expected_id_column
    )
    catalog_df = pd.read_csv(args.catalog_file, encoding="latin-1")
    catalog_lookup = catalog_df[
        ["product_identifier", "product_name", "series", "publisher"]
    ].copy()
    catalog_lookup["product_identifier"] = catalog_lookup["product_identifier"].astype(str)

    candidate_rows = input_df[
        input_df["adopted_curriculum_source_url"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    candidate_rows["adopted_curriculum_source_url_url_type"] = ""

    from curriculum_matcher.url_evidence import classify_url

    candidate_rows["adopted_curriculum_source_url_url_type"] = candidate_rows[
        "adopted_curriculum_source_url"
    ].fillna("").astype(str).map(classify_url)

    sampled_frames = []
    for url_type, group in candidate_rows.groupby("adopted_curriculum_source_url_url_type", dropna=False):
        sampled_frames.append(group.head(args.max_per_type).copy())
    pilot_df = pd.concat(sampled_frames, ignore_index=True) if sampled_frames else candidate_rows.head(0).copy()

    pilot_df["expected_match_id"] = pilot_df[expected_id_column].map(_normalize_id)
    pilot_df = pilot_df.merge(
        catalog_lookup.rename(
            columns={
                "product_identifier": "expected_match_id",
                "product_name": "expected_catalog_product_name",
                "series": "expected_catalog_series",
                "publisher": "expected_catalog_publisher",
            }
        ),
        on="expected_match_id",
        how="left",
    )

    rows = []
    for _, row in pilot_df.iterrows():
        flattened = row.to_dict()
        audits = audit_row_urls(
            row.to_dict(),
            fetch_live=True,
            timeout_seconds=8,
            max_bytes=65536,
        )
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
        evidence_text = normalize_evidence_text(*evidence_parts)
        flattened["combined_evidence_text"] = evidence_text
        flattened["evidence_mentions_expected_product"] = (
            normalize_evidence_text(flattened.get("expected_catalog_product_name", "")) in evidence_text
            if flattened.get("expected_catalog_product_name")
            else False
        )
        flattened["evidence_mentions_expected_series"] = (
            normalize_evidence_text(flattened.get("expected_catalog_series", "")) in evidence_text
            if flattened.get("expected_catalog_series")
            else False
        )
        flattened["evidence_mentions_expected_publisher"] = (
            normalize_evidence_text(flattened.get("expected_catalog_publisher", "")) in evidence_text
            if flattened.get("expected_catalog_publisher")
            else False
        )
        rows.append(flattened)

    result_df = pd.DataFrame(rows)
    summary = {
        "row_count": int(len(result_df)),
        "url_type_counts": result_df["adopted_curriculum_source_url_url_type"].value_counts(dropna=False).to_dict()
        if not result_df.empty
        else {},
        "fetch_status_counts": result_df["adopted_curriculum_source_url_fetch_status"].value_counts(dropna=False).to_dict()
        if not result_df.empty
        else {},
        "live_fetch_success_rate": round(
            float(result_df["adopted_curriculum_source_url_fetch_status"].eq("success").mean()),
            4,
        )
        if not result_df.empty
        else 0.0,
        "evidence_mentions_expected_product": round(
            float(result_df["evidence_mentions_expected_product"].mean()), 4
        )
        if "evidence_mentions_expected_product" in result_df.columns and not result_df.empty
        else 0.0,
        "evidence_mentions_expected_series": round(
            float(result_df["evidence_mentions_expected_series"].mean()), 4
        )
        if "evidence_mentions_expected_series" in result_df.columns and not result_df.empty
        else 0.0,
        "evidence_mentions_expected_publisher": round(
            float(result_df["evidence_mentions_expected_publisher"].mean()), 4
        )
        if "evidence_mentions_expected_publisher" in result_df.columns and not result_df.empty
        else 0.0,
    }

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Live URL Pilot Report",
        "",
        f"- row count: {summary['row_count']}",
        f"- live fetch success rate: {summary['live_fetch_success_rate']}",
        f"- evidence mentions expected product: {summary['evidence_mentions_expected_product']}",
        f"- evidence mentions expected series: {summary['evidence_mentions_expected_series']}",
        f"- evidence mentions expected publisher: {summary['evidence_mentions_expected_publisher']}",
        "",
        "## URL Type Counts",
        "",
        _markdown_table(pd.DataFrame(summary["url_type_counts"].items(), columns=["url_type", "count"]))
        if summary["url_type_counts"]
        else "_none_",
        "",
        "## Fetch Status Counts",
        "",
        _markdown_table(pd.DataFrame(summary["fetch_status_counts"].items(), columns=["fetch_status", "count"]))
        if summary["fetch_status_counts"]
        else "_none_",
    ]

    success_sample = result_df.loc[
        result_df["adopted_curriculum_source_url_fetch_status"].eq("success"),
        [
            "product_name_raw",
            "publisher_raw",
            "adopted_curriculum_source_url_url_type",
            "adopted_curriculum_source_url_page_title",
            "evidence_mentions_expected_publisher",
            "evidence_mentions_expected_product",
        ],
    ].head(10)
    if not success_sample.empty:
        lines.extend(["", "## Successful Fetch Sample", "", _markdown_table(success_sample)])

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
