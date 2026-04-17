from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _normalize_text(value).lower() in {"true", "1", "yes"}


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _catalog_lookup(catalog_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    catalog = catalog_df.copy()
    catalog["product_identifier"] = catalog["product_identifier"].map(_normalize_text)
    return catalog.set_index("product_identifier").to_dict(orient="index")


def _expected_catalog(row: pd.Series, catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected_id = (
        _normalize_text(row.get("expected_match_id"))
        or _normalize_text(row.get("expected_product_identifier"))
        or _normalize_text(row.get("product_identifier"))
    )
    return catalog.get(expected_id, {})


def _text_blob(row: dict[str, Any]) -> str:
    return " ".join(
        _normalize_text(row.get(column)).lower()
        for column in ("product_name", "series", "product_type", "publisher")
    )


def _label_family(row: pd.Series, expected: dict[str, Any]) -> str:
    placeholder = _normalize_text(row.get("placeholder_mapping"))
    product_type_usage = _normalize_text(row.get("product_type_usage"))
    title = _normalize_text(row.get("product_name_raw")).lower()
    blob = _text_blob(expected)

    if placeholder == "catalog_unspecified":
        return "catalog_unspecified"
    if placeholder == "no_information_available" or title in {"not available", "n/a", "na"}:
        return "no_information_available"
    if _as_bool(expected.get("district_created")):
        return "district_created"
    if product_type_usage == "Assessment" or _as_bool(expected.get("embedded_assessment")):
        return "assessment_like"
    if "unspecified" in blob:
        return "catalog_unspecified_catalog_text_only"
    return "standard"


def _consistency_issue(row: pd.Series, expected: dict[str, Any], family: str) -> str:
    placeholder = _normalize_text(row.get("placeholder_mapping"))
    product_type_usage = _normalize_text(row.get("product_type_usage"))
    expected_product_type = _normalize_text(expected.get("product_type"))
    expected_product = _normalize_text(expected.get("product_name"))
    expected_series = _normalize_text(expected.get("series"))
    top10_correct = _as_bool(row.get("top10_correct"))
    top1_correct = _as_bool(row.get("top1_correct"))

    if family == "catalog_unspecified":
        if top1_correct or top10_correct:
            return "unspecified_label_but_retrievable"
        if expected_product_type in {"Core Curriculum", "Supplemental"}:
            return "unspecified_structural_placeholder"
        return "unspecified_review_needed"
    if family == "no_information_available":
        return "policy_no_information_placeholder"
    if family == "district_created":
        return "district_created_placeholder"
    if family == "assessment_like":
        if product_type_usage == "Assessment" and expected_product_type != "Assessment":
            if expected_product in {"State Created Assessments", "Other Assessment - Not in Catalog"}:
                return "assessment_bucket_gold_label"
            return "assessment_usage_catalog_type_mismatch"
        if top10_correct and not top1_correct:
            return "assessment_gold_present_selection_issue"
        return "assessment_like_consistent"
    if placeholder == "standard_catalog_mapping" and "unspecified" in (
        expected_product + " " + expected_series
    ).lower():
        return "missing_unspecified_derived_label"
    return "standard_or_not_focal"


def build_audit(records: pd.DataFrame, catalog_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    catalog = _catalog_lookup(catalog_df)
    rows = []
    for _, record in records.iterrows():
        expected = _expected_catalog(record, catalog)
        family = _label_family(record, expected)
        issue = _consistency_issue(record, expected, family)
        if family == "standard" and issue == "standard_or_not_focal":
            continue
        row = record.to_dict()
        row.update(
            {
                "i2_label_family": family,
                "i2_consistency_issue": issue,
                "expected_catalog_product_name": _normalize_text(expected.get("product_name")),
                "expected_catalog_series": _normalize_text(expected.get("series")),
                "expected_catalog_product_type": _normalize_text(expected.get("product_type")),
                "expected_catalog_publisher": _normalize_text(expected.get("publisher")),
                "expected_catalog_district_created": _as_bool(expected.get("district_created")),
                "expected_catalog_embedded_assessment": _as_bool(expected.get("embedded_assessment")),
                "expected_catalog_state_specific_version": _as_bool(
                    expected.get("state_specific_version")
                ),
            }
        )
        rows.append(row)

    focal = pd.DataFrame(rows)
    family_counts = (
        focal["i2_label_family"].value_counts().sort_index().to_dict()
        if not focal.empty
        else {}
    )
    issue_counts = (
        focal["i2_consistency_issue"].value_counts().sort_index().to_dict()
        if not focal.empty
        else {}
    )
    by_family_issue = (
        pd.crosstab(focal["i2_label_family"], focal["i2_consistency_issue"])
        .astype(int)
        .to_dict(orient="index")
        if not focal.empty
        else {}
    )
    audit = {
        "records_row_count": int(len(records)),
        "focal_row_count": int(len(focal)),
        "family_counts": {str(k): int(v) for k, v in family_counts.items()},
        "issue_counts": {str(k): int(v) for k, v in issue_counts.items()},
        "by_family_issue": {
            str(family): {str(issue): int(count) for issue, count in values.items()}
            for family, values in by_family_issue.items()
        },
    }
    return audit, focal


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def _count_rows(counts: dict[str, int], total: int, label_col: str) -> list[dict[str, Any]]:
    return [
        {label_col: key, "count": value, "rate": _safe_rate(value, total)}
        for key, value in sorted(counts.items())
    ]


def _sample_rows(df: pd.DataFrame, issue: str, limit: int = 6) -> list[dict[str, Any]]:
    sample = df[df["i2_consistency_issue"] == issue].head(limit)
    columns = [
        "selection_identifier",
        "product_type_usage",
        "product_name_raw",
        "publisher_raw",
        "placeholder_mapping",
        "assessment_slice",
        "i2_label_family",
        "i2_consistency_issue",
        "expected_catalog_product_name",
        "expected_catalog_series",
        "expected_catalog_product_type",
        "top10_correct",
        "top1_correct",
    ]
    present = [column for column in columns if column in sample.columns]
    return sample[present].fillna("").astype(str).to_dict(orient="records")


def build_report(audit: dict[str, Any], focal: pd.DataFrame) -> str:
    total = audit["focal_row_count"]
    lines = [
        "# I2 Catalog-Label Consistency Audit",
        "",
        "- Task: `I2` diagnostic checkpoint",
        f"- records rows: `{audit['records_row_count']}`",
        f"- focal rows: `{audit['focal_row_count']}`",
        "- behavior changed: `no`",
        "",
        "## Label Families",
        "",
        _markdown_table(
            _count_rows(audit["family_counts"], total, "family"),
            ["family", "count", "rate"],
        ),
        "",
        "## Consistency Issues",
        "",
        _markdown_table(
            _count_rows(audit["issue_counts"], total, "issue"),
            ["issue", "count", "rate"],
        ),
        "",
        "## Family by Issue",
        "",
        "```json",
        json.dumps(audit["by_family_issue"], indent=2),
        "```",
        "",
        "## Example Rows",
        "",
    ]
    for issue in [
        "unspecified_structural_placeholder",
        "unspecified_label_but_retrievable",
        "assessment_bucket_gold_label",
        "assessment_usage_catalog_type_mismatch",
        "policy_no_information_placeholder",
    ]:
        lines.extend([f"### `{issue}`", ""])
        samples = _sample_rows(focal, issue)
        if not samples:
            lines.append("- none")
        else:
            for sample in samples:
                lines.append(
                    "- `{selection_identifier}`: `{product_type_usage}` `{product_name_raw}` -> expected `{expected_catalog_product_name}` / `{expected_catalog_series}` (`{expected_catalog_product_type}`); labels `{placeholder_mapping}`, `{assessment_slice}`; top10 `{top10_correct}`, top1 `{top1_correct}`".format(
                        **sample
                    )
                )
        lines.append("")

    issue_counts = audit["issue_counts"]
    lines.extend(
        [
            "## Interpretation",
            "",
            f"- `catalog_unspecified` is not a normal retrieval family: `{issue_counts.get('unspecified_structural_placeholder', 0)}` rows are unresolved `Unspecified` catalog placeholders, while `{issue_counts.get('unspecified_label_but_retrievable', 0)}` are already retrievable despite the label.",
            f"- Assessment ambiguity is concentrated in catalog bucket labels: `{issue_counts.get('assessment_bucket_gold_label', 0)}` rows point to broad gold labels such as `State Created Assessments` or `Other Assessment - Not in Catalog`.",
            f"- Assessment usage/catalog-type mismatch accounts for `{issue_counts.get('assessment_usage_catalog_type_mismatch', 0)}` rows.",
            "",
            "## Decision",
            "",
            "- accept `I2` as a diagnostic checkpoint",
            "- no behavioral baseline change",
            "- use this audit to drive one narrow derived-label normalization in `I3`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit catalog-label consistency for I2."
    )
    parser.add_argument("--records-csv", required=True, help="Accepted evaluation records CSV.")
    parser.add_argument("--catalog-file", required=True, help="Catalog CSV file.")
    parser.add_argument("--output-json", required=True, help="Audit JSON output path.")
    parser.add_argument("--output-md", required=True, help="Audit Markdown output path.")
    parser.add_argument("--output-csv", required=True, help="Row-level audit CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = pd.read_csv(args.records_csv)
    catalog_df = pd.read_csv(args.catalog_file, encoding="latin-1")
    audit, focal = build_audit(records, catalog_df)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    focal.to_csv(output_csv, index=False)

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_report(audit, focal), encoding="utf-8")
    print(output_md)


if __name__ == "__main__":
    main()
