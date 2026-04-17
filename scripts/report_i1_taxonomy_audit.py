from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


FOCAL_SLICE_FILTERS = {
    "Assessment": ("product_type_usage", "Assessment"),
    "catalog_unspecified": ("placeholder_mapping", "catalog_unspecified"),
}

OVERLAP_COLUMNS = [
    "assessment_slice",
    "placeholder_mapping",
    "state_specific_risk",
    "evidence_richness",
    "publisher_presence",
    "product_type_usage",
]


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


def _catalog_lookup(catalog_df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if catalog_df is None or catalog_df.empty:
        return {}
    lookup = catalog_df.copy()
    lookup["product_identifier"] = lookup["product_identifier"].map(_normalize_text)
    return lookup.set_index("product_identifier").to_dict(orient="index")


def _expected_catalog_row(row: pd.Series, catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected_id = (
        _normalize_text(row.get("expected_match_id"))
        or _normalize_text(row.get("expected_product_identifier"))
        or _normalize_text(row.get("product_identifier"))
    )
    return catalog.get(expected_id, {})


def _catalog_text(catalog_row: dict[str, Any]) -> str:
    parts = [
        catalog_row.get("product_name", ""),
        catalog_row.get("series", ""),
        catalog_row.get("product_type", ""),
        catalog_row.get("publisher", ""),
    ]
    return " ".join(_normalize_text(part).lower() for part in parts)


def _is_short_or_acronym(value: Any) -> bool:
    text = _normalize_text(value)
    if not text:
        return False
    tokens = [token for token in text.replace("/", " ").replace("-", " ").split() if token]
    if len(tokens) <= 2:
        return True
    alpha = "".join(ch for ch in text if ch.isalpha())
    return bool(alpha) and alpha.upper() == alpha and len(tokens) <= 4


def _classification(row: pd.Series, catalog_row: dict[str, Any]) -> tuple[str, list[str]]:
    top10_correct = _as_bool(row.get("top10_correct"))
    top1_correct = _as_bool(row.get("top1_correct"))
    llm_joined = "llm_top1_correct" in row and not pd.isna(row.get("llm_top1_correct"))
    llm_top1_correct = _as_bool(row.get("llm_top1_correct")) if llm_joined else False
    llm_abstained = _as_bool(row.get("llm_abstained")) if llm_joined else False

    product_type_usage = _normalize_text(row.get("product_type_usage"))
    title = _normalize_text(row.get("product_name_raw"))
    publisher = _normalize_text(row.get("publisher_raw"))
    evidence_richness = _normalize_text(row.get("evidence_richness"))
    placeholder_mapping = _normalize_text(row.get("placeholder_mapping"))
    assessment_slice = _normalize_text(row.get("assessment_slice"))
    state_specific_risk = _normalize_text(row.get("state_specific_risk"))
    llm_primary_reason = _normalize_text(row.get("llm_primary_reason"))
    catalog_blob = _catalog_text(catalog_row)

    signals: list[str] = []
    if not top10_correct:
        signals.append("gold_absent_from_top10")
    if top10_correct and not top1_correct:
        signals.append("ranker_gold_present_top1_failure")
    if llm_joined and top10_correct and not llm_top1_correct:
        signals.append("llm_gold_present_selection_failure")
    if llm_joined and top10_correct and llm_abstained:
        signals.append("llm_abstained_with_gold_present")
    if placeholder_mapping != "standard_catalog_mapping":
        signals.append(f"placeholder_mapping:{placeholder_mapping or 'missing'}")
    if "unspecified" in catalog_blob:
        signals.append("expected_catalog_text_contains_unspecified")
    if "not publicly available" in catalog_blob:
        signals.append("expected_catalog_text_not_publicly_available")
    if _as_bool(catalog_row.get("district_created")):
        signals.append("expected_catalog_district_created")
    if _as_bool(catalog_row.get("embedded_assessment")):
        signals.append("expected_catalog_embedded_assessment")
    if product_type_usage == "Assessment":
        signals.append("district_usage_assessment")
    if assessment_slice and assessment_slice != "not_assessment":
        signals.append(f"assessment_slice:{assessment_slice}")
    if _is_short_or_acronym(title):
        signals.append("short_or_acronym_title")
    if not publisher:
        signals.append("publisher_missing")
    if evidence_richness in {"sparse", "policy_placeholder_no_info"}:
        signals.append(f"evidence_richness:{evidence_richness}")
    if state_specific_risk != "standard_state":
        signals.append(f"state_specific_risk:{state_specific_risk or 'missing'}")
    if llm_primary_reason:
        signals.append(f"llm_primary_reason:{llm_primary_reason}")

    catalog_structure_signal = (
        placeholder_mapping
        in {"catalog_unspecified", "district_created", "no_information_available"}
        or "unspecified" in catalog_blob
        or "not publicly available" in catalog_blob
        or _as_bool(catalog_row.get("district_created"))
    )
    ambiguous_ground_truth_signal = (
        product_type_usage == "Assessment"
        and (
            assessment_slice
            in {
                "assessment_short_or_acronym_title",
                "assessment_publisher_missing",
                "assessment_state_specific_expected",
            }
            or _is_short_or_acronym(title)
            or not publisher
        )
    ) or evidence_richness in {"sparse", "policy_placeholder_no_info"}

    if top1_correct or (llm_joined and llm_top1_correct):
        return "already_correct_or_recovered", signals
    if top10_correct and (
        not top1_correct or (llm_joined and (not llm_top1_correct or llm_abstained))
    ):
        return "gold_present_selection_failure", signals
    if catalog_structure_signal:
        return "catalog_label_or_placeholder_structure", signals
    if ambiguous_ground_truth_signal:
        return "ambiguous_historical_ground_truth", signals
    if not top10_correct:
        return "likely_retrieval_miss", signals
    return "unclassified_review_needed", signals


def _slice_mask(df: pd.DataFrame, column: str, value: str) -> pd.Series:
    return df[column].fillna("").astype(str) == value


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.fillna("missing").astype(str).value_counts().sort_index().items()
    }


def _bucket_metrics(df: pd.DataFrame) -> dict[str, Any]:
    row_count = int(len(df))
    if row_count == 0:
        return {
            "row_count": 0,
            "bucket_counts": {},
            "top10_absent_count": 0,
            "top10_absent_rate": 0.0,
            "gold_present_selection_failure_count": 0,
            "gold_present_selection_failure_rate": 0.0,
            "llm_joined_count": 0,
        }

    top10_correct = df["top10_correct"].fillna(False).map(_as_bool)
    top1_correct = df["top1_correct"].fillna(False).map(_as_bool)
    gold_present_selection_failure = top10_correct & ~top1_correct
    if "llm_top1_correct" in df.columns:
        llm_joined = df["llm_top1_correct"].notna()
        llm_top1_correct = df["llm_top1_correct"].fillna(False).map(_as_bool)
        llm_abstained = df["llm_abstained"].fillna(False).map(_as_bool)
        gold_present_selection_failure = gold_present_selection_failure | (
            llm_joined & top10_correct & (~llm_top1_correct | llm_abstained)
        )
        llm_joined_count = int(llm_joined.sum())
    else:
        llm_joined_count = 0

    return {
        "row_count": row_count,
        "bucket_counts": _value_counts(df["i1_taxonomy_bucket"]),
        "top10_absent_count": int((~top10_correct).sum()),
        "top10_absent_rate": _safe_rate(int((~top10_correct).sum()), row_count),
        "gold_present_selection_failure_count": int(gold_present_selection_failure.sum()),
        "gold_present_selection_failure_rate": _safe_rate(
            int(gold_present_selection_failure.sum()), row_count
        ),
        "llm_joined_count": llm_joined_count,
    }


def _cross_tab(df: pd.DataFrame, row_col: str, col_col: str) -> dict[str, dict[str, int]]:
    if row_col not in df.columns or col_col not in df.columns or df.empty:
        return {}
    table = pd.crosstab(
        df[row_col].fillna("missing").astype(str),
        df[col_col].fillna("missing").astype(str),
    )
    return {
        str(index): {str(col): int(value) for col, value in values.items()}
        for index, values in table.to_dict(orient="index").items()
    }


def build_audit(
    records: pd.DataFrame,
    llm_scored: pd.DataFrame | None,
    catalog_df: pd.DataFrame | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    records = records.copy()
    if llm_scored is not None:
        llm_cols = [
            "selection_identifier",
            "llm_selected_candidate_id",
            "llm_decision",
            "llm_primary_reason",
            "llm_top1_correct",
            "llm_abstained",
        ]
        records = records.merge(
            llm_scored[llm_cols],
            on="selection_identifier",
            how="left",
        )

    catalog = _catalog_lookup(catalog_df)
    buckets: list[str] = []
    signals: list[str] = []
    expected_product_names: list[str] = []
    expected_series: list[str] = []
    expected_publishers: list[str] = []
    expected_product_types: list[str] = []
    expected_district_created: list[bool] = []
    expected_embedded_assessment: list[bool] = []

    for _, row in records.iterrows():
        catalog_row = _expected_catalog_row(row, catalog)
        bucket, row_signals = _classification(row, catalog_row)
        buckets.append(bucket)
        signals.append("|".join(row_signals))
        expected_product_names.append(_normalize_text(catalog_row.get("product_name")))
        expected_series.append(_normalize_text(catalog_row.get("series")))
        expected_publishers.append(_normalize_text(catalog_row.get("publisher")))
        expected_product_types.append(_normalize_text(catalog_row.get("product_type")))
        expected_district_created.append(_as_bool(catalog_row.get("district_created")))
        expected_embedded_assessment.append(_as_bool(catalog_row.get("embedded_assessment")))

    records["i1_taxonomy_bucket"] = buckets
    records["i1_taxonomy_signals"] = signals
    records["expected_catalog_product_name"] = expected_product_names
    records["expected_catalog_series"] = expected_series
    records["expected_catalog_publisher"] = expected_publishers
    records["expected_catalog_product_type"] = expected_product_types
    records["expected_catalog_district_created"] = expected_district_created
    records["expected_catalog_embedded_assessment"] = expected_embedded_assessment

    focal_mask = pd.Series(False, index=records.index)
    slice_summaries: dict[str, Any] = {}
    for label, (column, value) in FOCAL_SLICE_FILTERS.items():
        mask = _slice_mask(records, column, value)
        focal_mask = focal_mask | mask
        slice_df = records[mask]
        slice_summaries[label] = _bucket_metrics(slice_df)

    focal_df = records[focal_mask].copy()
    audit = {
        "records_row_count": int(len(records)),
        "focal_row_count": int(len(focal_df)),
        "llm_joined_row_count": int(records["llm_top1_correct"].notna().sum())
        if "llm_top1_correct" in records.columns
        else 0,
        "overall_focal": _bucket_metrics(focal_df),
        "slices": slice_summaries,
        "overlaps": {
            "bucket_by_assessment_slice": _cross_tab(
                focal_df, "i1_taxonomy_bucket", "assessment_slice"
            ),
            "bucket_by_placeholder_mapping": _cross_tab(
                focal_df, "i1_taxonomy_bucket", "placeholder_mapping"
            ),
            "bucket_by_state_specific_risk": _cross_tab(
                focal_df, "i1_taxonomy_bucket", "state_specific_risk"
            ),
            "bucket_by_evidence_richness": _cross_tab(
                focal_df, "i1_taxonomy_bucket", "evidence_richness"
            ),
            "bucket_by_publisher_presence": _cross_tab(
                focal_df, "i1_taxonomy_bucket", "publisher_presence"
            ),
        },
    }
    return audit, focal_df


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def _bucket_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bucket_counts = summary.get("bucket_counts", {})
    row_count = int(summary.get("row_count", 0))
    for bucket, count in sorted(bucket_counts.items()):
        rows.append(
            {
                "bucket": bucket,
                "count": count,
                "rate": _safe_rate(int(count), row_count),
            }
        )
    return rows


def _sample_rows(df: pd.DataFrame, bucket: str, limit: int = 5) -> list[dict[str, Any]]:
    sample = df[df["i1_taxonomy_bucket"] == bucket].head(limit).copy()
    columns = [
        "selection_identifier",
        "product_type_usage",
        "product_name_raw",
        "publisher_raw",
        "placeholder_mapping",
        "assessment_slice",
        "evidence_richness",
        "state_specific_risk",
        "expected_catalog_product_name",
        "expected_catalog_series",
        "expected_catalog_product_type",
        "top10_correct",
        "top1_correct",
        "llm_decision",
        "llm_primary_reason",
    ]
    present = [column for column in columns if column in sample.columns]
    for column in ["llm_decision", "llm_primary_reason"]:
        if column in sample.columns:
            sample[column] = sample[column].fillna("").replace("", "not_joined")
    return sample[present].fillna("").astype(str).to_dict(orient="records")


def build_report(audit: dict[str, Any], focal_df: pd.DataFrame) -> str:
    slice_rows = []
    for label, metrics in audit["slices"].items():
        row = {"slice": label}
        row.update(
            {
                "row_count": metrics["row_count"],
                "top10_absent_count": metrics["top10_absent_count"],
                "top10_absent_rate": metrics["top10_absent_rate"],
                "gold_present_selection_failure_bucket_count": metrics[
                    "bucket_counts"
                ].get("gold_present_selection_failure", 0),
                "llm_joined_count": metrics["llm_joined_count"],
            }
        )
        slice_rows.append(row)

    lines = [
        "# I1 Assessment and Catalog-Unspecified Taxonomy Audit",
        "",
        "- Task: `I1` diagnostic checkpoint",
        f"- records rows: `{audit['records_row_count']}`",
        f"- focal rows: `{audit['focal_row_count']}`",
        f"- LLM joined rows: `{audit['llm_joined_row_count']}`",
        "- behavior changed: `no`",
        "",
        "## Bucket Definitions",
        "",
        "- `likely_retrieval_miss`: gold is absent from top-10 and no stronger catalog-structure or ambiguity signal is present.",
        "- `catalog_label_or_placeholder_structure`: expected catalog row or derived placeholder label points to `Unspecified`, no-information, district-created, or similar catalog structure.",
        "- `ambiguous_historical_ground_truth`: row evidence is too sparse, acronym-like, assessment-subtype-like, or publisher-missing to treat the historical gold label as a clean retrieval target without review.",
        "- `gold_present_selection_failure`: gold is present in top-10, but the ranker or joined canonical `F2` LLM output did not select it.",
        "- `already_correct_or_recovered`: accepted ranker or joined LLM output already selected the expected gold id.",
        "",
        "## Slice Summary",
        "",
        _markdown_table(
            slice_rows,
            [
                "slice",
                "row_count",
                "top10_absent_count",
                "top10_absent_rate",
                "gold_present_selection_failure_bucket_count",
                "llm_joined_count",
            ],
        ),
        "",
        "## Overall Focal Bucket Counts",
        "",
        _markdown_table(_bucket_rows(audit["overall_focal"]), ["bucket", "count", "rate"]),
        "",
    ]

    for label, metrics in audit["slices"].items():
        lines.extend(
            [
                f"## `{label}` Bucket Counts",
                "",
                _markdown_table(_bucket_rows(metrics), ["bucket", "count", "rate"]),
                "",
            ]
        )

    lines.extend(
        [
            "## Key Overlaps",
            "",
            "### Bucket by Placeholder Mapping",
            "",
            "```json",
            json.dumps(audit["overlaps"]["bucket_by_placeholder_mapping"], indent=2),
            "```",
            "",
            "### Bucket by Assessment Slice",
            "",
            "```json",
            json.dumps(audit["overlaps"]["bucket_by_assessment_slice"], indent=2),
            "```",
            "",
            "## Example Rows",
            "",
        ]
    )

    for bucket in [
        "likely_retrieval_miss",
        "catalog_label_or_placeholder_structure",
        "ambiguous_historical_ground_truth",
        "gold_present_selection_failure",
    ]:
        samples = _sample_rows(focal_df, bucket)
        lines.extend([f"### `{bucket}`", ""])
        if not samples:
            lines.append("- none")
        else:
            for sample in samples:
                lines.append(
                    "- `{selection_identifier}`: `{product_type_usage}` `{product_name_raw}` / expected `{expected_catalog_product_name}`; labels `{placeholder_mapping}`, `{assessment_slice}`, `{evidence_richness}`; top10 `{top10_correct}`, top1 `{top1_correct}`; LLM `{llm_decision}` `{llm_primary_reason}`".format(
                        **sample
                    )
                )
        lines.append("")

    overall = audit["overall_focal"]
    bucket_counts = overall["bucket_counts"]
    assessment = audit["slices"]["Assessment"]
    unspecified = audit["slices"]["catalog_unspecified"]
    lines.extend(
        [
            "## Interpretation",
            "",
            f"- The focal audit covers `{overall['row_count']}` rows across `Assessment` and `catalog_unspecified` slices.",
            f"- Top-10 absence is still substantial: `{overall['top10_absent_count']}` focal rows have gold absent from the accepted top-10 shortlist.",
            f"- `Assessment` is mostly an ambiguity / historical-ground-truth problem in this taxonomy: `{assessment['bucket_counts'].get('ambiguous_historical_ground_truth', 0)} / {assessment['row_count']}` assessment rows land in `ambiguous_historical_ground_truth`, while only `{assessment['bucket_counts'].get('likely_retrieval_miss', 0)} / {assessment['row_count']}` land in the clean `likely_retrieval_miss` bucket.",
            f"- `catalog_unspecified` is mostly a catalog-label / placeholder-structure problem: `{unspecified['bucket_counts'].get('catalog_label_or_placeholder_structure', 0)} / {unspecified['row_count']}` rows land in `catalog_label_or_placeholder_structure`.",
            f"- Gold-present primary selection failures account for `{bucket_counts.get('gold_present_selection_failure', 0)}` focal rows: `{assessment['bucket_counts'].get('gold_present_selection_failure', 0)}` assessment rows and `{unspecified['bucket_counts'].get('gold_present_selection_failure', 0)}` `catalog_unspecified` rows.",
            "",
            "## Decision",
            "",
            "- accept `I1` as a diagnostic checkpoint",
            "- no behavioral baseline change",
            "- use this artifact to decide whether `I2` should inspect catalog-label consistency before any retrieval change",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Assessment and catalog_unspecified rows into I1 taxonomy buckets."
    )
    parser.add_argument("--records-csv", required=True, help="Accepted evaluation records CSV.")
    parser.add_argument(
        "--llm-scored-csv",
        default=None,
        help="Optional canonical LLM scored CSV joined by selection_identifier.",
    )
    parser.add_argument(
        "--catalog-file",
        default=None,
        help="Optional catalog CSV for expected catalog metadata.",
    )
    parser.add_argument("--output-json", required=True, help="Audit JSON output path.")
    parser.add_argument("--output-md", required=True, help="Audit Markdown output path.")
    parser.add_argument("--output-csv", required=True, help="Row-level focal audit CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = pd.read_csv(args.records_csv)
    llm_scored = pd.read_csv(args.llm_scored_csv) if args.llm_scored_csv else None
    catalog_df = (
        pd.read_csv(args.catalog_file, encoding="latin-1") if args.catalog_file else None
    )
    audit, focal_df = build_audit(records, llm_scored, catalog_df)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_report(audit, focal_df), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    focal_df.to_csv(output_csv, index=False)

    print(output_md)


if __name__ == "__main__":
    main()
