from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_id(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _blankish(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().eq("")


def _join_catalog_metadata(records: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    lookup = catalog[
        [
            "product_identifier",
            "product_name",
            "series",
            "publisher",
            "publisher_prior",
            "intended_grades",
            "copyright_year",
        ]
    ].copy()
    lookup["product_identifier"] = _normalize_id(lookup["product_identifier"])

    merged = records.copy()
    for column in [
        "expected_match_id",
        "predicted_match_id",
        "pred_rank_1_id",
        "pred_rank_2_id",
        "pred_rank_3_id",
    ]:
        if column in merged.columns:
            merged[column] = _normalize_id(merged[column])

    expected_lookup = lookup.add_prefix("expected_").rename(
        columns={"expected_product_identifier": "expected_match_id"}
    )
    predicted_lookup = lookup.add_prefix("predicted_").rename(
        columns={"predicted_product_identifier": "predicted_match_id"}
    )

    merged = merged.merge(expected_lookup, on="expected_match_id", how="left")
    merged = merged.merge(predicted_lookup, on="predicted_match_id", how="left")
    return merged


def _format_ratio(value: float) -> str:
    return f"{value:.1%}"


def _value_counts_lines(series: pd.Series, *, topn: int = 5) -> list[str]:
    counts = series.fillna("[missing]").replace("", "[missing]").value_counts().head(topn)
    return [f"- `{label}`: {count}" for label, count in counts.items()]


def _series_product_lines(df: pd.DataFrame, *, topn: int = 5) -> list[str]:
    counts = (
        df[["expected_series", "expected_product_name"]]
        .fillna("[missing]")
        .value_counts()
        .head(topn)
    )
    return [
        f"- `{series}` / `{product}`: {count}"
        for (series, product), count in counts.items()
    ]


def _sample_case_lines(df: pd.DataFrame, *, topn: int = 5) -> list[str]:
    sample = df.head(topn)
    lines: list[str] = []
    for _, row in sample.iterrows():
        raw_title = _display_value(row.get("product_name_raw", ""))
        publisher = _display_value(row.get("publisher_raw", ""))
        grade = _display_value(row.get("grade", ""))
        expected_name = _display_value(row.get("expected_product_name", ""))
        predicted_name = _display_value(row.get("predicted_product_name", ""))
        expected_rank = _display_value(row.get("expected_rank", ""))
        lines.append(
            "- "
            f"`{raw_title}` | publisher=`{publisher}` | grade=`{grade}` | "
            f"expected=`{expected_name}` | "
            f"predicted=`{predicted_name}` | expected_rank=`{expected_rank}`"
        )
    return lines


def _display_value(value: object) -> str:
    if pd.isna(value):
        return "[blank]"
    text = str(value).strip()
    return text if text else "[blank]"


def build_report(records: pd.DataFrame, summary: dict) -> str:
    total = len(records)
    no_prediction = records["predicted_match_id"].eq("")
    top1_wrong_top3_right = (~records["top1_correct"]) & (records["top3_correct"])
    top3_miss = ~records["top3_correct"]
    blank_publisher = _blankish(records["publisher_raw"])

    no_pred_df = records.loc[no_prediction].copy()
    near_miss_df = records.loc[top1_wrong_top3_right].copy()
    miss_df = records.loc[top3_miss].copy()

    near_miss_df["same_series_as_top1"] = (
        near_miss_df["expected_series"].fillna("")
        == near_miss_df["predicted_series"].fillna("")
    )

    lines: list[str] = []
    lines.append("# Benchmark Error Analysis")
    lines.append("")
    lines.append(f"- benchmark: `{summary['benchmark_file']}`")
    lines.append(f"- profile: `{summary['profile']}`")
    lines.append(f"- record count: {total}")
    lines.append(f"- top-1 accuracy: {_format_ratio(summary['top1_accuracy'])}")
    lines.append(f"- top-3 recall: {_format_ratio(summary['top3_recall'])}")
    lines.append(f"- prediction rate: {_format_ratio(summary['prediction_rate'])}")
    lines.append("")
    lines.append("## Failure Buckets")
    lines.append("")
    lines.append(f"- no prediction: {int(no_prediction.sum())} / {total}")
    lines.append(f"- top-1 wrong but top-3 right: {int(top1_wrong_top3_right.sum())} / {total}")
    lines.append(f"- top-3 miss: {int(top3_miss.sum())} / {total}")
    lines.append("")
    lines.append("## Key Patterns")
    lines.append("")
    lines.append(
        "- Blank publishers underperform the rest of the set:"
        f" top-1 accuracy {_format_ratio(records.loc[blank_publisher, 'top1_correct'].mean())}"
        f" vs {_format_ratio(records.loc[~blank_publisher, 'top1_correct'].mean())},"
        f" top-3 recall {_format_ratio(records.loc[blank_publisher, 'top3_correct'].mean())}"
        f" vs {_format_ratio(records.loc[~blank_publisher, 'top3_correct'].mean())}."
    )
    lines.append(
        "- No-prediction rows are concentrated in a few catalog families:"
    )
    lines.extend(_series_product_lines(no_pred_df, topn=8))
    lines.append(
        "- Wrong top-1 / right top-3 cases are almost entirely same-family confusions:"
        f" {int(near_miss_df['same_series_as_top1'].sum())} of {len(near_miss_df)}"
        " have the expected and predicted top-1 rows in the same catalog series."
    )
    lines.append("- Top-3 misses are also concentrated in a few families:")
    lines.extend(_series_product_lines(miss_df, topn=8))
    lines.append("")
    lines.append("## Publisher Signal")
    lines.append("")
    lines.append("- No-prediction publisher distribution:")
    lines.extend(_value_counts_lines(no_pred_df["publisher_raw"], topn=8))
    lines.append("- Top-3 miss publisher distribution:")
    lines.extend(_value_counts_lines(miss_df["publisher_raw"], topn=8))
    lines.append("")
    lines.append("## Representative Cases")
    lines.append("")
    lines.append("- No prediction examples:")
    lines.extend(_sample_case_lines(no_pred_df, topn=6))
    lines.append("- Wrong top-1 but top-3 right examples:")
    lines.extend(_sample_case_lines(near_miss_df, topn=6))
    lines.append("- Top-3 miss examples:")
    lines.extend(_sample_case_lines(miss_df, topn=6))
    lines.append("")
    lines.append("## Takeaways")
    lines.append("")
    lines.append(
        "- The current matcher is losing a large share of recoverable rows before final ranking,"
        " especially where title normalization is weak or the publisher is blank."
    )
    lines.append(
        "- The first ranking problem is not random confusion across unrelated products;"
        " it is mostly within-program grade, edition, or series-sibling confusion."
    )
    lines.append(
        "- The best next matcher changes are likely:"
        " stronger title normalization and aliasing, softer handling for missing publishers,"
        " and family-aware grade tie-breaking inside the same catalog series."
    )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a markdown error-analysis report from saved benchmark outputs."
    )
    parser.add_argument("--records-csv", required=True, help="Per-record evaluation CSV.")
    parser.add_argument("--summary-json", required=True, help="Evaluation summary JSON.")
    parser.add_argument(
        "--catalog-csv",
        help="Optional catalog CSV. Defaults to the catalog_file stored in the summary JSON.",
    )
    parser.add_argument("--output-md", help="Optional markdown output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_path = Path(args.summary_json)
    summary = _read_summary(summary_path)

    records = pd.read_csv(args.records_csv)
    catalog_path = Path(args.catalog_csv or summary["catalog_file"])
    catalog = pd.read_csv(catalog_path, encoding="latin-1")

    merged = _join_catalog_metadata(records, catalog)
    report = build_report(merged, summary)

    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
