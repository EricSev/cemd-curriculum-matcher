from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _safe_mean(series: pd.Series) -> float:
    return 0.0 if series.empty else float(series.mean())


def _markdown_table(df: pd.DataFrame, *, include_index: bool = True) -> str:
    working = df.copy()
    if include_index:
        working = working.reset_index()
    cols = [str(c) for c in working.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in working.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in working.columns) + " |")
    return "\n".join(lines)


def _precision_recall(df: pd.DataFrame, predicted_col: str, actual_col: str) -> tuple[float, float]:
    predicted = df[predicted_col].fillna(False)
    actual = df[actual_col].fillna(False)
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(group_col, dropna=False):
        precision, recall = _precision_recall(group, "human_match_challenge_flag", "qa_expected_incorrect")
        rows.append(
            {
                group_col: value,
                "row_count": int(len(group)),
                "incorrect_rate": round(_safe_mean(group["qa_expected_incorrect"].fillna(False)), 4),
                "challenge_rate": round(_safe_mean(group["human_match_challenge_flag"].fillna(False)), 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("row_count", ascending=False)


def _threshold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    thresholds = [0.25, 0.35, 0.45, 0.55, 0.65]
    rows = []
    for threshold in thresholds:
        predicted = df["human_match_support_score"].fillna(0.0) < threshold
        temp = df.copy()
        temp["threshold_flag"] = predicted
        precision, recall = _precision_recall(temp, "threshold_flag", "qa_expected_incorrect")
        rows.append(
            {
                "support_score_lt": threshold,
                "challenge_rate": round(_safe_mean(predicted), 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            }
        )
    return pd.DataFrame(rows)


def _reason_metrics(df: pd.DataFrame) -> pd.DataFrame:
    exploded = []
    challenged = df[df["human_match_challenge_flag"].fillna(False)].copy()
    for _, row in challenged.iterrows():
        reasons = str(row.get("human_match_challenge_reasons", "")).split("|")
        for reason in reasons:
            reason = reason.strip()
            if not reason:
                continue
            exploded.append(
                {
                    "reason": reason,
                    "qa_expected_incorrect": bool(row.get("qa_expected_incorrect", False)),
                }
            )
    if not exploded:
        return pd.DataFrame(columns=["reason", "row_count", "incorrect_rate"])
    reason_df = pd.DataFrame(exploded)
    grouped = (
        reason_df.groupby("reason", dropna=False)
        .agg(
            row_count=("reason", "count"),
            incorrect_rate=("qa_expected_incorrect", "mean"),
        )
        .round(4)
        .sort_values(["incorrect_rate", "row_count"], ascending=[False, False])
    )
    return grouped.reset_index()


def _recommendation_text(df: pd.DataFrame) -> list[str]:
    priority_metrics = _group_metrics(df, "human_match_review_priority")
    lines = []
    lookup = {
        row["human_match_review_priority"]: row for _, row in priority_metrics.iterrows()
    }
    if "high" in lookup:
        lines.append(
            f"- `high` priority currently yields precision {lookup['high']['precision']:.4f} on {lookup['high']['row_count']} rows; use this bucket as the default must-review queue."
        )
    if "medium" in lookup:
        lines.append(
            f"- `medium` priority currently yields precision {lookup['medium']['precision']:.4f}; treat this bucket as overflow review or sampled audit rather than the default queue."
        )
    if "low" in lookup:
        lines.append(
            f"- `low` priority currently yields precision {lookup['low']['precision']:.4f}; treat this bucket as pass-through with periodic calibration sampling only."
        )
    threshold_metrics = _threshold_metrics(df)
    best_balanced = threshold_metrics.iloc[(threshold_metrics["precision"] + threshold_metrics["recall"]).idxmax()]
    lines.append(
        f"- Simple support-score thresholds remain useful as a reference, but the current operating point should stay compound: strong AI disagreement first, then low-support rows only when title evidence is weak."
    )
    return lines


def build_report(df: pd.DataFrame) -> str:
    overall_precision, overall_recall = _precision_recall(
        df, "human_match_challenge_flag", "qa_expected_incorrect"
    )
    priority_metrics = _group_metrics(df, "human_match_review_priority")
    slice_metrics = _group_metrics(df, "qa_benchmark_slice")
    case_type_metrics = _group_metrics(df, "qa_case_type")
    strength_metrics = _group_metrics(df, "qa_label_strength")
    threshold_metrics = _threshold_metrics(df)
    reason_metrics = _reason_metrics(df).head(12)

    lines = [
        "# Human QA Tuning Report",
        "",
        f"- row count: {len(df)}",
        f"- overall challenge precision: {overall_precision:.4f}",
        f"- overall challenge recall: {overall_recall:.4f}",
        f"- overall challenge rate: {_safe_mean(df['human_match_challenge_flag'].fillna(False)):.4f}",
        "",
        "## By Review Priority",
        "",
        _markdown_table(priority_metrics, include_index=False),
        "",
        "## By Benchmark Slice",
        "",
        _markdown_table(slice_metrics, include_index=False),
        "",
        "## By Case Type",
        "",
        _markdown_table(case_type_metrics, include_index=False),
        "",
        "## By Label Strength",
        "",
        _markdown_table(strength_metrics, include_index=False),
        "",
        "## Support Score Threshold Sweep",
        "",
        _markdown_table(threshold_metrics, include_index=False),
        "",
        "## Most Informative Challenge Reasons",
        "",
        _markdown_table(reason_metrics, include_index=False) if not reason_metrics.empty else "_none_",
        "",
        "## Reviewer Workflow Recommendation",
        "",
    ]
    lines.extend(_recommendation_text(df))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a QA tuning report from human QA evaluation records."
    )
    parser.add_argument("--records-csv", required=True, help="Human QA evaluation records CSV.")
    parser.add_argument("--output-md", required=True, help="Markdown report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.records_csv)
    report = build_report(df)
    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
