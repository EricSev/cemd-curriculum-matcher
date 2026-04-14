from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _strategy_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("match_selected_strategy", dropna=False).agg(
        row_count=("selection_identifier", "count"),
        top1_accuracy=("top1_correct", "mean"),
        top3_recall=("top3_correct", "mean"),
        mean_top1_score=("top1_score", "mean"),
    )
    return grouped.round(4).sort_values("row_count", ascending=False)


def _confidence_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["match_selected_strategy", "confidence_band"], dropna=False)
        .agg(
            row_count=("selection_identifier", "count"),
            top1_accuracy=("top1_correct", "mean"),
            top3_recall=("top3_correct", "mean"),
        )
        .round(4)
    )
    return grouped


def _df_to_markdown(df: pd.DataFrame, *, include_index: bool = True) -> str:
    working = df.copy()
    if include_index:
        working = working.reset_index()
    columns = [str(column) for column in working.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in working.iterrows():
        values = [str(row[column]) for column in working.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _variant_columns(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["match_variant_scores"].fillna("{}").apply(json.loads)
    primary_top1_id = parsed.apply(lambda d: _normalize_id(d.get("primary", {}).get("top1_id", "")))
    primary_prediction_count = parsed.apply(lambda d: int(d.get("primary", {}).get("prediction_count", 0)))
    return pd.DataFrame(
        {
            "primary_top1_id": primary_top1_id,
            "primary_prediction_count": primary_prediction_count,
        }
    )


def build_report(df: pd.DataFrame) -> str:
    variants = _variant_columns(df)
    enriched = pd.concat([df.reset_index(drop=True), variants], axis=1)
    enriched["expected_match_id_norm"] = enriched["expected_match_id"].map(_normalize_id)
    enriched["predicted_match_id_norm"] = enriched["predicted_match_id"].map(_normalize_id)

    fallback_rows = enriched[enriched["match_used_fallback"].fillna(False)].copy()
    repair_rescued_no_prediction = (
        fallback_rows["primary_prediction_count"].eq(0)
        & fallback_rows["predicted_match_id_norm"].ne("")
    )
    repair_improved_top1 = (
        fallback_rows["primary_top1_id"] != fallback_rows["expected_match_id_norm"]
    ) & (fallback_rows["predicted_match_id_norm"] == fallback_rows["expected_match_id_norm"])
    repair_hurt_top1 = (
        fallback_rows["primary_top1_id"] == fallback_rows["expected_match_id_norm"]
    ) & (fallback_rows["predicted_match_id_norm"] != fallback_rows["expected_match_id_norm"])
    repair_unchanged_top1 = (
        fallback_rows["primary_top1_id"] == fallback_rows["predicted_match_id_norm"]
    )

    strategy_table = _strategy_table(enriched)
    confidence_table = _confidence_table(enriched)

    lines = [
        "# Repair Strategy Report",
        "",
        f"- row count: {len(enriched)}",
        f"- repair attempted rate: {enriched['repair_attempted'].mean():.4f}",
        f"- fallback usage rate: {enriched['match_used_fallback'].mean():.4f}",
        f"- repaired rows: {len(fallback_rows)}",
        "",
        "## Repair Outcome Summary",
        "",
        f"- repair rescued no-prediction rows: {int(repair_rescued_no_prediction.sum())}",
        f"- repair improved top-1 correctness: {int(repair_improved_top1.sum())}",
        f"- repair hurt top-1 correctness: {int(repair_hurt_top1.sum())}",
        f"- repair left top-1 unchanged: {int(repair_unchanged_top1.sum())}",
        "",
        "## By Selected Strategy",
        "",
        _df_to_markdown(strategy_table),
        "",
        "## By Strategy And Confidence Band",
        "",
        _df_to_markdown(confidence_table),
    ]

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Keep repair selection score-based in general, but retain a heuristic gate on `title_plus_publisher` when it tries to replace an existing primary match.",
            f"- In this benchmark slice, fallback repairs still rescued {int(repair_rescued_no_prediction.sum())} no-prediction rows while reducing repair hurt rows to {int(repair_hurt_top1.sum())}.",
        ]
    )
    publisher_as_title_rows = enriched[
        enriched["match_selected_strategy"] == "publisher_as_title"
    ]
    if not publisher_as_title_rows.empty:
        publisher_as_title_accuracy = float(
            publisher_as_title_rows["top1_correct"].mean()
        )
        lines.append(
            f"- `publisher_as_title` remains a low-trust strategy in this slice with top-1 accuracy {publisher_as_title_accuracy:.4f}; keep it out of any looser replacement policy."
        )

    if not fallback_rows.empty:
        helped_examples = fallback_rows.loc[repair_improved_top1, [
            "product_name_raw",
            "publisher_raw",
            "grade",
            "expected_match_id_norm",
            "primary_top1_id",
            "predicted_match_id_norm",
            "match_selected_strategy",
        ]].head(10)
        hurt_examples = fallback_rows.loc[repair_hurt_top1, [
            "product_name_raw",
            "publisher_raw",
            "grade",
            "expected_match_id_norm",
            "primary_top1_id",
            "predicted_match_id_norm",
            "match_selected_strategy",
        ]].head(10)
        lines.extend(["", "## Example Rows"])
        if not helped_examples.empty:
            lines.extend(["", "### Helped"])
            lines.append(_df_to_markdown(helped_examples, include_index=False))
        if not hurt_examples.empty:
            lines.extend(["", "### Hurt"])
            lines.append(_df_to_markdown(hurt_examples, include_index=False))

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize repair strategy outcomes from evaluation records."
    )
    parser.add_argument("--records-csv", required=True, help="Evaluation records CSV.")
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
