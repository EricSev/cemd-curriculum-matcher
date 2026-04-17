from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalize_text(value).lower()))


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _normalized_placeholder(row: pd.Series) -> str:
    current = _normalize_text(row.get("placeholder_mapping"))
    if current != "catalog_unspecified":
        return current

    title_tokens = _tokens(row.get("product_name_raw"))
    series_tokens = _tokens(row.get("expected_catalog_series"))
    product_tokens = _tokens(row.get("expected_catalog_product_name"))
    product_tokens.discard("unspecified")

    if series_tokens and title_tokens and len(title_tokens & series_tokens) >= 1:
        return "catalog_unspecified_named_series"
    if product_tokens and title_tokens and len(title_tokens & product_tokens) >= 1:
        return "catalog_unspecified_named_series"
    return "catalog_unspecified_structural_placeholder"


def _metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"count": 0, "top1_accuracy": 0.0, "top10_recall": 0.0}
    return {
        "count": int(len(df)),
        "top1_accuracy": round(float(df["top1_correct"].mean()), 4),
        "top10_recall": round(float(df["top10_correct"].mean()), 4),
    }


def build_audit(rows: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    candidate = rows.copy()
    candidate["i3_placeholder_mapping"] = candidate.apply(_normalized_placeholder, axis=1)
    changed = candidate[
        candidate["placeholder_mapping"].fillna("").astype(str)
        != candidate["i3_placeholder_mapping"].fillna("").astype(str)
    ].copy()

    summary = {
        "row_count": int(len(candidate)),
        "changed_row_count": int(len(changed)),
        "baseline_placeholder_counts": {
            str(k): int(v)
            for k, v in candidate["placeholder_mapping"]
            .fillna("missing")
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "candidate_placeholder_counts": {
            str(k): int(v)
            for k, v in candidate["i3_placeholder_mapping"]
            .fillna("missing")
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "candidate_slice_metrics": {
            str(label): _metrics(group)
            for label, group in candidate.groupby("i3_placeholder_mapping", dropna=False)
        },
    }
    return summary, candidate


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


def _metric_rows(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label, values in sorted(metrics.items()):
        row = {"candidate_label": label}
        row.update(values)
        rows.append(row)
    return rows


def build_report(summary: dict[str, Any], candidate: pd.DataFrame) -> str:
    total = summary["row_count"]
    sample_cols = [
        "selection_identifier",
        "product_type_usage",
        "product_name_raw",
        "expected_catalog_product_name",
        "expected_catalog_series",
        "placeholder_mapping",
        "i3_placeholder_mapping",
        "top10_correct",
        "top1_correct",
    ]
    samples = candidate[
        candidate["placeholder_mapping"].astype(str)
        != candidate["i3_placeholder_mapping"].astype(str)
    ][sample_cols].head(10)
    lines = [
        "# I3 Narrow Label-Normalization Candidate Review",
        "",
        "- Task: `I3` narrow label-normalization candidate",
        "- Single variable: split `catalog_unspecified` into named-series versus structural-placeholder labels",
        f"- row-level source rows: `{total}`",
        f"- changed labels: `{summary['changed_row_count']}`",
        "- behavior changed: `no candidate generation, scoring, shortlist, model, or prompt wording changed`",
        "",
        "## Baseline Placeholder Counts",
        "",
        _markdown_table(
            _count_rows(summary["baseline_placeholder_counts"], total, "baseline_label"),
            ["baseline_label", "count", "rate"],
        ),
        "",
        "## Candidate Placeholder Counts",
        "",
        _markdown_table(
            _count_rows(summary["candidate_placeholder_counts"], total, "candidate_label"),
            ["candidate_label", "count", "rate"],
        ),
        "",
        "## Candidate Slice Metrics",
        "",
        _markdown_table(
            _metric_rows(summary["candidate_slice_metrics"]),
            ["candidate_label", "count", "top1_accuracy", "top10_recall"],
        ),
        "",
        "## Example Changed Rows",
        "",
    ]
    for sample in samples.fillna("").astype(str).to_dict(orient="records"):
        lines.append(
            "- `{selection_identifier}`: `{product_type_usage}` `{product_name_raw}` -> `{expected_catalog_product_name}` / `{expected_catalog_series}`; `{placeholder_mapping}` -> `{i3_placeholder_mapping}`; top10 `{top10_correct}`, top1 `{top1_correct}`".format(
                **sample
            )
        )
    if samples.empty:
        lines.append("- none")

    named_count = summary["candidate_placeholder_counts"].get(
        "catalog_unspecified_named_series", 0
    )
    structural_count = summary["candidate_placeholder_counts"].get(
        "catalog_unspecified_structural_placeholder", 0
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- accept `I3` as a diagnostic label-normalization candidate",
            "- no retrieval, scoring, shortlist, model, reasoning, or Tkinter behavior changed",
            f"- preserve the split for future prompt/evaluation diagnostics: `{named_count}` named-series rows versus `{structural_count}` structural-placeholder rows",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply and report the I3 catalog_unspecified label split."
    )
    parser.add_argument("--i2-rows-csv", required=True, help="I2 row-level audit CSV.")
    parser.add_argument("--output-json", required=True, help="I3 summary JSON.")
    parser.add_argument("--output-md", required=True, help="I3 Markdown review.")
    parser.add_argument("--output-csv", required=True, help="I3 row-level CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.i2_rows_csv)
    summary, candidate = build_audit(rows)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output_csv, index=False)

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_report(summary, candidate), encoding="utf-8")
    print(output_md)


if __name__ == "__main__":
    main()
