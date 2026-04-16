from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SLICE_FILTERS = {
    "Assessment": ("product_type_usage", "Assessment"),
    "catalog_unspecified": ("placeholder_mapping", "catalog_unspecified"),
    "adoption_state_high_risk": ("state_specific_risk", "adoption_state_high_risk"),
    "catalog_state_specific_expected": (
        "state_specific_risk",
        "catalog_state_specific_expected",
    ),
}


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _count_metrics(df: pd.DataFrame) -> dict[str, Any]:
    row_count = int(len(df))
    if row_count == 0:
        return {
            "row_count": 0,
            "gold_absent_from_top10": 0,
            "gold_present_in_top10": 0,
            "ranker_top1_failure_with_gold_present": 0,
            "ranker_top1_success": 0,
            "gold_absent_from_top10_rate": 0.0,
            "gold_present_in_top10_rate": 0.0,
            "ranker_top1_failure_with_gold_present_rate": 0.0,
        }

    top10_correct = _as_bool(df["top10_correct"])
    top1_correct = _as_bool(df["top1_correct"])
    gold_present = int(top10_correct.sum())
    gold_absent = row_count - gold_present
    ranker_top1_success = int(top1_correct.sum())
    ranker_failure_with_gold_present = int((top10_correct & ~top1_correct).sum())
    return {
        "row_count": row_count,
        "gold_absent_from_top10": gold_absent,
        "gold_present_in_top10": gold_present,
        "ranker_top1_failure_with_gold_present": ranker_failure_with_gold_present,
        "ranker_top1_success": ranker_top1_success,
        "gold_absent_from_top10_rate": _safe_rate(gold_absent, row_count),
        "gold_present_in_top10_rate": _safe_rate(gold_present, row_count),
        "ranker_top1_failure_with_gold_present_rate": _safe_rate(
            ranker_failure_with_gold_present,
            row_count,
        ),
    }


def _llm_metrics(df: pd.DataFrame) -> dict[str, Any]:
    row_count = int(len(df))
    if row_count == 0:
        return {
            "llm_row_count": 0,
            "llm_gold_absent_from_top10": 0,
            "llm_gold_present_in_top10": 0,
            "llm_selection_failure_with_gold_present": 0,
            "llm_top1_success": 0,
            "llm_abstained_with_gold_present": 0,
            "llm_selection_failure_with_gold_present_rate": 0.0,
        }

    top10_correct = _as_bool(df["top10_correct"])
    llm_top1_correct = _as_bool(df["llm_top1_correct"])
    llm_abstained = _as_bool(df["llm_abstained"])
    gold_present = int(top10_correct.sum())
    selection_failure = int((top10_correct & ~llm_top1_correct).sum())
    abstained_with_gold_present = int((top10_correct & llm_abstained).sum())
    return {
        "llm_row_count": row_count,
        "llm_gold_absent_from_top10": row_count - gold_present,
        "llm_gold_present_in_top10": gold_present,
        "llm_selection_failure_with_gold_present": selection_failure,
        "llm_top1_success": int(llm_top1_correct.sum()),
        "llm_abstained_with_gold_present": abstained_with_gold_present,
        "llm_selection_failure_with_gold_present_rate": _safe_rate(
            selection_failure,
            row_count,
        ),
    }


def _slice_df(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    return df[df[column].fillna("").astype(str) == value]


def build_audit(records: pd.DataFrame, llm_scored: pd.DataFrame | None) -> dict[str, Any]:
    records = records.copy()
    llm_joined: pd.DataFrame | None = None
    if llm_scored is not None:
        llm_cols = [
            "selection_identifier",
            "llm_top1_correct",
            "llm_abstained",
            "llm_decision",
        ]
        llm_joined = records.merge(
            llm_scored[llm_cols],
            on="selection_identifier",
            how="inner",
        )

    slices: dict[str, Any] = {}
    for label, (column, value) in DEFAULT_SLICE_FILTERS.items():
        base_slice = _slice_df(records, column, value)
        metrics = _count_metrics(base_slice)
        if llm_joined is not None:
            llm_slice = _slice_df(llm_joined, column, value)
            metrics.update(_llm_metrics(llm_slice))
        slices[label] = metrics

    overall = _count_metrics(records)
    if llm_joined is not None:
        overall.update(_llm_metrics(llm_joined))

    return {
        "records_row_count": int(len(records)),
        "llm_joined_row_count": int(len(llm_joined)) if llm_joined is not None else 0,
        "overall": overall,
        "slices": slices,
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def build_report(audit: dict[str, Any]) -> str:
    rows = []
    for label, metrics in audit["slices"].items():
        row = {"slice": label}
        row.update(metrics)
        rows.append(row)

    retrieval_columns = [
        "slice",
        "row_count",
        "gold_absent_from_top10",
        "gold_absent_from_top10_rate",
        "gold_present_in_top10",
        "ranker_top1_failure_with_gold_present",
    ]
    llm_columns = [
        "slice",
        "llm_row_count",
        "llm_gold_absent_from_top10",
        "llm_gold_present_in_top10",
        "llm_selection_failure_with_gold_present",
        "llm_abstained_with_gold_present",
        "llm_top1_success",
    ]

    lines = [
        "# H1 Shortlist Failure Audit",
        "",
        "- Task: `H1` diagnostic checkpoint",
        f"- records rows: `{audit['records_row_count']}`",
        f"- LLM joined rows: `{audit['llm_joined_row_count']}`",
        "- behavior changed: `no`",
        "",
        "## Retrieval / Candidate Recall Buckets",
        "",
        _markdown_table(rows, retrieval_columns),
        "",
        "## LLM Selection Buckets On Joined Rows",
        "",
        _markdown_table(rows, llm_columns),
        "",
        "## Interpretation",
        "",
    ]

    assessment = audit["slices"]["Assessment"]
    unspecified = audit["slices"]["catalog_unspecified"]
    state_specific = audit["slices"]["catalog_state_specific_expected"]
    adoption = audit["slices"]["adoption_state_high_risk"]

    lines.extend(
        [
            f"- `Assessment` has `{assessment['gold_absent_from_top10']}` top-10 recall failures and `{assessment['ranker_top1_failure_with_gold_present']}` ranker-selection failures with the gold row present.",
            f"- `catalog_unspecified` remains recall-limited: `{unspecified['gold_absent_from_top10']}` of `{unspecified['row_count']}` rows do not have the gold row in the top-10.",
            f"- `catalog_state_specific_expected` has a mixed failure shape: `{state_specific['gold_absent_from_top10']}` absent-from-top10 rows and `{state_specific['ranker_top1_failure_with_gold_present']}` gold-present ranker failures.",
            f"- `adoption_state_high_risk` is split across both failure modes: `{adoption['gold_absent_from_top10']}` recall misses and `{adoption['ranker_top1_failure_with_gold_present']}` gold-present ranker failures.",
            "",
            "## Decision",
            "",
            "- accept `H1` as a diagnostic checkpoint",
            "- no behavioral baseline change",
            "- use this audit to prioritize `H2` assessment-aware candidate recall next",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit top-10 recall failures versus selection failures for hard slices."
    )
    parser.add_argument("--records-csv", required=True, help="Evaluation records CSV.")
    parser.add_argument(
        "--llm-scored-csv",
        default=None,
        help="Optional LLM rerank scored CSV to join by selection_identifier.",
    )
    parser.add_argument("--output-json", required=True, help="Audit JSON output path.")
    parser.add_argument("--output-md", required=True, help="Audit Markdown output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = pd.read_csv(args.records_csv)
    llm_scored = pd.read_csv(args.llm_scored_csv) if args.llm_scored_csv else None
    audit = build_audit(records, llm_scored)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_report(audit), encoding="utf-8")

    print(output_md)


if __name__ == "__main__":
    main()
