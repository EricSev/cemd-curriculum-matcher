from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIORITY_SLICE_COLUMNS = [
    "product_type_usage",
    "evidence_richness",
    "placeholder_mapping",
    "state_specific_risk",
    "assessment_slice",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a short shortlist-bottleneck report from an evaluation summary JSON."
    )
    parser.add_argument("--summary-json", required=True, help="Evaluation summary JSON path.")
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional Markdown output path.",
    )
    return parser.parse_args()


def _worst_slice_entry(summary: dict, column: str) -> tuple[str, dict] | None:
    metrics = summary.get("slice_metrics", {}).get(column, {})
    if not metrics:
        return None

    def sort_key(item: tuple[str, dict]) -> tuple[float, int]:
        _, values = item
        return (
            float(values.get("top10_recall", values.get("top3_recall", 0.0))),
            -int(values.get("count", 0)),
        )

    label, values = sorted(metrics.items(), key=sort_key)[0]
    return label, values


def build_report(summary: dict) -> str:
    lines = [
        "# Shortlist Bottleneck Snapshot",
        "",
        f"- benchmark: `{summary.get('benchmark_file', '')}`",
        f"- profile: `{summary.get('profile', '')}`",
        f"- topn_final: `{summary.get('topn_final', '')}`",
        f"- hit@1: `{summary.get('shortlist_metrics', {}).get('hit_rate_at_1', summary.get('top1_accuracy', 0.0))}`",
        f"- hit@3: `{summary.get('shortlist_metrics', {}).get('hit_rate_at_3', summary.get('top3_recall', 0.0))}`",
        f"- hit@10: `{summary.get('shortlist_metrics', {}).get('hit_rate_at_10', 0.0)}`",
        f"- MRR: `{summary.get('shortlist_metrics', {}).get('mrr', 0.0)}`",
        f"- nDCG@10: `{summary.get('shortlist_metrics', {}).get('ndcg_at_10', 0.0)}`",
        "",
        "## Worst recall-limiting slices",
        "",
    ]

    for column in PRIORITY_SLICE_COLUMNS:
        entry = _worst_slice_entry(summary, column)
        if not entry:
            continue
        label, values = entry
        lines.append(
            "- `{column}` worst slice: `{label}` | count `{count}` | top3 `{top3}` | top10 `{top10}`".format(
                column=column,
                label=label,
                count=values.get("count", 0),
                top3=values.get("top3_recall", 0.0),
                top10=values.get("top10_recall", 0.0),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    report = build_report(summary)
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
