from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from curriculum_matcher.app import EnhancedCurriculumMatcherV312
from curriculum_matcher.evaluation import evaluate_matcher_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a directory of corrupted benchmark slices and write a comparison report."
    )
    parser.add_argument(
        "--benchmark-dir",
        default="benchmarks/corruptions/starter_gold_100",
        help="Directory containing corrupted benchmark CSVs.",
    )
    parser.add_argument("--catalog-file", required=True, help="Catalog CSV file.")
    parser.add_argument(
        "--profile",
        choices=["fast", "accurate"],
        default="fast",
        help="Matcher profile to evaluate.",
    )
    parser.add_argument(
        "--topn-final",
        type=int,
        default=3,
        help="How many final recommendations to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/outputs/corruptions/starter_gold_100_fast",
        help="Directory for per-slice summary and record outputs.",
    )
    parser.add_argument(
        "--report-md",
        default="docs/analysis/2026-04-03-starter-gold-field-corruption-report.md",
        help="Markdown report output path.",
    )
    return parser.parse_args()


def _normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _load_matcher(catalog_file: Path, profile: str) -> EnhancedCurriculumMatcherV312:
    catalog_df = pd.read_csv(catalog_file, encoding="latin-1")
    matcher = EnhancedCurriculumMatcherV312(log_callback=lambda *args, **kwargs: None)
    matcher.load_models(profile=profile)
    matcher.prepare_catalog(catalog_df)
    matcher.catalog_df[matcher.CATALOG_ID] = matcher.catalog_df[matcher.CATALOG_ID].astype(str)
    return matcher


def _expected_gate_reject_rate(
    matcher: EnhancedCurriculumMatcherV312,
    records_df: pd.DataFrame,
) -> float:
    idx_map = {
        catalog_id: idx
        for idx, catalog_id in matcher.catalog_df[matcher.CATALOG_ID].items()
    }
    rejects = 0
    eligible = 0
    for _, row in records_df.iterrows():
        expected_id = _normalize_id(row.get("expected_match_id"))
        if not expected_id or expected_id not in idx_map:
            continue
        candidate = matcher.catalog_df.iloc[idx_map[expected_id]]
        search_text = matcher._normalize(row.get(matcher.INPUT_PRODUCT_NAME), True)
        if not search_text:
            rejects += 1
            eligible += 1
            continue
        input_vec = matcher.model_fast.encode([search_text])
        sem, fuzz = matcher._name_scores(
            input_vec,
            candidate["embedding_fast"],
            row.get(matcher.INPUT_PRODUCT_NAME),
            candidate["search_text"],
        )
        eligible += 1
        if sem < 0.70 and fuzz < 0.55:
            rejects += 1
    return 0.0 if eligible == 0 else rejects / eligible


def _build_report(rows: list[dict[str, Any]], profile: str, topn_final: int) -> str:
    df = pd.DataFrame(rows).sort_values("slice_name")
    clean_row = df.loc[df["slice_name"] == "clean_reference"].iloc[0]

    lines = [
        "# Field Corruption Benchmark Report",
        "",
        f"- profile: `{profile}`",
        f"- topn_final: `{topn_final}`",
        f"- slice count: {len(df)}",
        "",
        "## Summary",
        "",
        "| Slice | Top-1 | Delta | Top-3 | Delta | Prediction | Delta | Gate Reject | Blank Title | Blank Publisher |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in df.iterrows():
        lines.append(
            "| "
            f"{row['slice_name']} | "
            f"{row['top1_accuracy']:.3f} | {row['top1_accuracy'] - clean_row['top1_accuracy']:+.3f} | "
            f"{row['top3_recall']:.3f} | {row['top3_recall'] - clean_row['top3_recall']:+.3f} | "
            f"{row['prediction_rate']:.3f} | {row['prediction_rate'] - clean_row['prediction_rate']:+.3f} | "
            f"{row['expected_gate_reject_rate']:.3f} | "
            f"{row['blank_title_rate']:.3f} | "
            f"{row['blank_publisher_rate']:.3f} |"
        )

    worst_recall = df[df["slice_name"] != "clean_reference"].sort_values("top3_recall").iloc[0]
    worst_prediction = df[df["slice_name"] != "clean_reference"].sort_values("prediction_rate").iloc[0]
    highest_gate = df[df["slice_name"] != "clean_reference"].sort_values(
        "expected_gate_reject_rate", ascending=False
    ).iloc[0]

    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            f"- Worst top-3 recall drop: `{worst_recall['slice_name']}` moved from {clean_row['top3_recall']:.3f} to {worst_recall['top3_recall']:.3f}.",
            f"- Worst prediction-rate drop: `{worst_prediction['slice_name']}` moved from {clean_row['prediction_rate']:.3f} to {worst_prediction['prediction_rate']:.3f}.",
            f"- Highest expected-row gate rejection rate: `{highest_gate['slice_name']}` at {highest_gate['expected_gate_reject_rate']:.3f}.",
            "- High gate rejection suggests the right catalog row is being filtered out before publisher and grade can help.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "./.venv/bin/python scripts/build_corruption_benchmarks.py \\",
            "  --source-benchmark benchmarks/gold/starter_gold_100.csv",
            "",
            "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/evaluate_corruption_benchmarks.py \\",
            "  --benchmark-dir benchmarks/corruptions/starter_gold_100 \\",
            "  --catalog-file '/absolute/path/to/catalog.csv' \\",
            f"  --profile {profile}",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_md)
    catalog_path = Path(args.catalog_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    matcher = _load_matcher(catalog_path, args.profile)

    report_rows: list[dict[str, Any]] = []
    for benchmark_file in sorted(benchmark_dir.glob("*.csv")):
        slice_name = benchmark_file.stem
        summary_path = output_dir / f"{slice_name}-summary.json"
        records_path = output_dir / f"{slice_name}-records.csv"

        artifacts = evaluate_matcher_run(
            benchmark_file,
            catalog_path,
            profile=args.profile,
            topn_final=args.topn_final,
            output_json=summary_path,
            output_csv=records_path,
            log_callback=lambda *args, **kwargs: None,
        )

        record_df = artifacts.record_results
        report_rows.append(
            {
                "slice_name": slice_name,
                "top1_accuracy": float(artifacts.summary["top1_accuracy"]),
                "top3_recall": float(artifacts.summary["top3_recall"]),
                "prediction_rate": float(artifacts.summary["prediction_rate"]),
                "expected_gate_reject_rate": _expected_gate_reject_rate(
                    matcher, record_df
                ),
                "blank_title_rate": float(
                    record_df["product_name_raw"].fillna("").astype(str).str.strip().eq("").mean()
                ),
                "blank_publisher_rate": float(
                    record_df["publisher_raw"].fillna("").astype(str).str.strip().eq("").mean()
                ),
            }
        )

    report_path.write_text(
        _build_report(report_rows, args.profile, args.topn_final),
        encoding="utf-8",
    )
    print(json.dumps(report_rows, indent=2))


if __name__ == "__main__":
    main()
