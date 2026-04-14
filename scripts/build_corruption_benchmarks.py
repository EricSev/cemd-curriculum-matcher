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

from curriculum_matcher.benchmark_corruptions import build_corruption_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create corrupted benchmark slices from a source benchmark CSV."
    )
    parser.add_argument(
        "--source-benchmark",
        default="benchmarks/gold/starter_gold_100.csv",
        help="Source benchmark CSV to corrupt.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/corruptions/starter_gold_100",
        help="Directory where corrupted slices should be written.",
    )
    parser.add_argument(
        "--manifest-json",
        default="benchmarks/corruptions/starter_gold_100/manifest.json",
        help="Path for the generated corruption manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_path = Path(args.source_benchmark)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_json)

    source_df = pd.read_csv(source_path, encoding="latin-1")
    assets = build_corruption_assets(source_df)

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_benchmark": str(source_path),
        "row_count": int(len(source_df)),
        "assets": {},
    }

    for name, df in assets.items():
        output_path = output_dir / f"{name}.csv"
        df.to_csv(output_path, index=False)
        manifest["assets"][name] = {
            "path": str(output_path),
            "rows": int(len(df)),
            "columns_added": [
                column
                for column in [
                    "original_product_name_raw",
                    "original_publisher_raw",
                    "original_grade",
                    "corruption_type",
                    "corruption_notes",
                ]
                if column in df.columns
            ],
        }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
