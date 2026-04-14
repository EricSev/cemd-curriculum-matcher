from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONSULTING_ROOT = Path(
    "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data"
)

RAW_SAMPLE_FILE = CONSULTING_ROOT / "raw-input/samples/CEMD curriculum matching sample dataset.csv"
TRAINING_SAMPLE_FILE = (
    CONSULTING_ROOT
    / "evaluation/historical-runs/Test Data 002/matched_training_sample dataset.csv"
)
CATALOG_FILE = CONSULTING_ROOT / "catalog/snapshots/CEMD Product Catalog - 05212025.csv"

BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
GOLD_DIR = BENCHMARK_ROOT / "gold"
WEAK_LABEL_DIR = BENCHMARK_ROOT / "weak_labels"
SLICE_DIR = BENCHMARK_ROOT / "slices"


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin-1")


def _prepare_benchmark_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["expected_product_identifier"] = result["product_identifier"].astype(str)
    return result


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_assets() -> dict[str, object]:
    raw_df = _load_csv(RAW_SAMPLE_FILE)
    training_df = _load_csv(TRAINING_SAMPLE_FILE)

    gold_baseline = _prepare_benchmark_frame(raw_df.head(100).copy())
    weak_label_baseline = _prepare_benchmark_frame(training_df.head(250).copy())

    missing_publisher_slice = _prepare_benchmark_frame(
        raw_df[raw_df["publisher_raw"].isna() | (raw_df["publisher_raw"].astype(str).str.strip() == "")]
        .head(50)
        .copy()
    )

    publisher_variant_mask = raw_df["publisher_raw"].astype(str).str.contains(
        r"McGraw|Houghton|Savvas|Benchmark|McGraw-Hill",
        case=False,
        na=False,
    )
    publisher_variants_slice = _prepare_benchmark_frame(
        raw_df[publisher_variant_mask].head(50).copy()
    )

    ela_core_slice = _prepare_benchmark_frame(
        raw_df[
            (raw_df["subject"].astype(str).str.upper() == "ELA")
            & (raw_df["product_type_usage"].astype(str) == "Core Curriculum")
        ]
        .head(50)
        .copy()
    )

    _write_csv(gold_baseline, GOLD_DIR / "starter_gold_100.csv")
    _write_csv(weak_label_baseline, WEAK_LABEL_DIR / "starter_weak_labels_250.csv")
    _write_csv(missing_publisher_slice, SLICE_DIR / "slice_missing_publisher_50.csv")
    _write_csv(publisher_variants_slice, SLICE_DIR / "slice_publisher_variants_50.csv")
    _write_csv(ela_core_slice, SLICE_DIR / "slice_ela_core_50.csv")

    manifest = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source_files": {
            "raw_sample": str(RAW_SAMPLE_FILE),
            "training_sample": str(TRAINING_SAMPLE_FILE),
            "catalog": str(CATALOG_FILE),
        },
        "assets": {
            "gold": {
                "starter_gold_100.csv": {
                    "rows": int(len(gold_baseline)),
                    "source": RAW_SAMPLE_FILE.name,
                }
            },
            "weak_labels": {
                "starter_weak_labels_250.csv": {
                    "rows": int(len(weak_label_baseline)),
                    "source": TRAINING_SAMPLE_FILE.name,
                }
            },
            "slices": {
                "slice_missing_publisher_50.csv": {
                    "rows": int(len(missing_publisher_slice)),
                    "source": RAW_SAMPLE_FILE.name,
                    "rule": "publisher_raw missing or blank",
                },
                "slice_publisher_variants_50.csv": {
                    "rows": int(len(publisher_variants_slice)),
                    "source": RAW_SAMPLE_FILE.name,
                    "rule": "publisher_raw contains common publisher variants",
                },
                "slice_ela_core_50.csv": {
                    "rows": int(len(ela_core_slice)),
                    "source": RAW_SAMPLE_FILE.name,
                    "rule": "subject=ELA and product_type_usage=Core Curriculum",
                },
            },
        },
    }

    manifest_path = BENCHMARK_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    manifest = build_assets()
    print(json.dumps(manifest, indent=2))
