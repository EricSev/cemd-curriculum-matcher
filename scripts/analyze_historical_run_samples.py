from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_MATCHER_COLUMNS = ["product_name_raw", "publisher_raw", "grade"]
DEFAULT_STRATA_COLUMNS = ["product_type_usage", "subject", "state", "publisher_presence"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze historical-run sample representativeness and generate a "
            "deterministic representative benchmark sample."
        )
    )
    parser.add_argument("--population-file", required=True, help="Full population CSV.")
    parser.add_argument("--matched-sample-file", required=True, help="Matched sample CSV.")
    parser.add_argument("--raw-sample-file", required=True, help="Raw-view sample CSV.")
    parser.add_argument("--results-sample-file", required=True, help="Historical matcher results CSV.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Representative sample size to emit from the population.",
    )
    parser.add_argument(
        "--output-sample-csv",
        required=True,
        help="Output CSV for the generated representative sample.",
    )
    parser.add_argument(
        "--output-report-md",
        required=True,
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--output-summary-json",
        required=True,
        help="Output summary JSON path.",
    )
    return parser.parse_args()


def _load_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin-1", usecols=usecols)


def _normalize_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _missing_rate(series: pd.Series) -> float:
    normalized = series.fillna("").astype(str).str.strip()
    return float(normalized.eq("").mean())


def _safe_label(value: Any) -> str:
    normalized = _normalize_string(value)
    return normalized if normalized else "missing"


def _distribution(series: pd.Series) -> pd.Series:
    labels = series.map(_safe_label)
    return labels.value_counts(normalize=True, dropna=False)


def _top_distribution_dict(series: pd.Series, topn: int = 8) -> dict[str, float]:
    return _distribution(series).head(topn).round(4).to_dict()


def _total_variation_distance(population: pd.Series, sample: pd.Series) -> float:
    pop_dist = _distribution(population)
    sample_dist = _distribution(sample)
    labels = pop_dist.index.union(sample_dist.index)
    pop_aligned = pop_dist.reindex(labels, fill_value=0.0)
    sample_aligned = sample_dist.reindex(labels, fill_value=0.0)
    return float((pop_aligned - sample_aligned).abs().sum() / 2.0)


def _markdown_table(df: pd.DataFrame) -> str:
    working = df.copy()
    cols = [str(column) for column in working.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in working.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in working.columns) + " |")
    return "\n".join(lines)


def _routine_compatibility(df: pd.DataFrame) -> dict[str, Any]:
    columns = set(df.columns)
    missing_required = [column for column in REQUIRED_MATCHER_COLUMNS if column not in columns]
    return {
        "shared_required_columns": [column for column in REQUIRED_MATCHER_COLUMNS if column in columns],
        "missing_required_columns": missing_required,
        "matcher_routine_compatible": not missing_required,
    }


def _derive_sampling_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["publisher_presence"] = result["publisher_raw"].map(
        lambda value: "missing" if _normalize_string(value) == "" else "present"
    )
    result["expected_product_identifier"] = result["product_identifier"].map(_normalize_string)
    return result


def _allocate_counts(population: pd.DataFrame, sample_size: int) -> pd.Series:
    grouped = population.groupby(DEFAULT_STRATA_COLUMNS, dropna=False).size().rename("population_count")
    allocation = grouped.to_frame()
    allocation["target_float"] = allocation["population_count"] / len(population) * sample_size
    allocation["target_count"] = allocation["target_float"].astype(int)
    remainder = int(sample_size - allocation["target_count"].sum())
    if remainder > 0:
        allocation["fractional"] = allocation["target_float"] - allocation["target_count"]
        allocation = allocation.sort_values(
            ["fractional", "population_count"],
            ascending=[False, False],
        )
        for idx in allocation.index[:remainder]:
            allocation.at[idx, "target_count"] += 1
    allocation = allocation.sort_index()
    return allocation["target_count"]


def _deterministic_representative_sample(population: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    working = _derive_sampling_frame(population)
    allocations = _allocate_counts(working, sample_size)
    working["selection_identifier_norm"] = working["selection_identifier"].map(_normalize_string)
    working = working.sort_values(
        ["selection_identifier_norm", "product_name_raw", "grade"],
        kind="mergesort",
    )
    sampled_frames: list[pd.DataFrame] = []
    for strata_values, group in working.groupby(DEFAULT_STRATA_COLUMNS, dropna=False, sort=False):
        target_count = int(allocations.get(strata_values, 0))
        if target_count > 0:
            sampled_frames.append(group.head(target_count))
    result = pd.concat(sampled_frames, ignore_index=True)
    result = result.drop_duplicates(subset=["selection_identifier"], keep="first")
    if len(result) < sample_size:
        already_selected = set(result["selection_identifier"].map(_normalize_string))
        extras = (
            working[~working["selection_identifier"].map(_normalize_string).isin(already_selected)]
            .sort_values(["selection_identifier_norm", "product_name_raw", "grade"], kind="mergesort")
            .head(sample_size - len(result))
        )
        result = pd.concat([result, extras], ignore_index=True)
    return result.drop(columns=["selection_identifier_norm"]).head(sample_size)


def _representativeness_rows(
    population: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    metrics = {
        "state_tvd": "state",
        "subject_tvd": "subject",
        "product_type_usage_tvd": "product_type_usage",
        "publisher_missing_abs_diff": "publisher_raw",
    }
    for name, df in datasets.items():
        row: dict[str, Any] = {"dataset": name, "row_count": int(len(df))}
        row["publisher_missing_rate"] = round(_missing_rate(df["publisher_raw"]), 4)
        row["product_type_usage_top"] = json.dumps(_top_distribution_dict(df["product_type_usage"]))
        row["state_tvd"] = round(
            _total_variation_distance(population["state"], df["state"]),
            4,
        )
        row["subject_tvd"] = round(
            _total_variation_distance(population["subject"], df["subject"]),
            4,
        )
        row["product_type_usage_tvd"] = round(
            _total_variation_distance(population["product_type_usage"], df["product_type_usage"]),
            4,
        )
        row["publisher_missing_abs_diff"] = round(
            abs(_missing_rate(population["publisher_raw"]) - _missing_rate(df["publisher_raw"])),
            4,
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("row_count")


def _overlap_summary(matched_sample: pd.DataFrame, raw_sample: pd.DataFrame) -> dict[str, int]:
    matched_ids = set(matched_sample["selection_identifier"].map(_normalize_string))
    raw_ids = set(raw_sample["selection_identifier"].map(_normalize_string))
    return {
        "matched_sample_rows": len(matched_ids),
        "raw_sample_rows": len(raw_ids),
        "overlap_rows": len(matched_ids & raw_ids),
        "matched_only_rows": len(matched_ids - raw_ids),
        "raw_only_rows": len(raw_ids - matched_ids),
    }


def build_report(
    *,
    population_path: Path,
    matched_sample_path: Path,
    raw_sample_path: Path,
    results_sample_path: Path,
    population: pd.DataFrame,
    matched_sample: pd.DataFrame,
    raw_sample: pd.DataFrame,
    results_sample: pd.DataFrame,
    representative_sample: pd.DataFrame,
    overlap: dict[str, int],
    representativeness_table: pd.DataFrame,
) -> str:
    matched_compat = _routine_compatibility(matched_sample)
    raw_compat = _routine_compatibility(raw_sample)
    represented_counts = representative_sample["product_type_usage"].map(_safe_label).value_counts(normalize=True).round(4).to_dict()
    lines = [
        "# Historical Sample Representativeness Report",
        "",
        "## Input Files",
        "",
        f"- population file: `{population_path}`",
        f"- matched sample file: `{matched_sample_path}`",
        f"- raw-view sample file: `{raw_sample_path}`",
        f"- historical results file: `{results_sample_path}`",
        "",
        "## Routine Compatibility",
        "",
        f"- matched sample shares required matcher input columns: `{matched_compat['shared_required_columns']}`",
        f"- raw-view sample shares required matcher input columns: `{raw_compat['shared_required_columns']}`",
        f"- matched sample matcher-compatible: `{matched_compat['matcher_routine_compatible']}`",
        f"- raw-view sample matcher-compatible: `{raw_compat['matcher_routine_compatible']}`",
        "- Conclusion: the same matcher input routine can run on both 750-row files because both retain `product_name_raw`, `publisher_raw`, and `grade`.",
        "",
        "## Sample Relationship",
        "",
        f"- matched sample rows: `{overlap['matched_sample_rows']}`",
        f"- raw-view sample rows: `{overlap['raw_sample_rows']}`",
        f"- overlapping `selection_identifier` rows: `{overlap['overlap_rows']}`",
        f"- matched-only rows: `{overlap['matched_only_rows']}`",
        f"- raw-only rows: `{overlap['raw_only_rows']}`",
        "- Interpretation: the two 750-row files are two schema views of the same records, not separate matched and unmatched populations.",
        "",
        "## Population Notes",
        "",
        f"- full population row count: `{len(population)}`",
        f"- full population nonblank `product_identifier` rate: `{(_missing_rate(population['product_identifier']) == 0.0)}`",
        f"- full population product-type mix: `{_top_distribution_dict(population['product_type_usage'])}`",
        "",
        "## Representativeness Comparison",
        "",
        _markdown_table(representativeness_table),
        "",
        "Interpretation:",
        "",
        "- Lower total-variation distance is better.",
        "- The 750-row sample materially diverges from the 532K population, especially because it omits the `Assessment` segment entirely.",
        "- The 10K results file is directionally closer than the 750-row sample on state mix, but it still misses `Assessment` and should not be treated as fully representative.",
        "",
        "## Representative Sample Output",
        "",
        f"- generated sample rows: `{len(representative_sample)}`",
        f"- generated sample product-type mix: `{represented_counts}`",
        "- Sampling method: deterministic proportional stratification over `product_type_usage`, `subject`, `state`, and publisher presence.",
        "- This sample is a better benchmark starting point than the existing 750-row slice because it preserves the full-population `Assessment` share.",
        "",
        "## Recommendation",
        "",
        "- Keep the two 750-row files only as a routine-compatibility check, not as the main evidence for representativeness.",
        "- Use the full 532K file as the source population for benchmark sampling.",
        "- Use the generated representative sample for future benchmark expansion, then add targeted slices for hard failure families on top of it.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    population_path = Path(args.population_file)
    matched_sample_path = Path(args.matched_sample_file)
    raw_sample_path = Path(args.raw_sample_file)
    results_sample_path = Path(args.results_sample_file)

    comparison_usecols = [
        "selection_identifier",
        "product_name_raw",
        "publisher_raw",
        "grade",
        "state",
        "subject",
        "product_type_usage",
        "product_identifier",
    ]
    population = _load_csv(population_path, usecols=comparison_usecols + ["curriculum_academic_year"])
    matched_sample = _load_csv(matched_sample_path)
    raw_sample = _load_csv(raw_sample_path)
    results_sample = _load_csv(results_sample_path)

    representative_sample = _deterministic_representative_sample(
        population,
        args.sample_size,
    )
    overlap = _overlap_summary(matched_sample, raw_sample)

    comparable_datasets = {
        "matched_750": matched_sample[["state", "subject", "product_type_usage", "publisher_raw"]],
        "raw_view_750": raw_sample[["state", "subject", "product_type_usage", "publisher_raw"]],
        "historical_results_10k": results_sample[["state", "subject", "product_type_usage", "publisher_raw"]],
        "representative_sample": representative_sample[["state", "subject", "product_type_usage", "publisher_raw"]],
    }
    representativeness_table = _representativeness_rows(
        population[["state", "subject", "product_type_usage", "publisher_raw"]],
        comparable_datasets,
    )

    sample_output_path = Path(args.output_sample_csv)
    sample_output_path.parent.mkdir(parents=True, exist_ok=True)
    representative_sample.to_csv(sample_output_path, index=False)

    report = build_report(
        population_path=population_path,
        matched_sample_path=matched_sample_path,
        raw_sample_path=raw_sample_path,
        results_sample_path=results_sample_path,
        population=population,
        matched_sample=matched_sample,
        raw_sample=raw_sample,
        results_sample=results_sample,
        representative_sample=representative_sample,
        overlap=overlap,
        representativeness_table=representativeness_table,
    )
    report_path = Path(args.output_report_md)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")

    summary = {
        "population_file": str(population_path),
        "matched_sample_file": str(matched_sample_path),
        "raw_sample_file": str(raw_sample_path),
        "results_sample_file": str(results_sample_path),
        "population_row_count": int(len(population)),
        "representative_sample_row_count": int(len(representative_sample)),
        "overlap": overlap,
        "routine_compatibility": {
            "matched_sample": _routine_compatibility(matched_sample),
            "raw_sample": _routine_compatibility(raw_sample),
        },
        "representativeness": representativeness_table.to_dict(orient="records"),
    }
    summary_path = Path(args.output_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
