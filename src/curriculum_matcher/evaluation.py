from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .app import EnhancedCurriculumMatcherV312


DEFAULT_SLICE_COLUMNS = [
    "subject",
    "product_type_usage",
    "state",
    "publisher_raw",
]

POLICY_SLICE_COLUMNS = [
    "evidence_richness",
    "usage_ambiguity",
    "state_specific_risk",
    "placeholder_mapping",
    "assessment_slice",
    "assessment_short_or_acronym",
    "assessment_publisher_missing",
    "assessment_state_specific_expected",
]

CROSS_SLICE_COLUMNS = [
    ("product_type_usage", "evidence_richness"),
    ("product_type_usage", "placeholder_mapping"),
    ("product_type_usage", "state_specific_risk"),
    ("product_type_usage", "assessment_slice"),
]

SUPPORTIVE_USAGE_PATTERNS = [
    r"\bintervention\b",
    r"\btier ii\b",
    r"\btier iii\b",
    r"\bsupplement(?:al)?\b",
    r"\bresource\b",
    r"\bpractice\b",
    r"\bremedial\b",
]

CORE_USAGE_PATTERNS = [
    r"\bcore\b",
    r"\bmain\b",
    r"\bbenchmark\b",
    r"\badopted\b",
    r"\bcurriculum\b",
]

VARIANT_HINT_PATTERNS = [
    r"\bca\b",
    r"\bcalifornia\b",
    r"\btx\b",
    r"\btexas\b",
    r"\bfl\b",
    r"\bflorida\b",
    r"\bspanish\b",
    r"\bbilingual\b",
    r"\bintervention\b",
    r"\bel\b",
    r"\bell\b",
    r"\bgrowth\b",
    r"\baspire\b",
    r"\badvance\b",
    r"\bedition\b",
]


@dataclass
class EvaluationArtifacts:
    summary: dict[str, Any]
    record_results: pd.DataFrame


def _normalize_expected_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def resolve_expected_id_column(
    benchmark_df: pd.DataFrame, requested_column: str | None = None
) -> str:
    if requested_column and requested_column in benchmark_df.columns:
        return requested_column

    fallbacks = [
        "expected_product_identifier",
        "expected_catalog_id",
        "expected_match_id",
        "product_identifier",
    ]
    for column in fallbacks:
        if column in benchmark_df.columns:
            return column
    raise ValueError(
        "Benchmark file must include an expected id column. "
        "Supported defaults: expected_product_identifier, expected_catalog_id, "
        "expected_match_id, or product_identifier."
    )


def _confidence_band(score: float | None) -> str:
    if score is None:
        return "no_prediction"
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    if score >= 0.3:
        return "low"
    return "very_low"


def _find_expected_rank(matches: list[dict[str, Any]], expected_id: str) -> int | None:
    for idx, match in enumerate(matches, start=1):
        if _normalize_expected_id(match.get("catalog_id")) == expected_id:
            return idx
    return None


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _token_count(value: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", value))


def _has_year(value: str) -> bool:
    return bool(re.search(r"\b(?:19|20)\d{2}\b", value))


def _has_any_pattern(value: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def _is_short_or_acronym_title(value: str) -> bool:
    token_count = _token_count(value)
    if token_count <= 2:
        return True
    return bool(re.fullmatch(r"[A-Z0-9\-\s'/&:]+", value)) and token_count <= 4


def _truthy_flag(value: Any) -> bool:
    return _normalize_text(value).lower() == "true"


def _expected_catalog_record(
    record: dict[str, Any], catalog_lookup: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    expected_id = _normalize_expected_id(
        record.get("expected_product_identifier")
        or record.get("expected_match_id")
        or record.get("product_identifier")
    )
    return catalog_lookup.get(expected_id, {})


def _derive_evidence_richness(
    record: dict[str, Any], expected_catalog_row: dict[str, Any]
) -> str:
    title = _normalize_text(record.get("product_name_raw"))
    publisher = _normalize_text(record.get("publisher_raw"))
    lower_title = title.lower()

    if lower_title in {"not available", "n/a", "na"}:
        return "policy_placeholder_no_info"

    title_tokens = _token_count(title)
    has_publisher = bool(publisher)
    has_year = _has_year(title) or _has_year(publisher)
    has_variant_hint = _has_any_pattern(title, VARIANT_HINT_PATTERNS) or _truthy_flag(
        expected_catalog_row.get("state_specific_version")
    )

    if has_publisher and (has_year or has_variant_hint or title_tokens >= 4):
        return "rich"
    if has_publisher or has_year or has_variant_hint or title_tokens >= 3:
        return "medium"
    return "sparse"


def _derive_usage_ambiguity(record: dict[str, Any]) -> str:
    usage = _normalize_text(record.get("product_type_usage"))
    title = _normalize_text(record.get("product_name_raw"))

    if usage == "Assessment":
        return "assessment_naming_regime"
    if _has_any_pattern(title, SUPPORTIVE_USAGE_PATTERNS + CORE_USAGE_PATTERNS):
        return "explicit_usage"
    return "implicit_usage"


def _derive_state_specific_risk(
    record: dict[str, Any], expected_catalog_row: dict[str, Any]
) -> str:
    if _truthy_flag(expected_catalog_row.get("state_specific_version")):
        return "catalog_state_specific_expected"
    if _normalize_text(record.get("state")) in {"CA", "FL", "TX"}:
        return "adoption_state_high_risk"
    return "standard_state"


def _derive_placeholder_mapping(
    record: dict[str, Any], expected_catalog_row: dict[str, Any]
) -> str:
    raw_title = _normalize_text(record.get("product_name_raw")).lower()
    expected_product = _normalize_text(expected_catalog_row.get("product_name")).lower()
    expected_series = _normalize_text(expected_catalog_row.get("series")).lower()

    if _truthy_flag(expected_catalog_row.get("district_created")):
        return "district_created"
    if "unspecified" in expected_product or "unspecified" in expected_series:
        return "catalog_unspecified"
    if raw_title in {"not available", "n/a", "na"} or "not publicly available" in (
        expected_product + " " + expected_series
    ):
        return "no_information_available"
    return "standard_catalog_mapping"


def _derive_assessment_slice(
    record: dict[str, Any], expected_catalog_row: dict[str, Any]
) -> str:
    if _normalize_text(record.get("product_type_usage")) != "Assessment":
        return "not_assessment"

    title = _normalize_text(record.get("product_name_raw"))
    publisher = _normalize_text(record.get("publisher_raw"))

    if _truthy_flag(expected_catalog_row.get("state_specific_version")):
        return "assessment_state_specific_expected"
    if _is_short_or_acronym_title(title):
        return "assessment_short_or_acronym_title"
    if not publisher:
        return "assessment_publisher_missing"
    return "assessment_other"


def _enrich_record_for_policy_slices(
    record: dict[str, Any], catalog_lookup: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    expected_catalog_row = _expected_catalog_record(record, catalog_lookup)
    assessment_slice = _derive_assessment_slice(record, expected_catalog_row)
    is_assessment = _normalize_text(record.get("product_type_usage")) == "Assessment"

    return {
        "evidence_richness": _derive_evidence_richness(record, expected_catalog_row),
        "usage_ambiguity": _derive_usage_ambiguity(record),
        "state_specific_risk": _derive_state_specific_risk(record, expected_catalog_row),
        "placeholder_mapping": _derive_placeholder_mapping(record, expected_catalog_row),
        "assessment_slice": assessment_slice,
        "assessment_short_or_acronym": (
            "yes" if assessment_slice == "assessment_short_or_acronym_title" else "no"
        ),
        "assessment_publisher_missing": (
            "yes"
            if is_assessment and not _normalize_text(record.get("publisher_raw"))
            else "no"
        ),
        "assessment_state_specific_expected": (
            "yes"
            if is_assessment
            and _truthy_flag(expected_catalog_row.get("state_specific_version"))
            else "no"
        ),
    }


def _slice_accuracy(record_results: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in record_results.columns:
        return {}

    grouped: dict[str, Any] = {}
    for value, group in record_results.groupby(column, dropna=False):
        label = "missing" if pd.isna(value) or value == "" else str(value)
        grouped[label] = _group_ranking_metrics(group)
    return grouped


def _cross_slice_accuracy(
    record_results: pd.DataFrame, first_column: str, second_column: str
) -> dict[str, Any]:
    if first_column not in record_results.columns or second_column not in record_results.columns:
        return {}

    grouped: dict[str, Any] = {}
    grouped_frame = record_results.groupby([first_column, second_column], dropna=False)
    for (first_value, second_value), group in grouped_frame:
        first_label = (
            "missing" if pd.isna(first_value) or first_value == "" else str(first_value)
        )
        second_label = (
            "missing" if pd.isna(second_value) or second_value == "" else str(second_value)
        )
        grouped[f"{first_label} | {second_label}"] = _group_ranking_metrics(group)
    return grouped


def _confidence_band_accuracy(record_results: pd.DataFrame) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for band, group in record_results.groupby("confidence_band", dropna=False):
        grouped[str(band)] = _group_ranking_metrics(group)
    return grouped


def _available_rank_cutoffs(record_results: pd.DataFrame) -> list[int]:
    cutoffs: list[int] = []
    for cutoff in (1, 3, 10):
        rank_column = f"pred_rank_{cutoff}_id"
        if rank_column in record_results.columns:
            cutoffs.append(cutoff)
    return cutoffs


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return round(float(series.mean()), 4)


def _group_ranking_metrics(group: pd.DataFrame) -> dict[str, Any]:
    metrics = {
        "count": int(len(group)),
        "top1_accuracy": _safe_mean(group["top1_correct"]),
        "top3_recall": _safe_mean(group["top3_correct"]),
    }

    if "top10_correct" in group.columns:
        metrics["top10_recall"] = _safe_mean(group["top10_correct"])
    if "reciprocal_rank" in group.columns:
        metrics["mrr"] = _safe_mean(group["reciprocal_rank"])
    if "ndcg_at_10" in group.columns:
        metrics["ndcg_at_10"] = _safe_mean(group["ndcg_at_10"])
    return metrics


def _reciprocal_rank(expected_rank: int | None) -> float:
    if expected_rank is None or expected_rank <= 0:
        return 0.0
    return round(1.0 / expected_rank, 4)


def _ndcg_at_k(expected_rank: int | None, k: int) -> float:
    if expected_rank is None or expected_rank <= 0 or expected_rank > k:
        return 0.0
    return round(1.0 / math.log2(expected_rank + 1), 4)


def _build_summary(
    record_results: pd.DataFrame,
    benchmark_file: str,
    catalog_file: str,
    profile: str,
    expected_id_column: str,
    topn_final: int,
    rerank_experiment: str,
    cross_encoder_model: str | None,
) -> dict[str, Any]:
    available_cutoffs = _available_rank_cutoffs(record_results)
    shortlist_metrics = {}
    for cutoff in available_cutoffs:
        column_name = f"top{cutoff}_correct"
        if column_name in record_results.columns:
            shortlist_metrics[f"hit_rate_at_{cutoff}"] = _safe_mean(
                record_results[column_name]
            )

    summary = {
        "benchmark_file": benchmark_file,
        "catalog_file": catalog_file,
        "profile": profile,
        "rerank_experiment": rerank_experiment,
        "cross_encoder_model": cross_encoder_model,
        "expected_id_column": expected_id_column,
        "topn_final": topn_final,
        "record_count": int(len(record_results)),
        "top1_accuracy": round(float(record_results["top1_correct"].mean()), 4),
        "top3_recall": round(float(record_results["top3_correct"].mean()), 4),
        "prediction_rate": round(
            float((record_results["predicted_match_id"] != "").mean()), 4
        ),
        "mean_top1_score": round(float(record_results["top1_score"].mean()), 4),
        "mean_correct_top1_score": round(
            float(record_results.loc[record_results["top1_correct"], "top1_score"].mean())
            if record_results["top1_correct"].any()
            else 0.0,
            4,
        ),
        "shortlist_metrics": {
            **shortlist_metrics,
            "mrr": _safe_mean(record_results["reciprocal_rank"]),
            "ndcg_at_10": _safe_mean(record_results["ndcg_at_10"]),
            "available_cutoffs": available_cutoffs,
        },
        "confidence_bands": _confidence_band_accuracy(record_results),
        "repair_metrics": {
            "repair_attempt_rate": round(
                float(record_results["repair_attempted"].mean()), 4
            )
            if "repair_attempted" in record_results.columns
            else 0.0,
            "fallback_usage_rate": round(
                float(record_results["match_used_fallback"].mean()), 4
            )
            if "match_used_fallback" in record_results.columns
            else 0.0,
            "selected_strategy_counts": record_results["match_selected_strategy"]
            .fillna("primary")
            .value_counts()
            .to_dict()
            if "match_selected_strategy" in record_results.columns
            else {},
        },
        "slice_metrics": {},
        "cross_slice_metrics": {},
    }

    for column in DEFAULT_SLICE_COLUMNS + POLICY_SLICE_COLUMNS:
        slice_metrics = _slice_accuracy(record_results, column)
        if slice_metrics:
            summary["slice_metrics"][column] = slice_metrics

    for first_column, second_column in CROSS_SLICE_COLUMNS:
        cross_metrics = _cross_slice_accuracy(record_results, first_column, second_column)
        if cross_metrics:
            summary["cross_slice_metrics"][f"{first_column}__x__{second_column}"] = cross_metrics

    return summary


def evaluate_matcher_run(
    benchmark_file: str | Path,
    catalog_file: str | Path,
    *,
    profile: str = "accurate",
    retrieval_experiment: str | None = None,
    rerank_experiment: str | None = None,
    cross_encoder_model: str | None = None,
    expected_id_column: str | None = None,
    topn_final: int = 3,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    log_callback=print,
) -> EvaluationArtifacts:
    benchmark_path = Path(benchmark_file)
    catalog_path = Path(catalog_file)

    benchmark_df = pd.read_csv(benchmark_path, encoding="latin-1")
    catalog_df = pd.read_csv(catalog_path, encoding="latin-1")

    resolved_expected_column = resolve_expected_id_column(
        benchmark_df, requested_column=expected_id_column
    )

    matcher = EnhancedCurriculumMatcherV312(
        log_callback=log_callback,
        retrieval_experiment=retrieval_experiment,
        rerank_experiment=rerank_experiment,
        cross_encoder_model=cross_encoder_model,
    )
    matcher.load_models(profile=profile)
    matcher.prepare_catalog(catalog_df)
    catalog_lookup = {
        _normalize_expected_id(row.get("product_identifier")): row
        for row in catalog_df.to_dict(orient="records")
    }

    record_results: list[dict[str, Any]] = []
    for _, row in benchmark_df.iterrows():
        record = row.to_dict()
        expected_id = _normalize_expected_id(record.get(resolved_expected_column))
        match_result = matcher.match_record_with_repairs(record, topn_final=topn_final)
        matches = match_result["matches"]

        predicted_match_id = (
            _normalize_expected_id(matches[0]["catalog_id"]) if matches else ""
        )
        expected_rank = _find_expected_rank(matches, expected_id) if expected_id else None
        top1_score = float(matches[0]["final_score"]) if matches else 0.0

        result_row = {
            **record,
            "expected_match_id": expected_id,
            "predicted_match_id": predicted_match_id,
            "expected_rank": expected_rank or 0,
            "top1_correct": bool(expected_rank == 1),
            "top3_correct": bool(expected_rank is not None and expected_rank <= 3),
            "top10_correct": bool(expected_rank is not None and expected_rank <= 10),
            "reciprocal_rank": _reciprocal_rank(expected_rank),
            "ndcg_at_10": _ndcg_at_k(expected_rank, 10),
            "top1_score": round(top1_score, 4),
            "confidence_band": _confidence_band(top1_score if matches else None),
            "repair_attempted": bool(match_result.get("repair_attempted")),
            "match_used_fallback": bool(match_result.get("used_fallback")),
            "match_selected_strategy": match_result.get("selected_strategy", "primary"),
            "match_primary_top1_score": match_result.get("primary_top1_score", 0.0),
            "match_selected_top1_score": match_result.get("selected_top1_score", 0.0),
            "match_variant_scores": json.dumps(
                match_result.get("variant_scores", {}), sort_keys=True
            ),
        }
        result_row.update(_enrich_record_for_policy_slices(record, catalog_lookup))

        for rank in range(1, topn_final + 1):
            if rank <= len(matches):
                match = matches[rank - 1]
                result_row[f"pred_rank_{rank}_id"] = _normalize_expected_id(
                    match.get("catalog_id")
                )
                result_row[f"pred_rank_{rank}_score"] = round(
                    float(match.get("final_score", 0.0)), 4
                )
            else:
                result_row[f"pred_rank_{rank}_id"] = ""
                result_row[f"pred_rank_{rank}_score"] = ""

        record_results.append(result_row)

    record_results_df = pd.DataFrame(record_results)
    summary = _build_summary(
        record_results_df,
        benchmark_file=str(benchmark_path),
        catalog_file=str(catalog_path),
        profile=profile,
        expected_id_column=resolved_expected_column,
        topn_final=topn_final,
        rerank_experiment=matcher.rerank_experiment,
        cross_encoder_model=(
            matcher.cross_encoder_model_name
            if matcher.rerank_experiment == "cross_encoder"
            else None
        ),
    )

    if output_json:
        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if output_csv:
        output_csv_path = Path(output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        record_results_df.to_csv(output_csv_path, index=False)

    return EvaluationArtifacts(summary=summary, record_results=record_results_df)
