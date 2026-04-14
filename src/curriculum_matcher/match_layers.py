from __future__ import annotations

from typing import Any


def top_match_score(matches: list[dict[str, Any]]) -> float:
    if not matches:
        return 0.0
    return float(matches[0].get("final_score", 0.0))


def top_match_id(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return ""
    value = matches[0].get("catalog_id")
    return "" if value is None else str(value)


def should_try_repairs(
    matches: list[dict[str, Any]], confidence_threshold: float
) -> bool:
    return (not matches) or top_match_score(matches) < confidence_threshold


def choose_match_strategy(
    primary_strategy: str,
    primary_matches: list[dict[str, Any]],
    fallback_results: dict[str, list[dict[str, Any]]],
    *,
    confidence_threshold: float,
    replacement_margin: float,
    strategy_replacement_margins: dict[str, float] | None = None,
) -> dict[str, Any]:
    primary_score = top_match_score(primary_matches)
    selected_strategy = primary_strategy
    selected_matches = primary_matches
    selected_score = primary_score
    used_fallback = False

    variant_scores = {
        primary_strategy: {
            "top1_score": round(primary_score, 4),
            "top1_id": top_match_id(primary_matches),
            "prediction_count": len(primary_matches),
        }
    }

    for strategy_name, matches in fallback_results.items():
        variant_score = top_match_score(matches)
        strategy_margin = (
            strategy_replacement_margins.get(strategy_name, replacement_margin)
            if strategy_replacement_margins
            else replacement_margin
        )
        variant_scores[strategy_name] = {
            "top1_score": round(variant_score, 4),
            "top1_id": top_match_id(matches),
            "prediction_count": len(matches),
        }

        if not matches:
            continue

        primary_missing = not primary_matches
        beats_primary = variant_score >= primary_score + strategy_margin
        improves_selected = variant_score >= selected_score + strategy_margin
        if primary_missing or (
            primary_score < confidence_threshold and improves_selected
        ) or (beats_primary and variant_score > selected_score):
            selected_strategy = strategy_name
            selected_matches = matches
            selected_score = variant_score
            used_fallback = strategy_name != primary_strategy

    return {
        "matches": selected_matches,
        "selected_strategy": selected_strategy,
        "used_fallback": used_fallback,
        "primary_top1_score": round(primary_score, 4),
        "selected_top1_score": round(selected_score, 4),
        "variant_scores": variant_scores,
    }
