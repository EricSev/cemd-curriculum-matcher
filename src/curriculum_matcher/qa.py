from __future__ import annotations

from typing import Any


DEFAULT_SUPPORT_CHALLENGE_THRESHOLD = 0.45
DEFAULT_SCORE_GAP_CHALLENGE_MARGIN = 0.12
HIGH_PRIORITY_SCORE_GAP = 0.20


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _normalize_text(value).lower() in {"true", "1", "yes"}


def review_priority(challenge_flag: bool, score_gap: float) -> str:
    if challenge_flag and score_gap >= HIGH_PRIORITY_SCORE_GAP:
        return "high"
    if challenge_flag:
        return "medium"
    return "low"


def build_human_match_review(
    *,
    human_match_id: str,
    human_scores: dict[str, Any],
    ai_result: dict[str, Any],
    row_context: dict[str, Any] | None = None,
    challenge_margin: float = DEFAULT_SCORE_GAP_CHALLENGE_MARGIN,
    support_threshold: float = DEFAULT_SUPPORT_CHALLENGE_THRESHOLD,
) -> dict[str, Any]:
    if not human_match_id or not human_scores:
        return {}

    human_support_score = float(human_scores.get("human_match_final_score", 0.0))
    human_sem = float(human_scores.get("human_match_name_semantic", 0.0))
    human_fuzz = float(human_scores.get("human_match_name_fuzzy", 0.0))
    human_pub = float(human_scores.get("human_match_publisher_score", 0.0))
    best_ai_matches = ai_result.get("matches", [])
    best_ai_match = best_ai_matches[0] if best_ai_matches else {}
    best_ai_id = "" if not best_ai_match else str(best_ai_match.get("catalog_id", ""))
    best_ai_score = float(best_ai_match.get("final_score", 0.0)) if best_ai_match else 0.0
    context = row_context or {}

    agrees_with_ai_top1 = bool(best_ai_id) and best_ai_id == str(human_match_id)
    score_gap = round(best_ai_score - human_support_score, 4)
    reason_tags: list[str] = []
    weak_title_evidence = human_sem < 0.55 and human_fuzz < 0.45
    evidence_richness = _normalize_text(context.get("evidence_richness"))
    usage_ambiguity = _normalize_text(context.get("usage_ambiguity"))
    state_specific_risk = _normalize_text(context.get("state_specific_risk"))
    placeholder_mapping = _normalize_text(context.get("placeholder_mapping"))
    assessment_slice = _normalize_text(context.get("assessment_slice"))
    product_type_usage = _normalize_text(context.get("product_type_usage"))
    match_strategy = _normalize_text(ai_result.get("selected_strategy", "primary"))
    used_fallback = _as_bool(ai_result.get("used_fallback"))

    if weak_title_evidence:
        reason_tags.append("weak_title_evidence")
    if human_pub < 0.5:
        reason_tags.append("weak_publisher_evidence")
    challenge_flag = (
        weak_title_evidence and human_support_score < support_threshold
    ) or (
        bool(best_ai_id)
        and (not agrees_with_ai_top1 and score_gap >= challenge_margin)
    )
    if human_support_score < support_threshold:
        reason_tags.append("low_support_score")
    if challenge_flag and best_ai_id and not agrees_with_ai_top1:
        reason_tags.append("ai_prefers_alternative")
    if used_fallback:
        reason_tags.append("repaired_input_changed_prediction")
    if evidence_richness in {"sparse", "policy_placeholder_no_info"}:
        reason_tags.append("likely_sparse_evidence")
    if placeholder_mapping in {
        "catalog_unspecified",
        "district_created",
        "no_information_available",
    }:
        reason_tags.append("placeholder_or_unspecified_mapping")
    if state_specific_risk in {"catalog_state_specific_expected", "adoption_state_high_risk"}:
        reason_tags.append("state_specific_variant_risk")
    if usage_ambiguity == "assessment_naming_regime" or assessment_slice.startswith("assessment_"):
        reason_tags.append("assessment_alias_or_subtype_risk")
    if match_strategy and match_strategy != "primary":
        reason_tags.append("non_primary_strategy")
    if not reason_tags:
        reason_tags.append("well_supported")

    if not challenge_flag:
        primary_reason = "well_supported"
    elif (
        product_type_usage == "Assessment"
        or usage_ambiguity == "assessment_naming_regime"
        or assessment_slice in {
            "assessment_short_or_acronym_title",
            "assessment_publisher_missing",
            "assessment_state_specific_expected",
        }
    ):
        primary_reason = "likely_assessment_alias_or_subtype_ambiguity"
    elif placeholder_mapping in {
        "catalog_unspecified",
        "district_created",
        "no_information_available",
    } or state_specific_risk == "catalog_state_specific_expected" or (
        used_fallback and match_strategy != "primary"
    ):
        primary_reason = "likely_ui_catalog_selection_ambiguity"
    elif (
        evidence_richness in {"sparse", "policy_placeholder_no_info"}
        or weak_title_evidence
        or (human_pub < 0.5 and human_support_score < support_threshold)
    ):
        primary_reason = "likely_evidence_sparsity"
    else:
        primary_reason = "likely_matcher_error"

    return {
        "human_match_support_score": round(human_support_score, 4),
        "human_match_challenge_flag": challenge_flag,
        "human_match_challenge_primary_reason": primary_reason,
        "human_match_challenge_secondary_tags": "|".join(reason_tags),
        "human_match_challenge_reasons": "|".join([primary_reason, *reason_tags]),
        "human_match_review_priority": review_priority(
            challenge_flag, score_gap
        ),
        "human_match_best_ai_match_id": best_ai_id,
        "human_match_best_ai_score": round(best_ai_score, 4),
        "human_match_best_ai_strategy": ai_result.get("selected_strategy", "primary"),
        "human_match_agrees_with_ai_top1": agrees_with_ai_top1,
        "human_match_score_gap": score_gap,
    }
