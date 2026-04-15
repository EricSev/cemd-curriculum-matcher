from __future__ import annotations

import json
import re
from difflib import get_close_matches
from dataclasses import dataclass
from typing import Any

import pandas as pd


SYSTEM_PROMPT = """You are evaluating curriculum-to-catalog matches.

Pick the best candidate from a small shortlist using district-facing evidence, collection-policy realism, and candidate metadata.

Important rules:
- Prefer the best candidate from the provided shortlist only.
- Respect district usage context when present. Selection type reflects district usage, not intrinsic market identity.
- Sparse evidence is allowed. Do not invent missing publisher, year, or variant detail.
- Treat state-specific variants, unspecified placeholders, and assessment subtype aliases as meaningful ambiguity sources.
- If none of the provided candidates are well supported, you may abstain.

Return strict JSON only."""


JSON_RESPONSE_INSTRUCTIONS = {
    "selected_candidate_id": "catalog identifier from the shortlist, or empty string when abstaining",
    "decision": "one of: select_candidate, abstain",
    "confidence": "number between 0 and 1",
    "primary_reason": (
        "one of: likely_matcher_error, likely_evidence_sparsity, "
        "likely_ui_catalog_selection_ambiguity, likely_assessment_alias_or_subtype_ambiguity, "
        "well_supported"
    ),
    "secondary_tags": ["short flat tags such as acronym_only, state_specific_variant_risk"],
    "reasoning": "brief explanation grounded in row evidence and shortlist comparison",
}

RERANK_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_candidate_id": {"type": "string"},
        "decision": {
            "type": "string",
            "enum": ["select_candidate", "abstain"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "primary_reason": {
            "type": "string",
            "enum": [
                "likely_matcher_error",
                "likely_evidence_sparsity",
                "likely_ui_catalog_selection_ambiguity",
                "likely_assessment_alias_or_subtype_ambiguity",
                "well_supported",
            ],
        },
        "secondary_tags": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {"type": "string"},
    },
    "required": [
        "selected_candidate_id",
        "decision",
        "confidence",
        "primary_reason",
        "secondary_tags",
        "reasoning",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class LLMCandidate:
    catalog_id: str
    rank: int
    score: float
    product_name: str
    series: str
    publisher: str
    grades: str
    copyright_year: str
    product_type: str
    state_specific_version: str
    district_created: str
    embedded_assessment: str


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_id(value: Any) -> str:
    return _normalize_text(value)


def _normalize_bool(value: Any) -> str:
    return "true" if _normalize_text(value).lower() == "true" else "false"


def _catalog_lookup(catalog_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows = catalog_df.to_dict(orient="records")
    return {_normalize_id(row.get("product_identifier")): row for row in rows}


def _build_candidate(
    record_row: dict[str, Any],
    rank: int,
    catalog_row: dict[str, Any],
) -> LLMCandidate:
    return LLMCandidate(
        catalog_id=_normalize_id(record_row.get(f"pred_rank_{rank}_id")),
        rank=rank,
        score=float(record_row.get(f"pred_rank_{rank}_score") or 0.0),
        product_name=_normalize_text(catalog_row.get("product_name")),
        series=_normalize_text(catalog_row.get("series")),
        publisher=_normalize_text(catalog_row.get("publisher")),
        grades=_normalize_text(catalog_row.get("intended_grades")),
        copyright_year=_normalize_text(catalog_row.get("copyright_year")),
        product_type=_normalize_text(catalog_row.get("product_type")),
        state_specific_version=_normalize_bool(catalog_row.get("state_specific_version")),
        district_created=_normalize_bool(catalog_row.get("district_created")),
        embedded_assessment=_normalize_bool(catalog_row.get("embedded_assessment")),
    )


def build_candidate_shortlist(
    record_row: dict[str, Any],
    catalog_df: pd.DataFrame,
    *,
    topn_final: int = 3,
    catalog_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[LLMCandidate]:
    lookup = catalog_lookup or _catalog_lookup(catalog_df)
    candidates: list[LLMCandidate] = []
    for rank in range(1, topn_final + 1):
        candidate_id = _normalize_id(record_row.get(f"pred_rank_{rank}_id"))
        if not candidate_id:
            continue
        catalog_row = lookup.get(candidate_id)
        if not catalog_row:
            continue
        candidates.append(_build_candidate(record_row, rank, catalog_row))
    return candidates


def build_rerank_prompt(
    record_row: dict[str, Any],
    candidates: list[LLMCandidate],
) -> str:
    row_context = {
        "selection_identifier": _normalize_text(record_row.get("selection_identifier")),
        "state": _normalize_text(record_row.get("state")),
        "subject": _normalize_text(record_row.get("subject")),
        "grade": _normalize_text(record_row.get("grade")),
        "product_type_usage": _normalize_text(record_row.get("product_type_usage")),
        "product_name_raw": _normalize_text(record_row.get("product_name_raw")),
        "publisher_raw": _normalize_text(record_row.get("publisher_raw")),
        "publisher_presence": _normalize_text(record_row.get("publisher_presence")),
        "evidence_richness": _normalize_text(record_row.get("evidence_richness")),
        "usage_ambiguity": _normalize_text(record_row.get("usage_ambiguity")),
        "state_specific_risk": _normalize_text(record_row.get("state_specific_risk")),
        "placeholder_mapping": _normalize_text(record_row.get("placeholder_mapping")),
        "assessment_slice": _normalize_text(record_row.get("assessment_slice")),
        "confidence_band": _normalize_text(record_row.get("confidence_band")),
        "match_selected_strategy": _normalize_text(record_row.get("match_selected_strategy")),
    }
    candidate_payload = [
        {
            "rank": candidate.rank,
            "catalog_id": candidate.catalog_id,
            "product_name": candidate.product_name,
            "publisher": candidate.publisher,
            "grades": candidate.grades,
            "copyright_year": candidate.copyright_year,
            "product_type": candidate.product_type,
            "state_specific_version": candidate.state_specific_version,
            "district_created": candidate.district_created,
            "embedded_assessment": candidate.embedded_assessment,
        }
        for candidate in candidates
    ]
    return "\n".join(
        [
            "DISTRICT_ROW",
            json.dumps(row_context, indent=2, sort_keys=True),
            "",
            "SHORTLIST_CANDIDATES",
            json.dumps(candidate_payload, indent=2, sort_keys=True),
            "",
            "RESPONSE_JSON_SCHEMA",
            json.dumps(JSON_RESPONSE_INSTRUCTIONS, indent=2, sort_keys=True),
        ]
    )


def build_prompt_record(
    record_row: dict[str, Any],
    catalog_df: pd.DataFrame,
    *,
    topn_final: int = 3,
    catalog_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidates = build_candidate_shortlist(
        record_row,
        catalog_df,
        topn_final=topn_final,
        catalog_lookup=catalog_lookup,
    )
    return {
        "selection_identifier": _normalize_text(record_row.get("selection_identifier")),
        "expected_match_id": _normalize_text(record_row.get("expected_match_id")),
        "predicted_match_id": _normalize_text(record_row.get("predicted_match_id")),
        "top1_correct": bool(record_row.get("top1_correct")),
        "top3_correct": bool(record_row.get("top3_correct")),
        "candidate_ids": [candidate.catalog_id for candidate in candidates],
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": build_rerank_prompt(record_row, candidates),
    }


def build_catalog_lookup(catalog_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return _catalog_lookup(catalog_df)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_rerank_response(
    response: dict[str, Any],
    *,
    candidate_ids: list[str],
) -> dict[str, Any]:
    selected_candidate_id = _normalize_text(response.get("selected_candidate_id"))
    decision = _normalize_text(response.get("decision")) or "abstain"
    if decision not in {"select_candidate", "abstain"}:
        raise ValueError(f"Invalid decision: {decision}")
    if selected_candidate_id and selected_candidate_id not in candidate_ids:
        raise ValueError("selected_candidate_id is not in the provided shortlist")
    if decision == "select_candidate" and not selected_candidate_id:
        raise ValueError("selected_candidate_id is required for select_candidate")
    if decision == "abstain":
        selected_candidate_id = ""

    confidence = response.get("confidence", 0.0)
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    confidence_value = max(0.0, min(1.0, confidence_value))

    secondary_tags = response.get("secondary_tags", [])
    if isinstance(secondary_tags, str):
        secondary_tags = [tag for tag in secondary_tags.split("|") if tag]
    if not isinstance(secondary_tags, list):
        raise ValueError("secondary_tags must be a list or pipe-delimited string")

    return {
        "selected_candidate_id": selected_candidate_id,
        "decision": decision,
        "confidence": round(confidence_value, 4),
        "primary_reason": _normalize_text(response.get("primary_reason")),
        "secondary_tags": [str(tag).strip() for tag in secondary_tags if str(tag).strip()],
        "reasoning": _normalize_text(response.get("reasoning")),
    }


def validate_and_repair_rerank_response(
    response: dict[str, Any],
    *,
    candidate_ids: list[str],
    repair_similarity_cutoff: float = 0.95,
) -> dict[str, Any]:
    selected_candidate_id_original = _normalize_text(
        response.get("selected_candidate_id")
    )
    decision = _normalize_text(response.get("decision")) or "abstain"

    repaired_selected_candidate_id = False
    invalid_selected_candidate_id = False
    selected_candidate_id = selected_candidate_id_original

    if selected_candidate_id and selected_candidate_id not in candidate_ids:
        matches = get_close_matches(
            selected_candidate_id,
            candidate_ids,
            n=2,
            cutoff=repair_similarity_cutoff,
        )
        if len(matches) == 1:
            selected_candidate_id = matches[0]
            repaired_selected_candidate_id = True
        else:
            invalid_selected_candidate_id = True
            selected_candidate_id = ""
            decision = "abstain"

    validated = validate_rerank_response(
        {
            **response,
            "selected_candidate_id": selected_candidate_id,
            "decision": decision,
        },
        candidate_ids=candidate_ids,
    )
    validated["selected_candidate_id_original"] = selected_candidate_id_original
    validated["repaired_selected_candidate_id"] = repaired_selected_candidate_id
    validated["invalid_selected_candidate_id"] = invalid_selected_candidate_id
    return validated


def build_responses_api_body(
    prompt_record: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": prompt_record["system_prompt"]},
            {"role": "user", "content": prompt_record["user_prompt"]},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "curriculum_match_decision",
                "strict": True,
                "schema": RERANK_RESPONSE_SCHEMA,
            }
        },
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    return body
