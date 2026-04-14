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

from curriculum_matcher.llm_rerank import (
    extract_json_object,
    validate_and_repair_rerank_response,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score OpenAI Batch API rerank results against prompt-pack expectations."
    )
    parser.add_argument("--prompt-jsonl", required=True, help="Prompt-pack JSONL input.")
    parser.add_argument("--batch-output-jsonl", required=True, help="Downloaded batch output JSONL.")
    parser.add_argument("--output-csv", required=True, help="Scored results CSV.")
    parser.add_argument("--output-json", default=None, help="Optional summary JSON.")
    return parser.parse_args()


def _load_prompt_lookup(path: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            lookup[row["selection_identifier"]] = row
    return lookup


def _extract_output_text(response_body: dict) -> str:
    output = response_body.get("output", [])
    texts: list[str] = []
    for item in output:
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def score_batch_results(
    *,
    prompt_jsonl: str | Path,
    batch_output_jsonl: str | Path,
    output_csv: str | Path,
    output_json: str | Path | None = None,
) -> dict:
    prompt_lookup = _load_prompt_lookup(Path(prompt_jsonl))

    rows = []
    repaired_selected_id_count = 0
    invalid_selected_id_count = 0
    with Path(batch_output_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            batch_row = json.loads(line)
            selection_identifier = batch_row.get("custom_id", "")
            prompt_row = prompt_lookup.get(selection_identifier)
            if not prompt_row:
                continue

            response_body = batch_row.get("response", {}).get("body", {})
            response_text = _extract_output_text(response_body)
            parsed = extract_json_object(response_text)
            validated = validate_and_repair_rerank_response(
                parsed,
                candidate_ids=prompt_row["candidate_ids"],
            )
            selected_candidate_id = validated["selected_candidate_id"]
            expected_match_id = prompt_row.get("expected_match_id", "")
            repaired_selected_id_count += int(
                validated["repaired_selected_candidate_id"]
            )
            invalid_selected_id_count += int(validated["invalid_selected_candidate_id"])

            rows.append(
                {
                    "selection_identifier": selection_identifier,
                    "expected_match_id": expected_match_id,
                    "baseline_predicted_match_id": prompt_row.get("predicted_match_id", ""),
                    "llm_selected_candidate_id": selected_candidate_id,
                    "llm_selected_candidate_id_original": validated[
                        "selected_candidate_id_original"
                    ],
                    "llm_selected_candidate_id_repaired": validated[
                        "repaired_selected_candidate_id"
                    ],
                    "llm_selected_candidate_id_invalid": validated[
                        "invalid_selected_candidate_id"
                    ],
                    "llm_decision": validated["decision"],
                    "llm_confidence": validated["confidence"],
                    "llm_primary_reason": validated["primary_reason"],
                    "llm_secondary_tags": "|".join(validated["secondary_tags"]),
                    "llm_reasoning": validated["reasoning"],
                    "baseline_top1_correct": bool(prompt_row.get("top1_correct")),
                    "baseline_top3_correct": bool(prompt_row.get("top3_correct")),
                    "llm_top1_correct": bool(
                        selected_candidate_id and expected_match_id and selected_candidate_id == expected_match_id
                    ),
                    "llm_abstained": validated["decision"] == "abstain",
                }
            )

    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    summary = {
        "row_count": int(len(df)),
        "llm_selection_rate": round(float((~df["llm_abstained"]).mean()), 4) if not df.empty else 0.0,
        "llm_top1_accuracy_on_all_rows": round(float(df["llm_top1_correct"].mean()), 4) if not df.empty else 0.0,
        "llm_top1_accuracy_on_selected_rows": (
            round(float(df.loc[~df["llm_abstained"], "llm_top1_correct"].mean()), 4)
            if not df.empty and (~df["llm_abstained"]).any()
            else 0.0
        ),
        "primary_reason_counts": (
            df["llm_primary_reason"].fillna("missing").value_counts().sort_index().to_dict()
            if not df.empty
            else {}
        ),
        "repaired_selected_id_count": int(repaired_selected_id_count),
        "invalid_selected_id_count": int(invalid_selected_id_count),
    }

    if output_json:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    args = parse_args()
    summary = score_batch_results(
        prompt_jsonl=args.prompt_jsonl,
        batch_output_jsonl=args.batch_output_jsonl,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
