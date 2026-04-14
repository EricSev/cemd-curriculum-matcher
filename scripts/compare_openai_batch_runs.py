from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two scored OpenAI Batch rerank summaries and write a recommendation."
    )
    parser.add_argument("--baseline-summary", required=True, help="Baseline scored-summary JSON.")
    parser.add_argument("--candidate-summary", required=True, help="Candidate scored-summary JSON.")
    parser.add_argument(
        "--baseline-pipeline",
        default=None,
        help="Optional baseline pipeline-summary JSON for model metadata.",
    )
    parser.add_argument(
        "--candidate-pipeline",
        default=None,
        help="Optional candidate pipeline-summary JSON for model metadata.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON comparison path.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional output Markdown comparison note path.",
    )
    return parser.parse_args()


def _load_json(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_delta(candidate: dict, baseline: dict, key: str) -> float:
    return round(float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0)), 4)


def _recommendation(candidate: dict, baseline: dict) -> dict[str, str]:
    delta_all = _metric_delta(
        candidate, baseline, "llm_top1_accuracy_on_all_rows"
    )
    delta_selected = _metric_delta(
        candidate, baseline, "llm_top1_accuracy_on_selected_rows"
    )
    delta_selection = _metric_delta(candidate, baseline, "llm_selection_rate")

    # Cost-aware default for larger-model promotion:
    # require clear lifts on both quality metrics and no selection-rate regression.
    should_promote = (
        delta_all >= 0.01 and delta_selected >= 0.01 and delta_selection >= 0.0
    )
    return {
        "decision": "promote_candidate" if should_promote else "keep_baseline",
        "rule": (
            "Promote only when top-1 on all rows and on selected rows each improve by "
            "at least 0.01, without a selection-rate regression."
        ),
        "reason": (
            "Candidate cleared the promotion thresholds."
            if should_promote
            else "Candidate did not clear the cost-aware promotion thresholds."
        ),
    }


def _build_markdown(
    baseline_summary: dict,
    candidate_summary: dict,
    baseline_pipeline: dict,
    candidate_pipeline: dict,
    comparison: dict,
) -> str:
    baseline_label = (
        baseline_pipeline.get("run_name")
        or Path(comparison["baseline_summary"]).stem.replace("-scored-summary", "")
        or baseline_pipeline.get("model")
        or "baseline"
    )
    candidate_label = (
        candidate_pipeline.get("run_name")
        or Path(comparison["candidate_summary"]).stem.replace("-scored-summary", "")
        or candidate_pipeline.get("model")
        or "candidate"
    )
    lines = [
        "# OpenAI Batch Rerank Comparison",
        "",
        f"- Baseline: `{baseline_label}`",
        f"- Candidate: `{candidate_label}`",
        f"- Decision: `{comparison['recommendation']['decision']}`",
        f"- Rule: {comparison['recommendation']['rule']}",
        f"- Reason: {comparison['recommendation']['reason']}",
        "",
        "## Metrics",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in (
        "row_count",
        "llm_selection_rate",
        "llm_top1_accuracy_on_all_rows",
        "llm_top1_accuracy_on_selected_rows",
        "repaired_selected_id_count",
        "invalid_selected_id_count",
    ):
        lines.append(
            "| {key} | {baseline} | {candidate} | {delta} |".format(
                key=key,
                baseline=baseline_summary.get(key, 0),
                candidate=candidate_summary.get(key, 0),
                delta=comparison["deltas"].get(key, 0),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    baseline_summary = _load_json(args.baseline_summary)
    candidate_summary = _load_json(args.candidate_summary)
    baseline_pipeline = _load_json(args.baseline_pipeline)
    candidate_pipeline = _load_json(args.candidate_pipeline)

    deltas = {
        key: _metric_delta(candidate_summary, baseline_summary, key)
        for key in (
            "llm_selection_rate",
            "llm_top1_accuracy_on_all_rows",
            "llm_top1_accuracy_on_selected_rows",
            "repaired_selected_id_count",
            "invalid_selected_id_count",
        )
    }
    deltas["row_count"] = int(candidate_summary.get("row_count", 0)) - int(
        baseline_summary.get("row_count", 0)
    )
    comparison = {
        "baseline_summary": args.baseline_summary,
        "candidate_summary": args.candidate_summary,
        "baseline_run_name": baseline_pipeline.get("run_name", ""),
        "candidate_run_name": candidate_pipeline.get("run_name", ""),
        "baseline_model": baseline_pipeline.get("model", ""),
        "candidate_model": candidate_pipeline.get("model", ""),
        "baseline_reasoning_effort": baseline_pipeline.get("reasoning_effort", ""),
        "candidate_reasoning_effort": candidate_pipeline.get("reasoning_effort", ""),
        "deltas": deltas,
        "recommendation": _recommendation(candidate_summary, baseline_summary),
    }

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(
            _build_markdown(
                baseline_summary,
                candidate_summary,
                baseline_pipeline,
                candidate_pipeline,
                comparison,
            ),
            encoding="utf-8",
        )

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
