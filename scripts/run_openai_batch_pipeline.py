from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from curriculum_matcher.openai_batch import (
    create_batch,
    download_file_content,
    retrieve_batch,
    upload_batch_file,
)

from score_openai_batch_results import score_batch_results


TERMINAL_BATCH_STATUSES = {
    "completed",
    "failed",
    "expired",
    "cancelled",
    "cancelling",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload, create, poll, download, and score an OpenAI Batch rerank run."
    )
    parser.add_argument("--input-jsonl", required=True, help="Prepared batch request JSONL.")
    parser.add_argument("--prompt-jsonl", required=True, help="Prompt-pack JSONL used to build the requests.")
    parser.add_argument("--run-name", required=True, help="Short name used for downloaded/scored artifacts.")
    parser.add_argument(
        "--output-dir",
        default="benchmarks/outputs/openai_batch_runs",
        help="Directory for downloaded batch output and scored artifacts.",
    )
    parser.add_argument("--endpoint", default="/v1/responses")
    parser.add_argument("--completion-window", default="24h")
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=30,
        help="Polling interval while waiting for batch completion.",
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON string for batch metadata.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model name to store in the pipeline summary metadata.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Optional reasoning effort to store in the pipeline summary metadata.",
    )
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="Upload/create/poll/download only; skip result scoring.",
    )
    return parser.parse_args()


def _download_if_present(file_id: str | None, output_path: Path) -> dict | None:
    if not file_id:
        return None
    content = download_file_content(file_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)
    return {
        "file_id": file_id,
        "output_path": str(output_path),
        "byte_count": len(content),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(args.metadata_json) if args.metadata_json else {}
    metadata.setdefault("run_name", args.run_name)

    print(f"Uploading batch input file: {args.input_jsonl}", flush=True)
    uploaded = upload_batch_file(args.input_jsonl)
    print(f"Uploaded file id: {uploaded['id']}", flush=True)
    print(
        f"Creating batch for run '{args.run_name}' with poll interval {args.poll_interval_seconds}s",
        flush=True,
    )
    batch = create_batch(
        input_file_id=uploaded["id"],
        endpoint=args.endpoint,
        completion_window=args.completion_window,
        metadata=metadata,
    )
    batch_id = batch["id"]
    print(f"Batch id: {batch_id}", flush=True)

    poll_count = 0
    started_at = time.time()
    while batch.get("status") not in TERMINAL_BATCH_STATUSES:
        poll_count += 1
        request_counts = batch.get("request_counts") or {}
        elapsed = int(time.time() - started_at)
        status_line = f"Batch status poll #{poll_count} after {elapsed}s: {batch.get('status')}"
        if request_counts:
            status_line += (
                f" | total={request_counts.get('total', 0)}"
                f" completed={request_counts.get('completed', 0)}"
                f" failed={request_counts.get('failed', 0)}"
            )
        print(status_line, flush=True)
        time.sleep(args.poll_interval_seconds)
        batch = retrieve_batch(batch_id)
    print(f"Batch reached terminal status: {batch.get('status')}", flush=True)

    result: dict[str, object] = {
        "uploaded_file_id": uploaded["id"],
        "batch_id": batch_id,
        "batch_status": batch.get("status"),
        "input_jsonl": args.input_jsonl,
        "prompt_jsonl": args.prompt_jsonl,
        "run_name": args.run_name,
        "model": args.model or metadata.get("model", ""),
        "reasoning_effort": args.reasoning_effort or metadata.get("reasoning_effort", ""),
    }

    output_download = _download_if_present(
        batch.get("output_file_id"),
        output_dir / f"{args.run_name}-batch-output.jsonl",
    )
    error_download = _download_if_present(
        batch.get("error_file_id"),
        output_dir / f"{args.run_name}-batch-errors.jsonl",
    )
    if output_download:
        result["output_download"] = output_download
        print(f"Downloaded batch output to: {output_download['output_path']}", flush=True)
    if error_download:
        result["error_download"] = error_download
        print(f"Downloaded batch errors to: {error_download['output_path']}", flush=True)

    if batch.get("status") == "completed" and output_download and not args.skip_score:
        score_csv = output_dir / f"{args.run_name}-scored.csv"
        score_json = output_dir / f"{args.run_name}-scored-summary.json"
        print("Scoring batch output...", flush=True)
        score_summary = score_batch_results(
            prompt_jsonl=args.prompt_jsonl,
            batch_output_jsonl=output_download["output_path"],
            output_csv=score_csv,
            output_json=score_json,
        )
        result["score_summary"] = score_summary
        result["score_csv"] = str(score_csv)
        result["score_json"] = str(score_json)
        print(
            "Score summary: "
            f"rows={score_summary.get('row_count', 0)} "
            f"selection_rate={score_summary.get('llm_selection_rate', 0.0):.4f} "
            f"top1_all={score_summary.get('llm_top1_accuracy_on_all_rows', 0.0):.4f} "
            f"top1_selected={score_summary.get('llm_top1_accuracy_on_selected_rows', 0.0):.4f}",
            flush=True,
        )

    pipeline_summary_path = output_dir / f"{args.run_name}-pipeline-summary.json"
    pipeline_summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["pipeline_summary_json"] = str(pipeline_summary_path)
    print(f"Pipeline summary saved to: {pipeline_summary_path}", flush=True)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
