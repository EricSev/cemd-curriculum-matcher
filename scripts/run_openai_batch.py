from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from curriculum_matcher.openai_batch import (
    cancel_batch,
    create_batch,
    download_file_content,
    retrieve_batch,
    upload_batch_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage OpenAI Batch API jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload", help="Upload a batch input JSONL file.")
    upload_parser.add_argument("--input-jsonl", required=True)

    create_parser = subparsers.add_parser("create", help="Create a batch from an uploaded file id.")
    create_parser.add_argument("--input-file-id", required=True)
    create_parser.add_argument("--endpoint", default="/v1/responses")
    create_parser.add_argument("--completion-window", default="24h")
    create_parser.add_argument("--metadata-json", default=None)

    status_parser = subparsers.add_parser("status", help="Retrieve batch status.")
    status_parser.add_argument("--batch-id", required=True)

    download_parser = subparsers.add_parser("download", help="Download a completed batch output or error file.")
    download_parser.add_argument("--file-id", required=True)
    download_parser.add_argument("--output-path", required=True)

    cancel_parser = subparsers.add_parser("cancel", help="Cancel an in-flight batch.")
    cancel_parser.add_argument("--batch-id", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "upload":
        result = upload_batch_file(args.input_jsonl)
    elif args.command == "create":
        metadata = json.loads(args.metadata_json) if args.metadata_json else None
        result = create_batch(
            input_file_id=args.input_file_id,
            endpoint=args.endpoint,
            completion_window=args.completion_window,
            metadata=metadata,
        )
    elif args.command == "status":
        result = retrieve_batch(args.batch_id)
    elif args.command == "cancel":
        result = cancel_batch(args.batch_id)
    elif args.command == "download":
        content = download_file_content(args.file_id)
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content)
        result = {
            "file_id": args.file_id,
            "output_path": str(output_path),
            "byte_count": len(content),
        }
    else:
        raise ValueError(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
