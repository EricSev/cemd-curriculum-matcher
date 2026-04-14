from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from curriculum_matcher.llm_rerank import build_responses_api_body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LLM rerank prompt-pack JSONL into OpenAI Batch API request JSONL."
    )
    parser.add_argument("--prompt-jsonl", required=True, help="Prompt-pack JSONL input.")
    parser.add_argument("--output-jsonl", required=True, help="Batch request JSONL output.")
    parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model to use.")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=[None, "low", "medium", "high"],
        help="Optional reasoning effort for supported models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_path = Path(args.prompt_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with prompt_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            prompt_record = json.loads(line)
            request_body = build_responses_api_body(
                prompt_record,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
            )
            batch_line = {
                "custom_id": prompt_record["selection_identifier"],
                "method": "POST",
                "url": "/v1/responses",
                "body": request_body,
            }
            dst.write(json.dumps(batch_line, ensure_ascii=True) + "\n")
            count += 1

    print(
        json.dumps(
            {
                "row_count": count,
                "prompt_jsonl": str(prompt_path),
                "output_jsonl": str(output_path),
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
