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

from curriculum_matcher.llm_rerank import build_catalog_lookup, build_prompt_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LLM rerank evaluation prompts from saved matcher outputs."
    )
    parser.add_argument("--records-csv", required=True, help="Saved matcher evaluation records CSV.")
    parser.add_argument("--catalog-file", required=True, help="Catalog CSV file.")
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Prompt-pack JSONL output path.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional summary JSON path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for prompt generation.",
    )
    parser.add_argument(
        "--topn-final",
        type=int,
        default=3,
        help="How many saved candidates to include from pred_rank_* columns.",
    )
    parser.add_argument(
        "--only-errors",
        action="store_true",
        help="Only emit prompts for rows where top-1 is incorrect or no prediction was made.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records_df = pd.read_csv(args.records_csv)
    catalog_df = pd.read_csv(args.catalog_file, encoding="latin-1")
    catalog_lookup = build_catalog_lookup(catalog_df)

    if args.only_errors:
        records_df = records_df.loc[
            (~records_df["top1_correct"].fillna(False))
            | (records_df["predicted_match_id"].fillna("").astype(str).str.strip() == "")
        ].copy()
    if args.max_rows is not None:
        records_df = records_df.head(args.max_rows).copy()

    prompts = []
    for _, row in records_df.iterrows():
        prompt_record = build_prompt_record(
            row.to_dict(),
            catalog_df,
            topn_final=args.topn_final,
            catalog_lookup=catalog_lookup,
        )
        if not prompt_record["candidate_ids"]:
            continue
        prompts.append(prompt_record)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt, ensure_ascii=True) + "\n")

    summary = {
        "row_count": int(len(prompts)),
        "source_records_csv": args.records_csv,
        "catalog_file": args.catalog_file,
        "topn_final": args.topn_final,
        "only_errors": bool(args.only_errors),
        "max_rows": args.max_rows,
        "candidate_count_distribution": (
            pd.Series([len(prompt["candidate_ids"]) for prompt in prompts])
            .value_counts()
            .sort_index()
            .to_dict()
            if prompts
            else {}
        ),
    }

    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
