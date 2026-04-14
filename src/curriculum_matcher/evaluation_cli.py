import argparse
import json

from .evaluation import evaluate_matcher_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the curriculum matcher against a benchmark dataset."
    )
    parser.add_argument("--benchmark-file", required=True, help="CSV benchmark file.")
    parser.add_argument("--catalog-file", required=True, help="Catalog CSV file.")
    parser.add_argument(
        "--expected-id-column",
        default=None,
        help="Optional expected id column. Defaults to expected_product_identifier or product_identifier if present.",
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "accurate"],
        default="accurate",
        help="Matcher profile to use during evaluation.",
    )
    parser.add_argument(
        "--retrieval-experiment",
        default=None,
        help="Optional retrieval experiment override.",
    )
    parser.add_argument(
        "--rerank-experiment",
        default=None,
        help="Optional rerank experiment override, for example `cross_encoder`.",
    )
    parser.add_argument(
        "--cross-encoder-model",
        default=None,
        help="Optional cross-encoder model override when rerank experiment is enabled.",
    )
    parser.add_argument(
        "--topn-final",
        type=int,
        default=3,
        help="How many final recommendations to evaluate.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for summary JSON output.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path for per-record evaluation CSV output.",
    )
    args = parser.parse_args()

    artifacts = evaluate_matcher_run(
        args.benchmark_file,
        args.catalog_file,
        profile=args.profile,
        retrieval_experiment=args.retrieval_experiment,
        rerank_experiment=args.rerank_experiment,
        cross_encoder_model=args.cross_encoder_model,
        expected_id_column=args.expected_id_column,
        topn_final=args.topn_final,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(json.dumps(artifacts.summary, indent=2))


if __name__ == "__main__":
    main()
