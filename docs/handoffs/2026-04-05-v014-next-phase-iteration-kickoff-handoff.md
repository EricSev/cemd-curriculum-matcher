# Next-Phase Iteration Kickoff Handoff

- Date: 2026-04-05
- Version: v014
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`
- Focus: implement the first rerank-first next-phase artifacts and lock the current rerank baseline decision

## What was completed

- Added shortlist-quality instrumentation to matcher evaluation outputs:
  - `hit_rate_at_1`
  - `hit_rate_at_3`
  - `hit_rate_at_10`
  - `mrr`
  - `ndcg_at_10`
- Extended confidence-band, slice, and cross-slice summaries to carry those rank metrics.
- Corrected `top3_correct` handling so top-3 and top-10 are no longer conflated when evaluation runs at `topn_final=10`.
- Regenerated the representative top-10 benchmark outputs with the richer metric contract.
- Added a shortlist bottleneck report script and generated the first snapshot report.
- Ran the controlled `gpt-5.4` vs `gpt-5.4-mini` rerank comparison at:
  - shortlist size: `10`
  - reasoning effort: `medium`
- Added a scored-summary comparison script and generated a formal keep/promote decision artifact.
- Added a next-phase iteration roadmap doc so the prioritized test queue now lives in the repo.

## Current rerank baseline decision

Baseline kept:

- shortlist size: `10`
- model: `gpt-5.4-mini`
- reasoning effort: `medium`

Reason:

- `gpt-5.4` improved accuracy on selected rows but regressed both overall selection rate and top-1 on all reviewed rows, so it did not clear the cost-aware promotion rule.

Comparison:

- baseline `historical-top10-gpt54mini-medium-batch`
  - selection rate: `0.4788`
  - top-1 on all reviewed rows: `0.1843`
  - top-1 on selected rows: `0.3850`
- candidate `historical-top10-gpt54-medium-batch`
  - selection rate: `0.3093`
  - top-1 on all reviewed rows: `0.1716`
  - top-1 on selected rows: `0.5548`

Decision artifact:

- `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54-vs-gpt54mini-medium-comparison.json`
- `docs/analysis/2026-04-05-gpt54-vs-gpt54mini-medium-comparison.md`

## New representative benchmark shortlist metrics

Representative benchmark:

- `benchmarks/gold/historical_07122025_representative_1000.csv`

Updated top-10 baseline summary:

- hit@1: `0.4600`
- hit@3: `0.5610`
- hit@10: `0.5990`
- MRR: `0.5126`
- nDCG@10: `0.5341`

Most important bottleneck slices from the first snapshot:

- `Assessment`
- `catalog_unspecified`
- adoption-state risk rows
- assessment state-specific expected rows

Snapshot artifact:

- `docs/analysis/2026-04-05-shortlist-bottleneck-snapshot.md`

## Files added or updated

- `src/curriculum_matcher/evaluation.py`
- `tests/test_evaluation.py`
- `scripts/compare_openai_batch_runs.py`
- `scripts/report_shortlist_bottlenecks.py`
- `scripts/run_openai_batch_pipeline.py`
- `docs/roadmap/2026-04-05-next-phase-build-test-iterations.md`

## What should be done next

Keep the next iterations one-variable-at-a-time.

Priority order:

1. test publisher alias normalization
2. test acronym / abbreviation expansion
3. test grade normalization tightening
4. test edition / year extraction improvements
5. then test retrieval-only upgrades:
   - RRF fusion
   - field-aware lexical weighting
   - character n-gram retrieval

Discipline:

- compare each experiment against the locked `gpt-5.4-mini + medium + top-10` rerank baseline
- preserve the same representative benchmark and output naming discipline
- do not change core matcher scoring logic yet

## Validation performed

- `python3 -m py_compile src/curriculum_matcher/evaluation.py src/curriculum_matcher/evaluation_cli.py scripts/compare_openai_batch_runs.py scripts/report_shortlist_bottlenecks.py scripts/run_openai_batch_pipeline.py`
- `PYTHONPATH=src python3 -m pytest tests/test_evaluation.py tests/test_llm_rerank.py -q`
- `PYTHONPATH=src ./.venv/bin/python -m curriculum_matcher.evaluation_cli --benchmark-file benchmarks/gold/historical_07122025_representative_1000.csv --catalog-file "...CEMD Product Catalog - 07122025.csv" --profile fast --topn-final 10 --output-json benchmarks/outputs/historical_07122025_representative_1000_fast_top10_summary.json --output-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top10_records.csv`
- `PYTHONPATH=src ./.venv/bin/python scripts/run_openai_batch_pipeline.py --input-jsonl benchmarks/outputs/openai_batch_runs/historical-top10-gpt54-medium-batch-requests.jsonl --prompt-jsonl benchmarks/outputs/historical_07122025_representative_1000_fast_top10_llm_rerank_prompts.jsonl --run-name historical-top10-gpt54-medium-batch --output-dir benchmarks/outputs/openai_batch_runs --poll-interval-seconds 30`
