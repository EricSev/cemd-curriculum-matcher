# Medium Beats Low Handoff

- Date: 2026-04-05
- Version: v012
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`
- Focus: record that `top-10 + gpt-5.4-mini + medium` beat the prior `low` baseline and recommend the next controlled experiment

## What happened

- Re-ran the `top-10` Batch rerank experiment using:
  - model: `gpt-5.4-mini`
  - reasoning effort: `medium`
  - shortlist size: `10`
- Confirmed the new run completed successfully from the UI with clean dedicated filenames:
  - `historical-top10-gpt54mini-medium-batch-*`
- Hardened scoring logic so one-character `selected_candidate_id` typos no longer fail the whole scoring run.
- Updated the UI so the suggested run name includes reasoning effort, which prevents accidental overwrites when switching from `low` to `medium`.

## Relevant artifacts

Current low baseline:

- `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-batch-scored-summary.json`

Clean medium rerun:

- `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-batch-batch-output.jsonl`
- `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-batch-scored.csv`
- `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-batch-scored-summary.json`
- `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-batch-pipeline-summary.json`

## Comparison: medium vs low

Low baseline summary:

- reviewed rows: `472`
- selection rate: `0.4640`
- top-1 accuracy on all reviewed rows: `0.1737`
- top-1 accuracy on selected rows: `0.3744`

Medium summary:

- reviewed rows: `472`
- selection rate: `0.4788`
- top-1 accuracy on all reviewed rows: `0.1843`
- top-1 accuracy on selected rows: `0.3850`
- repaired selected id count: `1`
- invalid selected id count: `0`

Deltas for medium vs low:

- selection rate: `+0.0148`
- top-1 on all reviewed rows: `+0.0106`
- top-1 on selected rows: `+0.0106`

## Interpretation

This is a positive result.

- `medium` beat `low` on all three primary rerank metrics.
- The gains are modest rather than dramatic, but directionally clean.
- The current best rerank benchmark setting is now:
  - shortlist size: `10`
  - model: `gpt-5.4-mini`
  - reasoning effort: `medium`

## Scoring robustness changes

Implemented in:

- `src/curriculum_matcher/llm_rerank.py`
- `scripts/score_openai_batch_results.py`
- `src/curriculum_matcher/app.py`

New scoring behavior:

- exact shortlist ID: accepted
- unique near-match shortlist ID: auto-repaired and flagged
- non-repairable invalid shortlist ID: converted to abstain and counted instead of crashing scoring

New scored output columns:

- `llm_selected_candidate_id_original`
- `llm_selected_candidate_id_repaired`
- `llm_selected_candidate_id_invalid`

New summary fields:

- `repaired_selected_id_count`
- `invalid_selected_id_count`

## UI / workflow changes relevant to future experiments

Updated in:

- `src/curriculum_matcher/app.py`

Relevant changes:

- suggested run name now updates when reasoning effort changes
- pipeline summary now records:
  - `run_name`
  - `model`
  - `reasoning_effort`
  - `shortlist_size`

This should prevent accidental file collisions between `low` and `medium` runs going forward.

## Recommendation for next experiment

Do not change the core matcher scoring logic yet.

Recommended next controlled experiment:

1. Keep shortlist size fixed at `10`.
2. Keep reasoning effort fixed at `medium`.
3. Change only the model size.

Best candidate next run:

- compare `gpt-5.4-mini` vs a larger model such as `gpt-5.4`

Suggested run name:

- `historical-top10-gpt54-medium-batch`

Why this is the best next step:

- shortlist size has already been improved
- `medium` has already beaten `low`
- the clean next question is whether a larger model delivers enough additional gain to justify the added cost

## Validation performed

- `python3 -m py_compile src/curriculum_matcher/app.py src/curriculum_matcher/llm_rerank.py scripts/score_openai_batch_results.py`
- `PYTHONPATH=src python3 -m pytest tests/test_llm_rerank.py tests/test_qa.py tests/test_evaluation.py -q`
