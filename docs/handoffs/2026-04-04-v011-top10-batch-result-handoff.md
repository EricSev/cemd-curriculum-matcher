# Top-10 Batch Result Handoff

- Date: 2026-04-04
- Version: v011
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`
- Focus: record the first validated `top-10` OpenAI Batch rerank result and recommend the next controlled experiment

## What was completed

- Ran a live `top-10` OpenAI Batch rerank from the updated Tkinter UI.
- Saved the batch output, scored CSV, scored summary JSON, and pipeline summary JSON.
- Fixed a UI-side scoring-summary rounding bug in `src/curriculum_matcher/app.py` that incorrectly wrote `llm_top1_accuracy_on_all_rows` as `0` in the new summary JSON.
- Regenerated the saved `top-10` summary JSON from the scored CSV so the artifact now matches the actual result.

## Relevant output artifacts

Top-10 run outputs:

- `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-batch-batch-output.jsonl`
- `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-batch-scored.csv`
- `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-batch-scored-summary.json`
- `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-batch-pipeline-summary.json`

Earlier top-3 comparison artifact:

- `benchmarks/outputs/openai_batch_runs/historical-1000-gpt54mini-batch-scored-summary.json`

## Corrected top-10 summary

Validated from the scored CSV:

- reviewed rows: `472`
- LLM selection rate: `0.4640`
- LLM top-1 accuracy on all reviewed rows: `0.1737`
- LLM top-1 accuracy on selected rows: `0.3744`

Primary reason counts:

- `likely_assessment_alias_or_subtype_ambiguity`: `82`
- `likely_evidence_sparsity`: `139`
- `likely_matcher_error`: `41`
- `likely_ui_catalog_selection_ambiguity`: `72`
- `well_supported`: `138`

## Comparison vs earlier top-3 run

Earlier top-3 run:

- reviewed rows: `472`
- LLM selection rate: `0.4216`
- LLM top-1 accuracy on all reviewed rows: `0.1398`
- LLM top-1 accuracy on selected rows: `0.3317`

Top-10 deltas vs top-3:

- selection rate: `+0.0424`
- top-1 on all reviewed rows: `+0.0339`
- top-1 on selected rows: `+0.0427`

## Interpretation

This is a positive result.

- Moving from `top-3` to `top-10` improved all key rerank metrics.
- The earlier bottleneck hypothesis was supported: shortlist size really was limiting the rerank layer.
- This result strengthens the case for treating `top-10` as the new rerank baseline for follow-on experiments.

## Code note

Bug fixed in:

- `src/curriculum_matcher/app.py`

Specific issue:

- `_score_batch_output()` used `round(float(df["llm_top1_correct"].mean()))` instead of `round(..., 4)`, which collapsed the all-rows metric to `0` in the saved summary JSON.

## Recommendation for next session

Do not change core matcher scoring logic yet.

Recommended next step:

1. Treat `top-10 + gpt-5.4-mini + low reasoning` as the new rerank benchmark baseline.
2. Run one controlled rerank experiment changing only one variable.

Best candidate follow-on experiments:

- keep `top-10` fixed and compare `gpt-5.4-mini` vs a larger model
- or keep model fixed and compare `low` vs `medium` reasoning effort

Preferred discipline:

- change only one variable at a time
- keep the historical representative benchmark and scoring path unchanged
- compare against the saved `historical-top10-gpt54mini-batch-scored-summary.json`

## Validation performed

- `python3 -m py_compile src/curriculum_matcher/app.py`
- `PYTHONPATH=src python3 -m pytest tests/test_llm_rerank.py tests/test_qa.py tests/test_evaluation.py -q`
