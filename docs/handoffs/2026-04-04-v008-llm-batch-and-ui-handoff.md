# LLM Batch And UI Handoff

- Date: 2026-04-04
- Version: v008
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`
- Next-session focus: build a usable UI flow for the new OpenAI Batch rerank workflow and improve operator feedback for long-running jobs

## What changed this session

- Added policy-aware benchmark slicing to the evaluation pipeline.
- Backfilled the representative historical benchmark outputs with new slice columns and slice metrics.
- Added a revised QA reason taxonomy with primary reasons and secondary tags.
- Built a representative historical QA benchmark artifact using the new policy-aware slices.
- Added an LLM rerank prompt-pack generator for error-focused benchmark rows.
- Added an OpenAI Batch API workflow for low-cost rerank experiments.
- Added a one-command Batch pipeline runner.
- Added repo-root `.env` loading for OpenAI Batch scripts so persistent API-key setup can use a gitignored `.env`.

## Key new analysis artifact

- benchmark and QA recommendation memo:
  - `docs/analysis/2026-04-04-benchmark-slicing-and-qa-recommendations.md`

## Benchmark and QA changes now in code

### Evaluation slices

Implemented in:

- `src/curriculum_matcher/evaluation.py`

New per-row derived slices:

- `evidence_richness`
- `usage_ambiguity`
- `state_specific_risk`
- `placeholder_mapping`
- `assessment_slice`
- `assessment_short_or_acronym`
- `assessment_publisher_missing`
- `assessment_state_specific_expected`

New summary sections:

- `slice_metrics` for all above slices
- `cross_slice_metrics` for:
  - `product_type_usage x evidence_richness`
  - `product_type_usage x placeholder_mapping`
  - `product_type_usage x state_specific_risk`
  - `product_type_usage x assessment_slice`

Representative benchmark outputs updated in place:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_records.csv`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_summary.json`

Notable current metric:

- `catalog_unspecified` placeholder rows: `56` rows, top-1 `0.0893`

### QA taxonomy

Implemented in:

- `src/curriculum_matcher/qa.py`
- `src/curriculum_matcher/app.py`

New QA primary reason field:

- `human_match_challenge_primary_reason`

Primary reason buckets:

- `likely_matcher_error`
- `likely_evidence_sparsity`
- `likely_ui_catalog_selection_ambiguity`
- `likely_assessment_alias_or_subtype_ambiguity`
- `well_supported`

Also added:

- `human_match_challenge_secondary_tags`

### QA benchmark builder

Updated:

- `scripts/build_human_qa_benchmark.py`
- `scripts/evaluate_human_match_qa.py`

New representative QA benchmark artifacts:

- `benchmarks/qa/historical_07122025_representative_1000_human_qa.csv`
- `benchmarks/qa/historical_07122025_representative_1000_human_qa_summary.json`

Representative QA benchmark slice counts:

- `assessment`: `594`
- `sparse_evidence`: `514`
- `state_specific_risk`: `443`
- `catalog_unspecified`: `89`

Important optimization:

- `scripts/evaluate_human_match_qa.py` now caches matcher results per source row
- this matters for `3K-10K` row workflows because QA rows can duplicate the same underlying source row many times

## LLM rerank / Batch workflow added

### New core modules

- `src/curriculum_matcher/llm_rerank.py`
- `src/curriculum_matcher/openai_batch.py`

### New scripts

- prompt-pack generator:
  - `scripts/evaluate_llm_rerank.py`
- convert prompt-pack to Batch request JSONL:
  - `scripts/prepare_openai_batch_requests.py`
- upload/create/status/download batch jobs:
  - `scripts/run_openai_batch.py`
- score downloaded batch results:
  - `scripts/score_openai_batch_results.py`
- convenience runner:
  - `scripts/run_openai_batch_pipeline.py`

### Generated artifacts

Prompt pack:

- `benchmarks/outputs/historical_07122025_representative_1000_llm_rerank_prompts.jsonl`
- `benchmarks/outputs/historical_07122025_representative_1000_llm_rerank_prompts_summary.json`

Current prompt-pack summary:

- row count: `472`
- candidate count distribution:
  - `1` candidate: `119`
  - `2` candidates: `79`
  - `3` candidates: `274`

Prepared Batch request JSONL:

- `benchmarks/outputs/historical_07122025_representative_1000_llm_rerank_batch_requests.jsonl`

Current default model in prepared batch file:

- `gpt-5.4-mini`
- reasoning effort: `low`

## OpenAI API-key setup

Current repo state:

- `.env` is already ignored in `.gitignore`
- added `.env.example`
- Batch scripts now auto-load repo-root `.env`

Relevant files:

- `.gitignore`
- `.env.example`
- `src/curriculum_matcher/openai_batch.py`

## Important operational note

The Batch pipeline works, but current CLI feedback is sparse.

Observed operator issue:

- when running `scripts/run_openai_batch_pipeline.py`, the terminal can appear idle because the script mostly polls in silence and prints structured results only at the end

This is the main reason next session should prioritize UI/feedback work.

## UI findings

The current active app already contains a usable Tkinter foundation with:

- input file picker
- catalog file picker
- output directory picker
- progress bar
- status label
- log pane
- processing button

Active UI location:

- `src/curriculum_matcher/app.py`

Relevant archived UI references:

- `archive/legacy_gui/gui_curriculum_matcher.py`
- `archive/legacy_gui/gui_curriculum_matcher2.py`
- `archive/legacy_core/enhanced_curriculum_matcher.py`

Conclusion:

- there is no need to resurrect an old UI from scratch
- the better path is to extend the current active UI so it can also support:
  - prompt-pack generation
  - Batch request preparation
  - Batch submission and polling
  - result download and scoring
  - clear status/log feedback for demos

## Recommended next-session UI scope

Build a first-pass demo/operator UI around the Batch rerank path.

### Minimum viable next-session UI features

1. Add a separate “LLM Batch Rerank” workflow area in the active Tkinter app.
2. Allow selecting:
   - saved matcher records CSV
   - catalog CSV
   - prompt-pack output path or output directory
   - batch run name
   - model choice
3. Add buttons for:
   - build prompt pack
   - build batch request file
   - run batch pipeline
4. Stream progress/status messages into the log pane:
   - prompt-pack row count
   - upload success
   - batch id
   - current batch status on each poll
   - output download paths
   - scored summary metrics
5. Keep the existing matcher UI intact rather than replacing it.

### Good second-pass features

- separate tab or frame for:
  - baseline matcher
  - benchmark evaluation
  - LLM rerank batch
- progress estimates for polling
- open-output-folder button
- visible summary card for:
  - batch status
  - selection rate
  - LLM top-1 on reviewed rows
  - abstain rate

## Constraints for next session

- do not throw away the current active UI
- preserve current matcher workflow while adding a second workflow
- keep the first UI enhancement practical and demoable rather than overly polished
- prefer live log/status clarity over visual complexity
- continue optimizing for cost-conscious Batch usage rather than synchronous API calls

## Validation status from this session

Passing checks:

- `PYTHONPATH=src python3 -m pytest tests/test_llm_rerank.py tests/test_qa.py tests/test_evaluation.py -q`
  - `20 passed`

- syntax check passed for:
  - `scripts/run_openai_batch_pipeline.py`
  - `scripts/run_openai_batch.py`
  - `scripts/prepare_openai_batch_requests.py`
  - `scripts/score_openai_batch_results.py`
  - `src/curriculum_matcher/openai_batch.py`

## Agent-optimized kickoff prompt for next session

Resume work on the curriculum matcher from:

- `docs/handoffs/2026-04-04-v008-llm-batch-and-ui-handoff.md`
- `docs/analysis/2026-04-04-benchmark-slicing-and-qa-recommendations.md`

Current priority:

- extend the active Tkinter app in `src/curriculum_matcher/app.py` so the new OpenAI Batch rerank workflow is usable and demo-friendly

Please:

1. review the current active UI in `src/curriculum_matcher/app.py`,
2. preserve the existing matcher workflow,
3. add a second UI flow for:
   - building the prompt pack
   - preparing the batch request JSONL
   - running the full batch pipeline
   - showing live progress/status in the log pane,
4. optimize for operator clarity and demo value rather than visual polish,
5. keep using Batch API semantics and repo-root `.env` loading,
6. do not start redesigning matcher logic yet.
