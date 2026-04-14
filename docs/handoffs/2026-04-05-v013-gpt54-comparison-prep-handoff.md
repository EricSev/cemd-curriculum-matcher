# GPT-5.4 Comparison Prep Handoff

- Date: 2026-04-05
- Version: v013
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`
- Focus: prepare the operator workflow for the next controlled `top-10` rerank comparison without changing matcher scoring logic

## Current validated benchmark baseline

- Best validated rerank setting remains:
  - shortlist size: `10`
  - model: `gpt-5.4-mini`
  - reasoning effort: `medium`
- Validated comparison:
  - low baseline:
    - reviewed rows: `472`
    - selection rate: `0.4640`
    - top-1 on all reviewed rows: `0.1737`
    - top-1 on selected rows: `0.3744`
  - medium rerun:
    - reviewed rows: `472`
    - selection rate: `0.4788`
    - top-1 on all reviewed rows: `0.1843`
    - top-1 on selected rows: `0.3850`

## Intended next experiment

Keep the experiment one-variable-at-a-time:

1. keep shortlist size fixed at `10`
2. keep reasoning effort fixed at `medium`
3. compare `gpt-5.4-mini` vs `gpt-5.4`

Suggested next run name:

- `historical-top10-gpt54-medium-batch`

Comparison target artifact:

- `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-batch-scored-summary.json`

## Small safeguards added

Updated in:

- `src/curriculum_matcher/app.py`

Changes:

- rerank UI defaults now align with the current best validated baseline:
  - model: `gpt-5.4-mini`
  - reasoning effort: `medium`
  - suggested run name default: `historical-top10-gpt54mini-medium-batch`
- run name is refreshed after settings load so historical auto-generated names stay aligned with the selected model / reasoning tags
- full Batch pipeline now refuses to start if the chosen run name would overwrite existing downloaded/scored artifacts
- scoring now derives the baseline shortlist key from the prompt pack row’s actual candidate count instead of the live UI shortlist control, which avoids mismatches if the operator changes UI settings mid-run

## Current UI / scoring readiness

Ready for the next controlled run:

- active operator surface remains `src/curriculum_matcher/app.py`
- pipeline summary still records:
  - `run_name`
  - `model`
  - `reasoning_effort`
  - `shortlist_size`
- scoring still includes repaired / invalid selected-id counts
- no core matcher scoring logic was changed in this pass

## What should be run next

From the Tkinter UI:

1. open the `LLM Batch Rerank` tab
2. keep shortlist size at `10`
3. keep reasoning effort at `medium`
4. switch model to `gpt-5.4`
5. use run name `historical-top10-gpt54-medium-batch`
6. run the full Batch pipeline
7. compare the resulting scored summary against:
   - `historical-top10-gpt54mini-medium-batch-scored-summary.json`

## Validation performed

- `python3 -m py_compile src/curriculum_matcher/app.py src/curriculum_matcher/llm_rerank.py scripts/score_openai_batch_results.py scripts/run_openai_batch_pipeline.py`
- `PYTHONPATH=src python3 -m pytest tests/test_llm_rerank.py -q`
