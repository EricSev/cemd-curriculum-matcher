# UI Batch Demo Handoff

- Date: 2026-04-04
- Version: v010
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`

## What changed

- Extended the active Tkinter app in `src/curriculum_matcher/app.py` instead of reviving archive UI code.
- Kept the existing matcher workflow intact and moved the new rerank operator flow into a separate `LLM Batch Rerank` tab.
- Added rerank controls for:
  - matcher records CSV
  - catalog CSV
  - artifact output directory
  - batch run output directory
  - run name
  - model
  - reasoning effort
  - shortlist size
  - poll interval
- Added buttons for:
  - build prompt pack
  - prepare Batch request JSONL
  - run full Batch pipeline
- Added shared status/progress/log updates for:
  - prompt-pack generation
  - request-file creation
  - upload
  - batch creation
  - each poll cycle
  - download
  - scoring
- Added end-of-run summary text in the UI with reviewed rows, selection rate, and top-1 metrics.
- Added a small CLI feedback improvement to `scripts/run_openai_batch_pipeline.py` so direct script runs are no longer silent during polling.

## Top-10 artifacts

- Confirmed the `top-10` matcher baseline already existed:
  - `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_records.csv`
  - `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_summary.json`
- Regenerated missing `top-10` rerank artifacts:
  - `benchmarks/outputs/historical_07122025_representative_1000_top10_llm_rerank_prompts.jsonl`
  - `benchmarks/outputs/historical_07122025_representative_1000_top10_llm_rerank_prompts_summary.json`
  - `benchmarks/outputs/historical_07122025_representative_1000_top10_llm_rerank_batch_requests.jsonl`

Prompt-pack summary after regeneration:

- reviewed rows: `472`
- shortlist size: `10`
- candidate count distribution:
  - `1`: `119`
  - `2`: `79`
  - `3`: `69`
  - `4`: `58`
  - `5`: `23`
  - `6`: `27`
  - `7`: `20`
  - `8`: `9`
  - `9`: `10`
  - `10`: `58`

## Validation

- `python3 -m py_compile src/curriculum_matcher/app.py scripts/run_openai_batch_pipeline.py`
- `PYTHONPATH=src python3 -m pytest tests/test_llm_rerank.py tests/test_qa.py tests/test_evaluation.py -q`

## Remaining gaps

- The new UI flow is optimized for the representative historical benchmark path and demo usability; it is not yet a polished general-purpose experiment manager.
- The full Batch action rebuilds local prompt/request artifacts before upload, which is intentional for operator safety, but there is not yet a separate “reuse existing artifacts without rebuild” toggle.
- The UI currently supports top-`3`, top-`10`, and top-`20`, but the intended/default shortlist is now top-`10`.
- No new matcher scoring logic was changed in this pass; shortlist recall remains the main model-quality bottleneck.
