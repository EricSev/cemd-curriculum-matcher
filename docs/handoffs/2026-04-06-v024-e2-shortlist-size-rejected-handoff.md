# Curriculum Matcher Handoff v024

- Date: `2026-04-06`
- Milestone: `E2` shortlist-size retuning rejected
- Operator surface: keep using `src/curriculum_matcher/app.py`
- Workflow: benchmark-driven, one variable at a time

## Current Verified Baseline

Accepted matcher changes:

- `C1` safe publisher alias normalization
- `C2` safe product-title acronym expansion
- `C5` safe state-specific token normalization

Rejected matcher / retrieval / shortlist changes:

- `C3` grade normalization tightening
- `C4` edition / year title cleanup
- `D1` hybrid retrieval with RRF
- `D2` field-aware lexical weighting
- `E2` shortlist-size retuning

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

Open blocker:

- `E1` cross-encoder pre-LLM reranker remains blocked because no offline cached cross-encoder model is available in this environment

## What This Session Did

This session first rechecked the `E1` blocker, then ran the next pending experiment `E2` without changing matcher or retrieval logic.

Confirmed blocker:

- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python` still could not load `cross-encoder/ms-marco-MiniLM-L-6-v2`
- result remained an offline cache miss, so `E1` could not proceed

Measured experiment:

- increased shortlist size only: `10 -> 20`
- command run:
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python -m curriculum_matcher.evaluation_cli --benchmark-file benchmarks/gold/historical_07122025_representative_1000.csv --catalog-file "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv" --profile fast --topn-final 20 --output-json benchmarks/outputs/historical_07122025_representative_1000_fast_top20_char_ngram_summary.json --output-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top20_char_ngram_records.csv`

Artifacts:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_top20_char_ngram_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top20_char_ngram_records.csv`
- `docs/analysis/2026-04-06-shortlist-size-retuning-review.md`

## Decision

Decision: reject `E2` and keep shortlist `10`.

Why:

- top-1 accuracy stayed `0.4880`
- top-3 recall stayed `0.5740`
- shortlist hit@10 stayed `0.6280`
- nDCG@10 stayed `0.5596`
- MRR improved only `+0.0003`
- only `4` rows moved from miss to ranks `11-20`, which does not improve the accepted top-10 rerank operating range
- a larger shortlist would add prompt payload and rerank cost without measurable top-10 quality gain

## Tracker Update

Updated live tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`
- latest checkpoint now records the `E2` measurement and rejection
- `E2` now shows `rejected`
- `E1` remains `blocked`

## Validation

No source code changed.

Measured artifacts were regenerated successfully.

## Next Task

Do exactly one of these next:

- stage a specific offline cross-encoder model in the local Hugging Face cache, then resume `E1`
- if offline model staging is still not possible, keep `E1` blocked and decide whether to open a new benchmark-driven experiment outside the completed `E1`/`E2` pair

Recommended first choice:

- cache `cross-encoder/ms-marco-MiniLM-L-6-v2` or another explicitly approved offline cross-encoder, then run the cascade reranker experiment against the locked `D4` baseline

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-06-v024-e2-shortlist-size-rejected-handoff.md`.
>
> Current verified matcher baseline:
> - `C1` safe publisher alias normalization accepted
> - `C2` safe product-title acronym expansion accepted
> - `C5` safe state-specific token normalization accepted
> - `C3` grade normalization rejected
> - `C4` edition / year cleanup rejected
>
> Current verified retrieval and rerank baseline:
> - stage-1 recall `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
> - shortlist `10`
> - model `gpt-5.4-mini`
> - reasoning `medium`
>
> `E1` is still blocked because this offline environment does not have a cached cross-encoder model, and `E2` was measured and rejected because top-20 did not improve top-10 shortlist quality. Please:
> 1. read the latest handoff and the live tracker,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. do not reopen the accepted matcher or retrieval baselines unless new measured evidence requires it,
> 4. if a cross-encoder model has now been staged offline, lift `E1` and run exactly that cascade experiment,
> 5. if a cross-encoder model is still unavailable offline, do not substitute another experiment silently; record the blocker and choose the next benchmark-driven task explicitly,
> 6. keep the workflow benchmark-driven and one-variable-at-a-time,
> 7. preserve the accepted matcher and rerank baselines unless a measured experiment clearly replaces them,
> 8. use `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for benchmark work in this environment,
> 9. leave a tracker update, a review note if a new experiment is measured, and a fresh handoff before ending.
