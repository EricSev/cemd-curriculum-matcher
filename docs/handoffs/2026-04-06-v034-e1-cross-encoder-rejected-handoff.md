# Curriculum Matcher Handoff v034

- Date: `2026-04-06`
- Milestone: `E1` cross-encoder cascade benchmark measured and rejected
- Operator surface: keep using `src/curriculum_matcher/app.py`
- Workflow: benchmark-driven, one variable at a time

## Current Verified Baseline

Accepted matcher changes:

- `C1` safe publisher alias normalization
- `C2` safe product-title acronym expansion
- `C5` safe state-specific token normalization

Rejected matcher / retrieval / rerank / shortlist changes:

- `C3` grade normalization tightening
- `C4` edition / year title cleanup
- `D1` hybrid retrieval with RRF
- `D2` field-aware lexical weighting
- `E1` cross-encoder pre-LLM reranker
- `E2` shortlist-size retuning

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## What This Session Did

- implemented an opt-in cross-encoder rerank path behind benchmark-only configuration so the default matcher behavior stays unchanged
- added CLI support to run evaluation with an explicit rerank experiment and cross-encoder model
- validated the new path with targeted unit tests
- ran the representative benchmark with:
  - profile `fast`
  - shortlist `10`
  - rerank experiment `cross_encoder`
  - model `cross-encoder/ms-marco-MiniLM-L-6-v2`

Benchmark artifacts:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_cross_encoder_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_cross_encoder_records.csv`

Review note:

- `docs/analysis/2026-04-06-cross-encoder-rerank-review.md`

## Decision

Decision: reject `E1`.

Why:

- overall results were mixed
- top-3 recall improved, but top-1 accuracy regressed `-0.0020`
- shortlist hit@10 also regressed `-0.0030`
- roadmap default is to reject mixed results rather than weaken the locked baseline

## Tracker Update

Updated live tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

Current roadmap state after this run:

- `E1` rejected
- `E2` rejected
- no active experiment remains in this roadmap

## Validation

Validated:

- `PYTHONPATH=src ./.venv/bin/python -m unittest tests.test_matcher_core tests.test_evaluation`

Benchmark command run:

- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python -m curriculum_matcher.evaluation_cli --benchmark-file benchmarks/gold/historical_07122025_representative_1000.csv --catalog-file "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv" --profile fast --topn-final 10 --rerank-experiment cross_encoder --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2 --output-json benchmarks/outputs/historical_07122025_representative_1000_fast_top10_cross_encoder_summary.json --output-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top10_cross_encoder_records.csv`

## Next Task

Do this next:

- define a new next-phase experiment set before changing matcher, retrieval, or rerank behavior again

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-06-v034-e1-cross-encoder-rejected-handoff.md`.
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
> Current roadmap state:
> - `E1` is rejected after measured cross-encoder benchmarking
> - `E2` remains rejected after measured higher-shortlist comparisons
> - no active experiment remains in the current roadmap
>
> Please:
> 1. read the latest handoff and the live tracker,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, and rerank baselines,
> 4. do not reopen `E1` or `E2` unless new measured evidence justifies it,
> 5. define or request the next experiment set before making further behavioral changes,
> 6. keep the workflow benchmark-driven and one-variable-at-a-time,
> 7. leave a tracker update and a fresh handoff before ending.
