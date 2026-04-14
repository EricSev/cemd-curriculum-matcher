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

- `E1` cross-encoder pre-LLM reranker remains blocked because offline mode still cannot load a cached cross-encoder model

## What This Session Did

- reconfirmed that `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` still cannot load `cross-encoder/ms-marco-MiniLM-L-6-v2`
- kept `E1` blocked and measured `E2` with one variable only: shortlist `10` -> `15`
- ran the representative historical benchmark against the locked `D4` baseline with `topn_final=15`
- generated matching prompt-pack coverage summaries for error rows at top-10 and top-15

Artifacts saved:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_top15_shortlist_retune_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top15_shortlist_retune_records.csv`
- `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-batch-prompts-summary.json`
- `benchmarks/outputs/openai_batch_runs/historical-top15-gpt54mini-medium-batch-prompts-summary.json`
- review note: `docs/analysis/2026-04-06-shortlist-size-retuning-review.md`

## Decision

Decision: reject `E2`

Why:

- `top-15` left `top1`, `top3`, `hit@10`, and `nDCG@10` unchanged
- `MRR` improved only `+0.0003`
- `hit@15` rose from `0.6280` to `0.6320`, which reflects only `4` newly rescued rows at ranks `11-15`
- prompt-pack row count stayed `455`, but average candidates per error row rose from `4.1714` to `4.6022`
- `59` error rows expanded beyond `10` candidates without a measured rerank win that would justify the added review payload

## Tracker Update

Updated live tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

Current roadmap state after this run:

- `E1` blocked
- `E2` rejected
- no active experiment remains in this roadmap

## Validation

No source files were changed.

Benchmark validation completed with:

- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./.venv/bin/python -m curriculum_matcher.evaluation_cli --benchmark-file benchmarks/gold/historical_07122025_representative_1000.csv --catalog-file "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv" --profile fast --topn-final 15 --output-json benchmarks/outputs/historical_07122025_representative_1000_fast_top15_shortlist_retune_summary.json --output-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top15_shortlist_retune_records.csv`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./.venv/bin/python scripts/evaluate_llm_rerank.py --records-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_records.csv --catalog-file "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv" --output-jsonl benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-batch-prompts.jsonl --output-json benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-batch-prompts-summary.json --topn-final 10 --only-errors`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./.venv/bin/python scripts/evaluate_llm_rerank.py --records-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top15_shortlist_retune_records.csv --catalog-file "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv" --output-jsonl benchmarks/outputs/openai_batch_runs/historical-top15-gpt54mini-medium-batch-prompts.jsonl --output-json benchmarks/outputs/openai_batch_runs/historical-top15-gpt54mini-medium-batch-prompts-summary.json --topn-final 15 --only-errors`

## Next Task

This roadmap has no active experiment left.

Do exactly one of these next:

- stage an approved offline cross-encoder model locally, then reopen `E1`
- if offline model staging is still not possible, close this roadmap and define a new next-phase experiment set before changing matcher or rerank behavior

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-06-v024-e2-shortlist-retune-rejected-handoff.md`.
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
> - `E1` blocked because offline cross-encoder weights are still unavailable
> - `E2` rejected after a measured higher-shortlist comparison
> - no active experiment remains in the current roadmap
>
> Please:
> 1. read the latest handoff, the `E2` review note, and the live tracker,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. do not reopen the accepted `D4` retrieval baseline or the rejected `E2` shortlist change unless new measured evidence requires it,
> 4. if an approved cross-encoder model is now staged offline, reopen `E1` and benchmark that exact cascade experiment,
> 5. if offline cross-encoder staging is still unavailable, do not substitute another reranker; instead define or request the next roadmap before changing code,
> 6. keep the workflow benchmark-driven and one-variable-at-a-time,
> 7. preserve the accepted matcher and rerank baselines unless a measured experiment clearly replaces them,
> 8. use `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for benchmark work in this environment,
> 9. leave a tracker update, a review note if a new experiment is measured, and a fresh handoff before ending.
