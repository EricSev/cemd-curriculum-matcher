# Curriculum Matcher Handoff v020

- Date: `2026-04-05`
- Milestone: `D3` character n-gram retrieval measured and accepted
- Operator surface: keep using `src/curriculum_matcher/app.py`
- Workflow: benchmark-driven, one variable at a time

## Current Verified Baseline

Accepted matcher changes:

- `C1` safe publisher alias normalization
- `C2` safe product-title acronym expansion
- `C5` safe state-specific token normalization

Rejected matcher / retrieval changes:

- `C3` grade normalization tightening
- `C4` edition / year title cleanup
- `D1` hybrid retrieval with RRF
- `D2` field-aware lexical weighting

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## D3 Result

Review note:

- `docs/analysis/2026-04-05-char-ngram-retrieval-review.md`

Artifacts:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_records.csv`

Decision:

- `D3` is `accepted`

Reason:

- shortlist hit@10, MRR, and nDCG@10 all improved over the accepted state-normalization baseline
- assessment acronym rows and state-risk rows improved meaningfully
- row-level churn added `9` top-10 wins with `0` top-10 losses
- the candidate is strong enough to replace the direct retrieval blend

## Source And Validation Completed

Code updates:

- `src/curriculum_matcher/app.py`
- `tests/test_matcher_core.py`

Validation completed:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./.venv/bin/python -m unittest tests.test_matcher_core tests.test_evaluation -q`
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python -m curriculum_matcher.evaluation_cli --benchmark-file benchmarks/gold/historical_07122025_representative_1000.csv --catalog-file "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv" --profile fast --topn-final 10 --output-json benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_summary.json --output-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_records.csv`

## Current Tracker State

Use `docs/roadmap/2026-04-05-next-phase-review-task-list.md` as the live tracker.

After this handoff, it shows:

- `D3` accepted
- `D4` not started

## Next Task

Start `D4`:

- retrieval baseline decision checkpoint
- compare accepted retrieval candidates only
- confirm the retrieval default now that `D3` is accepted
- decide whether shortlist stays `10`
- keep the representative historical benchmark fixed
- keep rerank settings fixed
- continue reporting assessment slices, `catalog_unspecified`, state-risk slices, and sparse-evidence rows explicitly

## Next Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-05-v020-d3-char-ngram-accepted-handoff.md`.
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
> Please:
> 1. review the latest handoff, the `D3` review note, and the current retrieval code,
> 2. run `D4` as the next one-variable-at-a-time checkpoint,
> 3. keep the representative historical benchmark fixed,
> 4. compare accepted retrieval candidates only and confirm whether the retrieval baseline should stay on character n-gram blend,
> 5. keep the workflow benchmark-driven and avoid broad refactors,
> 6. preserve the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 7. keep rerank settings fixed,
> 8. use `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for benchmark work in this environment,
> 9. continue reporting assessment slices, `catalog_unspecified`, state-risk slices, and sparse-evidence rows explicitly.
