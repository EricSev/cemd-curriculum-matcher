# Curriculum Matcher Handoff v019

- Date: 2026-04-05
- Milestone: `D3` character n-gram retrieval measured and rejected
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
- `D3` character n-gram retrieval

Accepted retrieval and rerank baseline:

- stage-1 recall: direct `0.5 * BM25 + 0.5 * semantic`
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

- `D3` is `rejected`

Reason:

- topline improved, but the win was not clean across protected slices
- `catalog_unspecified` top-3 recall regressed
- `assessment_publisher_missing` top-3 recall regressed
- review defaults for this phase say mixed retrieval results should remain rejected

## Validation Completed

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./.venv/bin/python -m unittest tests.test_matcher_core tests.test_evaluation`

## Benchmark Environment Note

The representative benchmark still needed offline flags in this environment:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

## Current Tracker State

Use `docs/roadmap/2026-04-05-next-phase-review-task-list.md` as the live tracker.

After this handoff, it shows:

- `D1` rejected
- `D2` rejected
- `D3` rejected
- `D4` not started

## Next Task

Start `D4`:

- retrieval baseline decision checkpoint
- compare accepted retrieval candidates only
- decide whether the direct recall blend remains the default retrieval path
- confirm whether shortlist stays `10`
- keep the representative historical benchmark fixed
- keep rerank settings fixed

## Next Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-05-v019-d3-char-ngram-rejected-handoff.md`.
>
> Current verified matcher baseline:
> - `C1` safe publisher alias normalization accepted
> - `C2` safe product-title acronym expansion accepted
> - `C5` safe state-specific token normalization accepted
> - `C3` grade normalization rejected
> - `C4` edition / year cleanup rejected
> - `D1` hybrid retrieval with RRF rejected
> - `D2` field-aware lexical weighting rejected
> - `D3` character n-gram retrieval rejected
>
> Current verified retrieval and rerank baseline:
> - stage-1 recall `0.5 * BM25 + 0.5 * semantic`
> - shortlist `10`
> - model `gpt-5.4-mini`
> - reasoning `medium`
>
> Please:
> 1. review the latest handoff, the `D3` review note, and the current tracker,
> 2. start `D4` retrieval baseline decision checkpoint as the next one-variable-at-a-time task,
> 3. compare only accepted retrieval candidates and rejected-candidate evidence already on disk,
> 4. decide whether the direct recall blend remains the default retrieval path and whether shortlist stays `10`,
> 5. keep the workflow benchmark-driven and avoid broad refactors,
> 6. preserve the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 7. keep rerank settings fixed,
> 8. use `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for any benchmark reruns in this environment,
> 9. continue reporting assessment slices, `catalog_unspecified`, state-risk slices, and sparse-evidence rows explicitly.
