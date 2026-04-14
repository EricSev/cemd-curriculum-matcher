# Curriculum Matcher Handoff v017

- Date: 2026-04-05
- Milestone: `D1` hybrid retrieval with RRF measured and rejected
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

Accepted rerank baseline:

- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## D1 Result

Review note:

- `docs/analysis/2026-04-05-rrf-retrieval-review.md`

Artifacts:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_rrf_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_rrf_records.csv`

Decision:

- `D1` is `rejected`

Reason:

- RRF produced only a tiny shortlist lift
- top-3 recall regressed overall
- protected recall metrics regressed for `catalog_unspecified` and `assessment_publisher_missing`
- the candidate does not beat the accepted `C1 + C2 + C5` baseline cleanly enough

## Validation Completed

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest tests/test_matcher_core.py tests/test_evaluation.py -q`

## Benchmark Environment Note

The representative benchmark needed offline flags in this environment:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

Without them, model loading attempted a network call and failed before evaluation started.

## Current Tracker State

Use `docs/roadmap/2026-04-05-next-phase-review-task-list.md` as the live tracker.

After this handoff, it should show:

- `C5` accepted
- `D1` rejected
- `D2` not started

## Next Task

Start `D2`:

- field-aware lexical weighting experiment
- keep the representative historical benchmark fixed
- compare against the accepted `C1 + C2 + C5` matcher baseline
- keep rerank settings fixed
- continue reporting assessment slices, `catalog_unspecified`, state-risk slices, and sparse-evidence rows explicitly

## Next Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-05-v017-d1-rrf-rejected-handoff.md`.
>
> Current verified matcher baseline:
> - `C1` safe publisher alias normalization accepted
> - `C2` safe product-title acronym expansion accepted
> - `C5` safe state-specific token normalization accepted
> - `C3` grade normalization rejected
> - `C4` edition / year cleanup rejected
> - `D1` hybrid retrieval with RRF rejected
>
> Current rerank baseline:
> - shortlist `10`
> - model `gpt-5.4-mini`
> - reasoning `medium`
>
> Please:
> 1. review the latest handoff, the `D1` review note, and current relevant retrieval code,
> 2. start `D2` field-aware lexical weighting as the next one-variable-at-a-time experiment,
> 3. keep the representative historical benchmark fixed,
> 4. compare against the accepted `C1 + C2 + C5` matcher baseline,
> 5. keep the workflow benchmark-driven and avoid broad refactors,
> 6. preserve the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 7. keep rerank settings fixed,
> 8. use `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for benchmark runs in this environment,
> 9. continue reporting assessment slices, `catalog_unspecified`, state-risk slices, and sparse-evidence rows explicitly.
