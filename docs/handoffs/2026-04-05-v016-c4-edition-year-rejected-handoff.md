# Curriculum Matcher Handoff v016

- Date: 2026-04-05
- Milestone: `C4` edition / year extraction measured and rejected
- Operator surface: keep using `src/curriculum_matcher/app.py`
- Workflow: benchmark-driven, one variable at a time

## Current Verified Baseline

Accepted matcher changes:

- `C1` safe publisher alias normalization
- `C2` safe product-title acronym expansion

Rejected matcher changes:

- `C3` grade normalization tightening
- `C4` edition / year title cleanup

Accepted rerank baseline:

- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## Important Repo Truth

There was a mismatch earlier between docs and code around `C2`.

That is now resolved:

- the code in `src/curriculum_matcher/app.py` again includes the accepted `C2` acronym expansion
- tests in `tests/test_matcher_core.py` again reflect the accepted `C2` baseline
- full regression pack passed after restoring that baseline

## C4 Result

Review note:

- `docs/analysis/2026-04-05-edition-year-review.md`

Artifacts:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_edition_year_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_edition_year_records.csv`

Decision:

- `C4` is `rejected`

Reason:

- it helped the intended year/edition-heavy subset
- but it regressed the accepted overall benchmark baseline and several protected slices
- especially `catalog_unspecified`, state-sensitive slices, and some assessment-adjacent rows

## Current Tracker State

Use `docs/roadmap/2026-04-05-next-phase-review-task-list.md` as the live tracker.

After this handoff, it should show:

- `C1` accepted
- `C2` accepted
- `C3` rejected
- `C4` rejected
- `C5` not started

## Validation Already Passed

- `python3 -m py_compile src/curriculum_matcher/app.py tests/test_matcher_core.py`
- `PYTHONPATH=src python3 -m pytest tests/test_matcher_core.py tests/test_evaluation.py tests/test_llm_rerank.py -q`

## Next Task

Start `C5`:

- state-specific token normalization
- keep the representative historical benchmark fixed
- compare against the accepted `C1 + C2` matcher baseline
- keep rerank settings fixed
- continue reporting assessment and state-risk slices explicitly

## Next Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-05-v016-c4-edition-year-rejected-handoff.md`.
>
> Current verified matcher baseline:
> - `C1` safe publisher alias normalization accepted
> - `C2` safe product-title acronym expansion accepted
> - `C3` grade normalization rejected
> - `C4` edition / year title cleanup rejected
>
> Current rerank baseline:
> - shortlist `10`
> - model `gpt-5.4-mini`
> - reasoning `medium`
>
> Please:
> 1. review the latest handoff, the `C4` review note, and current relevant code,
> 2. start `C5` state-specific token normalization as the next one-variable-at-a-time matcher experiment,
> 3. keep the representative historical benchmark fixed,
> 4. compare against the accepted `C1 + C2` matcher baseline,
> 5. keep the workflow benchmark-driven and avoid broad refactors,
> 6. continue reporting the assessment slices, `catalog_unspecified`, and state-risk slices explicitly.
