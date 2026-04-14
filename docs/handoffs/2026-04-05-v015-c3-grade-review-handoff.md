# Curriculum Matcher Handoff

- Date: 2026-04-05
- Version: `v015`
- Milestone: `C3` grade normalization measured and rejected

## Current Accepted Baseline

Matcher baseline:

- `C1` safe publisher alias normalization accepted
- `C2` safe product-title acronym expansion accepted
- `C3` grade normalization rejected and reverted

Rerank baseline:

- shortlist size `10`
- model `gpt-5.4-mini`
- reasoning effort `medium`

## What Happened In This Session

### `C3` was tested

A grade parsing fix was benchmarked to correct range-like values such as:

- `K-5`
- `K-12`
- `PK-12`
- `PK-K`
- `TK`

The change was benchmarked and then reverted because the result was too mixed against the accepted `C2` baseline.

### Validation on current reverted baseline

Passed:

- `python3 -m py_compile src/curriculum_matcher/app.py tests/test_matcher_core.py`
- `PYTHONPATH=src python3 -m pytest tests/test_matcher_core.py tests/test_evaluation.py tests/test_llm_rerank.py -q`

## `C3` Benchmark Result

Candidate artifacts:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_publisher_alias_grade_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_publisher_alias_grade_records.csv`

Compared against accepted baseline:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_acronym_alias_summary.json`

Overall deltas:

- top-1 accuracy: `0.4750 -> 0.4850` (`+0.0100`)
- top-3 recall: `0.5720 -> 0.5650` (`-0.0070`)
- prediction rate: `0.9400 -> 0.9320` (`-0.0080`)
- hit@10: `0.6130 -> 0.5990` (`-0.0140`)
- MRR: `0.5265 -> 0.5289` (`+0.0024`)

Important slice behavior:

- `assessment_short_or_acronym_title` regressed sharply
- `assessment_publisher_missing` regressed
- `product_type_usage / Assessment` regressed
- `placeholder_mapping / catalog_unspecified` improved
- expected `K` / `PK` / `TK` grade-span subset improved materially

Decision:

- `C3` rejected

Rationale:

- the targeted grade-span subset did improve
- but the accepted `C2` assessment improvements did not hold
- overall top-10 and prediction-rate behavior regressed too much to promote the change

## Current Tracker State

- `C1` accepted
- `C2` accepted
- `C3` rejected
- `C4` next

## Recommended Next Task

Start `C4` edition / year extraction improvement.

Why `C4` next:

- it keeps the work in the one-variable-at-a-time normalization lane
- it avoids immediately revisiting the assessment-vs-grade tradeoff from `C3`
- it may improve duplicate-title families without touching retrieval or rerank logic

## Files To Review First Next Session

- `docs/handoffs/2026-04-05-v015-c3-grade-review-handoff.md`
- `docs/analysis/2026-04-05-acronym-alias-review.md`
- `docs/analysis/2026-04-05-grade-normalization-review.md`
- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`
- `src/curriculum_matcher/app.py`
- `tests/test_matcher_core.py`

## Next Session Prompt

Use this prompt to resume:

> Resume work on the curriculum matcher from `docs/handoffs/2026-04-05-v015-c3-grade-review-handoff.md`.
>
> Current accepted matcher baseline:
> - `C1` safe publisher alias normalization
> - `C2` safe product-title acronym expansion
>
> Current rejected matcher experiment:
> - `C3` grade normalization tightening
>
> Current rerank baseline:
> - shortlist `10`
> - model `gpt-5.4-mini`
> - reasoning `medium`
>
> Please:
> 1. review the latest handoff, the `C2` and `C3` review notes, and current relevant code,
> 2. start `C4` edition / year extraction as the next one-variable-at-a-time matcher experiment,
> 3. keep the representative historical benchmark fixed,
> 4. compare against the accepted `C1 + C2` matcher baseline,
> 5. keep the work benchmark-driven and avoid broad refactors,
> 6. continue reporting the assessment slices explicitly so we do not reintroduce the `C3` regression pattern.
