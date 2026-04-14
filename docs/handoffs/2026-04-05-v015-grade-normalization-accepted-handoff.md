# Curriculum Matcher Handoff v015

- Date: 2026-04-05
- Milestone: `C3` grade normalization tightening accepted
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`

## Source Of Truth

Use the saved repo artifacts and docs as the source of truth.

Most relevant docs:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`
- `docs/analysis/2026-04-05-publisher-alias-normalization-review.md`
- `docs/analysis/2026-04-05-acronym-alias-review.md`
- `docs/analysis/2026-04-05-grade-normalization-review.md`

## Current Accepted Baseline

Matcher baseline:

- `C1` publisher alias normalization accepted
- `C2` acronym/title expansion rejected
- `C3` grade normalization tightening accepted

Locked rerank baseline:

- shortlist size `10`
- model `gpt-5.4-mini`
- reasoning effort `medium`

Do not treat the rejected `C2` acronym expansion as part of the matcher baseline.

## What Changed In C3

`src/curriculum_matcher/app.py`

- `_parse_grade_range(...)` now recognizes `TK`, `PK`, `Pre-K`, and `K` inside mixed ranges and grade lists instead of only as standalone tokens
- this fixed cases like `K-5`, `PK-K`, `PK-12`, and `TK,K,1`

`tests/test_matcher_core.py`

- added coverage for kindergarten-bearing ranges and overlap scoring
- removed the rejected acronym-expansion expectations so tests now reflect the accepted baseline

## C3 Benchmark Result

Baseline artifact:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_publisher_alias_summary.json`

Accepted candidate artifact:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_publisher_alias_grade_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_publisher_alias_grade_records.csv`

Overall delta vs accepted `C1` baseline:

- top-1 accuracy: `0.4650 -> 0.4850` (`+0.0200`)
- top-3 recall: `0.5590 -> 0.5650` (`+0.0060`)
- hit@10: `0.6000 -> 0.5990` (`-0.0010`)
- MRR: `0.5148 -> 0.5289` (`+0.0141`)
- nDCG@10: `0.5359 -> 0.5464` (`+0.0105`)

Grade-focused evidence:

- input grade `K/PK/TK` top-1: `0.4085 -> 0.4507`
- expected catalog grades containing `K/PK` top-1: `0.4988 -> 0.5718`
- expected catalog grades containing `K/PK` MRR: `0.5658 -> 0.6146`

Important caveat:

- assessment slices regressed
- `Assessment` top-1: `0.2962 -> 0.2577`
- `assessment_short_or_acronym_title` top-1: `0.3106 -> 0.2795`
- `assessment_publisher_missing` top-1: `0.2840 -> 0.2346`

Interpretation:

- accept `C3` because it clearly improves the intended grade-sensitive slices and overall ranking quality
- keep assessment regressions explicitly visible in future reviews

## Validation

Passed:

- `python3 -m py_compile src/curriculum_matcher/app.py tests/test_matcher_core.py`
- `PYTHONPATH=src python3 -m pytest tests/test_matcher_core.py tests/test_evaluation.py tests/test_llm_rerank.py -q`

## Next Recommended Task

Start `C4`: edition / year extraction

Constraints:

- keep the workflow benchmark-driven
- change one variable only
- preserve the Tkinter UI operator surface in `src/curriculum_matcher/app.py`
- do not change core matcher scoring weights
- compare only against the accepted `C1 + C3` baseline

## Next Session Prompt

Resume work on the curriculum matcher from:

- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-05-v015-grade-normalization-accepted-handoff.md`
- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/roadmap/2026-04-05-next-phase-review-task-list.md`
- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/analysis/2026-04-05-grade-normalization-review.md`

Current accepted matcher baseline:

- publisher alias normalization accepted
- acronym/title expansion rejected
- grade normalization tightening accepted

Current accepted rerank baseline:

- shortlist size `10`
- model `gpt-5.4-mini`
- reasoning effort `medium`

Please:

1. verify the accepted `C1 + C3` matcher baseline in current code and tests,
2. inspect `C4` edition / year extraction opportunities in the benchmark and current code,
3. implement only a narrow edition / year parsing improvement,
4. benchmark it against `historical_07122025_representative_1000_fast_top10_publisher_alias_grade_summary.json`,
5. record overall deltas plus the relevant duplicate-title / year-variant slices,
6. accept or reject the experiment explicitly,
7. update the tracker and leave the next handoff at the next natural milestone.

Important constraints:

- keep the work benchmark-driven
- do not reintroduce the rejected acronym expansion as baseline behavior
- avoid broad refactors
- preserve one-variable-at-a-time experimentation
