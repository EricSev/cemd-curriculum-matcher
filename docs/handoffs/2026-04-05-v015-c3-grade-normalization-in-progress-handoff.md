# Curriculum Matcher Handoff v015

- Date: 2026-04-05
- Milestone: `C3` grade normalization tightening implemented and validated locally; representative benchmark still running
- Operator surface: keep using `src/curriculum_matcher/app.py`
- Benchmark discipline: keep one experimental variable at a time

## Source Of Truth

Use the repo artifacts and docs in the workspace as the source of truth.

Current accepted / rejected experiment state:

- `C1` publisher alias normalization: `accepted`
- `C2` acronym and abbreviation expansion: `rejected`
- `C3` grade normalization tightening: `in progress`

Current accepted matcher + rerank baseline:

- safe publisher alias normalization is part of the matcher baseline
- shortlist size: `10`
- rerank model: `gpt-5.4-mini`
- reasoning effort: `medium`

Important clarification:

- despite conflicting subagent chatter earlier, the verified repo state still treats `C2` as `rejected`
- the live tracker and saved review note reflect that repo truth

## What Changed In C3

Code change is in `src/curriculum_matcher/app.py`.

The change is intentionally narrow:

- only grade parsing was tightened
- no scoring weights changed
- no retrieval logic changed
- no rerank settings changed

`_parse_grade_range(...)` now correctly handles mixed elementary and PK/K formats such as:

- `K-5` -> `(0, 5)`
- `K-12` -> `(0, 12)`
- `PK-12` -> `(0, 12)`
- `PK-K` -> `(0, 0)`
- `K,1,2,3` -> `(0, 3)`
- `TK` -> `(0, 0)`

This fixes a real normalization bug in the previous parser, which collapsed many catalog bands like `K-5` and `PK-12` down to the numeric portion only.

## Tests Added / Updated

Updated `tests/test_matcher_core.py` with coverage for:

- mixed elementary / PK / TK grade spans
- kindergarten-band overlap scoring

Local validation already passed:

- `python3 -m py_compile src/curriculum_matcher/app.py tests/test_matcher_core.py`
- `PYTHONPATH=src python3 -m pytest tests/test_matcher_core.py -q`

Representative regression pack also passed before this handoff:

- `PYTHONPATH=src python3 -m pytest tests/test_matcher_core.py tests/test_evaluation.py tests/test_llm_rerank.py -q`

## Current C3 Benchmark Status

The representative benchmark has been launched but has not yet written output files.

Two long-running benchmark processes are present:

1. PID `29143`
   - command writes to:
     - `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_publisher_alias_grade_summary.json`
     - `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_publisher_alias_grade_records.csv`

2. PID `29507`
   - command writes to:
     - `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_grade_normalization_summary.json`
     - `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_grade_normalization_records.csv`

Observed state before handoff:

- both processes were still active and using CPU
- neither output file pair existed yet
- this suggests `evaluation_cli` only writes the outputs at the end of the run

Recommended handling:

- use whichever of the two `C3` artifact pairs lands first
- prefer the `publisher_alias_grade_*` naming if both finish successfully, since that best reflects the true accepted baseline it is being compared against
- if one process fails but the other succeeds, use the successful one and document the artifact name explicitly in the review note

## Next Required Step

Finish `C3` as a measured experiment:

1. wait for one `C3` summary JSON and records CSV to land
2. compare it against:
   - `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_publisher_alias_summary.json`
   - `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_publisher_alias_records.csv`
3. record overall deltas:
   - top-1
   - top-3
   - hit@10
   - MRR
   - nDCG@10
4. record required hard-slice deltas:
   - `product_type_usage / Assessment`
   - `placeholder_mapping / catalog_unspecified`
   - `state_specific_risk / adoption_state_high_risk`
   - `state_specific_risk / catalog_state_specific_expected`
   - grade-sensitive row checks, especially kindergarten / PK / TK patterns
5. create the `C3` review note
6. mark `C3` as `accepted` or `rejected`
7. update the matcher baseline only if `C3` is accepted

## Recommended C3 Acceptance Standard

Because this is a real normalization bug fix, `C3` can be accepted if:

- the representative benchmark is directionally positive or neutral overall
- there is no meaningful regression in the tracked hard slices
- grade-sensitive rows show a credible lift, especially kindergarten / PK / TK / mixed-band rows

If the result is mixed or flat on the intended grade-sensitive rows, reject it and keep only `C1` in the matcher baseline.

## Next Session Prompt

Use this prompt to resume:

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-05-v015-c3-grade-normalization-in-progress-handoff.md`.
>
> Current verified state:
> - `C1` publisher alias normalization is accepted
> - `C2` acronym / abbreviation expansion is rejected
> - `C3` grade normalization tightening is implemented and locally validated, but the representative benchmark result still needs to be collected and reviewed
> - accepted rerank baseline remains shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
>
> Please:
> 1. check whether either `C3` benchmark artifact pair has landed,
> 2. compare the completed `C3` run against the accepted publisher-alias baseline,
> 3. write the `C3` review note and update the tracker with an explicit accept/reject decision,
> 4. if `C3` is accepted, update the baseline and then move to `C4` edition/year extraction,
> 5. if `C3` is rejected, keep the baseline unchanged and then move to `C4`,
> 6. keep the workflow benchmark-driven and one-variable-at-a-time,
> 7. do not start broad refactors.
