# Curriculum Matcher Handoff v044

- Date: `2026-04-14`
- Milestone: `G2` measured and rejected
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
- `F1` system prompt tightening
- `F3` candidate disambiguation block
- `F4` abstention wording calibration
- `G2` matcher-internal confidence label suppression

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline: `F2` no-score candidate payload

Accepted comparison-contract baseline:

- use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## What This Session Did

- completed the live `G2` batch run
- compared `G2` against the canonical accepted `F2` row-matched baseline
- rejected `G2`
- restored `src/curriculum_matcher/llm_rerank.py` and `tests/test_llm_rerank.py` to the accepted `F2` baseline after measurement

Primary review note:

- `docs/analysis/2026-04-14-matcher-internal-label-suppression-review.md`

Tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## G2 Decision Summary

Candidate run:

- `historical-top10-gpt54mini-medium-g2-internal-labels-batch`

Shared-row comparison versus the canonical accepted `F2` baseline:

- shared rows: `450`
- baseline-only rows: `5`
- selection rate `-0.0111`
- all-row top-1 `+0.0000`
- selected-row top-1 `+0.0089`

Targeted slice movement:

- wrong-top1-but-gold-in-shortlist rows: top-1 `+0.0119`
- `catalog_state_specific_expected`: top-1 `-0.0090`
- assessment rows: top-1 `+0.0000`

Why `G2` was rejected:

- all-row top-1 did not improve
- selected-row top-1 improved only `+0.0089`, below the standing `+0.01` promotion threshold
- selection rate regressed
- the candidate batch also returned only `450` scored rows because `5` requests failed

## Current Roadmap State

- `G1` is accepted
- `G2` is rejected
- `G3` and `G4` are defined and not started
- no experiment is currently in progress

## Recommended Next Step

Lift exactly `G3` and change only the `DISTRICT_ROW` payload in `src/curriculum_matcher/llm_rerank.py` by removing:

- `usage_ambiguity`
- `state_specific_risk`
- `placeholder_mapping`
- `assessment_slice`

Measure `G3` against the canonical accepted `F2` baseline artifact set.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-14-v044-g2-rejected-handoff.md`.
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
> - rerank prompt baseline is `F2` no-score exposure
>
> Accepted comparison-contract baseline:
> - use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`
>
> Current roadmap state:
> - `G1` is accepted
> - `G2` is rejected
> - `G3` and `G4` are not started
> - no experiment is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, the `G2` review note, and `docs/roadmap/2026-04-14-post-f-experiment-set.md`,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, shortlist, model, reasoning, and `F2` rerank prompt baselines,
> 4. use the canonical accepted `F2` row-matched artifact set as the direct comparison baseline,
> 5. lift exactly `G3` and remove only `usage_ambiguity`, `state_specific_risk`, `placeholder_mapping`, and `assessment_slice` from `DISTRICT_ROW`,
> 6. keep experiment-changing work serial and one-variable-at-a-time,
> 7. leave a tracker update, a review note, and a fresh handoff before ending.
