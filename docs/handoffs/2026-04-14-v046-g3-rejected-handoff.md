# Curriculum Matcher Handoff v046

- Date: `2026-04-14`
- Milestone: `G3` measured and rejected
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
- `G3` derived ambiguity label suppression

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline: `F2` no-score candidate payload

Accepted comparison-contract baseline:

- use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## What This Session Did

- resumed the live `G3` batch after it appeared stuck
- attempted to cancel it, but the API returned `409` because the batch had already completed
- downloaded the completed output and scored the run
- compared `G3` against the canonical accepted `F2` row-matched baseline
- rejected `G3`
- restored `src/curriculum_matcher/llm_rerank.py` and `tests/test_llm_rerank.py` to the accepted `F2` baseline after measurement

Primary review note:

- `docs/analysis/2026-04-14-derived-ambiguity-label-suppression-review.md`

Tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## G3 Decision Summary

Candidate run:

- `historical-top10-gpt54mini-medium-g3-derived-labels-batch`

Shared-row comparison versus the canonical accepted `F2` baseline:

- shared rows: `455`
- baseline-only rows: `0`
- selection rate `+0.0044`
- all-row top-1 `-0.0132`
- selected-row top-1 `-0.0292`

Targeted slice movement:

- `catalog_state_specific_expected`: top-1 `-0.1154`
- `catalog_unspecified`: top-1 `-0.0208`
- assessment rows: top-1 `+0.0000`
- wrong-top1-but-gold-in-shortlist rows: top-1 `-0.0429`

Why `G3` was rejected:

- all-row top-1 regressed materially
- selected-row top-1 regressed materially
- the state-specific, placeholder, and recovery target slices did not improve
- the slightly higher selection rate did not translate into better match quality

## Current Roadmap State

- `G1` is accepted
- `G2` is rejected
- `G3` is rejected
- `G4` is defined and not started
- no experiment is currently in progress

## Recommended Next Step

Lift exactly `G4` and change only the `SHORTLIST_CANDIDATES` payload in `src/curriculum_matcher/llm_rerank.py` by removing:

- `series`

Measure `G4` against the canonical accepted `F2` baseline artifact set.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-14-v046-g3-rejected-handoff.md`.
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
> - `G3` is rejected
> - `G4` is not started
> - no experiment is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, the `G3` review note, and `docs/roadmap/2026-04-14-post-f-experiment-set.md`,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, shortlist, model, reasoning, and `F2` rerank prompt baselines,
> 4. use the canonical accepted `F2` row-matched artifact set as the direct comparison baseline,
> 5. lift exactly `G4` and remove only `series` from `SHORTLIST_CANDIDATES`,
> 6. keep experiment-changing work serial and one-variable-at-a-time,
> 7. leave a tracker update, a review note, and a fresh handoff before ending.
