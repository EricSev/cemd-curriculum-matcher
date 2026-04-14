# Curriculum Matcher Handoff v040

- Date: `2026-04-06`
- Milestone: `F4` abstention wording calibration measured and rejected
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

Accepted prompt-contract change:

- `F2` suppress raw matcher scores in the rerank prompt payload

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## What This Session Did

- lifted `F4` and changed only the abstention guidance sentence in `SYSTEM_PROMPT`
- rebuilt the top-10 error-row prompt pack and batch requests
- ran a fresh batch rerank experiment with `gpt-5.4-mini`, reasoning `medium`
- compared the candidate directly against the accepted `F2` no-score baseline on the same `455` scored rows
- rejected `F4`
- restored `src/curriculum_matcher/llm_rerank.py` to the accepted `F2` no-score prompt baseline after measurement

Primary review note:

- `docs/analysis/2026-04-06-abstention-calibration-review.md`

Tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## F4 Decision Summary

Shared-row comparison versus the accepted `F2` baseline:

- selection rate `-0.0747`
- all-row top-1 `-0.0066`
- selected-row top-1 `+0.0526`

Why `F4` was rejected:

- it improved selected-row precision, but only by abstaining much more often
- that cost too much overall recovery
- `catalog_state_specific_expected` regressed by `-0.0769`
- wrong-top1-but-gold-in-shortlist rows regressed by `-0.0214`

Important artifact note:

- `F2` and `F4` aligned cleanly on the same `455` prompt rows
- the rejection therefore comes from a direct row-matched comparison against the accepted `F2` baseline

## Current Roadmap State

- `E1` remains rejected
- `E2` remains rejected
- `F1` remains rejected
- `F2` remains accepted
- `F3` remains rejected
- `F4` is now rejected
- no experiment is currently in progress

## Recommended Next Step

No active experiment remains after the `F`-series closeout. The next session should define a new experiment set before making further behavioral changes.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-06-v040-f4-abstention-rejected-handoff.md`.
>
> Current verified matcher baseline:
> - `C1` safe publisher alias normalization accepted
> - `C2` safe product-title acronym expansion accepted
> - `C5` safe state-specific token normalization accepted
> - `C3` grade normalization rejected
> - `C4` edition / year cleanup rejected
>
> Current verified prompt / retrieval / rerank baseline:
> - suppress raw matcher scores in the rerank prompt payload
> - stage-1 recall `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
> - shortlist `10`
> - model `gpt-5.4-mini`
> - reasoning `medium`
>
> Current roadmap state:
> - `E1` is rejected after measured cross-encoder benchmarking
> - `E2` remains rejected after measured higher-shortlist comparisons
> - `F1` is rejected after measured system-prompt benchmarking
> - `F2` is accepted after measured score-exposure benchmarking
> - `F3` is rejected after measured disambiguation-block benchmarking
> - `F4` is rejected after measured abstention-wording benchmarking
> - no experiment is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, the `F2` through `F4` review notes, and `docs/roadmap/2026-04-06-next-experiment-set.md`,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, and rerank baselines,
> 4. do not reopen `E1`, `E2`, `F1`, `F3`, or `F4` unless new measured evidence justifies it,
> 5. define the next experiment set before making further behavioral changes,
> 6. keep experiment-changing work serial and use subagents only for non-competing sidecar work,
> 7. keep the workflow benchmark-driven and one-variable-at-a-time,
> 8. leave a tracker update and a fresh handoff before ending.
