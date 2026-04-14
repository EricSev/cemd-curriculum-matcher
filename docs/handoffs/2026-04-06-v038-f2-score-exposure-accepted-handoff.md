# Curriculum Matcher Handoff v038

- Date: `2026-04-06`
- Milestone: `F2` raw matcher-score exposure measured and accepted
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

Accepted prompt-contract change:

- `F2` suppress raw matcher scores in the rerank prompt payload

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## What This Session Did

- lifted `F2` and changed only prompt score exposure in `src/curriculum_matcher/llm_rerank.py`
- removed raw candidate `score` from the LLM shortlist payload while keeping candidate order unchanged
- rebuilt the top-10 error-row prompt pack and batch requests
- ran a fresh batch rerank experiment with `gpt-5.4-mini`, reasoning `medium`
- compared the candidate against the accepted baseline on the `424` shared rows present in both scored outputs
- accepted `F2`
- kept the no-score prompt change in code as the new prompt baseline

Primary review note:

- `docs/analysis/2026-04-06-score-exposure-review.md`

Tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## F2 Decision Summary

Shared-row comparison versus the prior accepted baseline:

- selection rate `+0.0378`
- all-row top-1 `+0.0330`
- selected-row top-1 `+0.0379`

High-value slice movement:

- wrong-top1-but-gold-in-shortlist rows: top-1 `+0.1111`
- `catalog_state_specific_expected`: top-1 `+0.3333`
- `Assessment`: top-1 `+0.0583`

Why `F2` was accepted:

- it cleared the promotion threshold on both all-row and selected-row top-1
- invalid and repaired id counts stayed flat at `0`
- it improved the specific recovery scenario `F2` was meant to test

Important artifact note:

- the accepted historical scored artifact and the current error-row prompt pack still do not align one-to-one
- the measured keep / accept decision therefore used the `424` shared `selection_identifier` rows instead of treating the full historical artifact as row-matched to the new prompt pack

## Current Roadmap State

- `E1` remains rejected
- `E2` remains rejected
- `F1` remains rejected
- `F2` is accepted
- `F3` and `F4` are defined and not started
- no experiment is currently in progress

## Recommended Next Step

If work continues in this roadmap, lift exactly `F3` and test the candidate disambiguation block only, using the accepted `F2` no-score prompt baseline as the new comparison point.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-06-v038-f2-score-exposure-accepted-handoff.md`.
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
> - `F3` and `F4` are defined and not started
> - no experiment is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, the `F2` review note, and `docs/roadmap/2026-04-06-next-experiment-set.md`,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, and rerank baselines,
> 4. do not reopen `E1`, `E2`, or `F1` unless new measured evidence justifies it,
> 5. use the accepted `F2` no-score prompt baseline and lift exactly `F3`,
> 6. keep experiment-changing work serial and use subagents only for non-competing sidecar work,
> 7. keep the workflow benchmark-driven and one-variable-at-a-time,
> 8. leave a tracker update, a review note, and a fresh handoff before ending.
