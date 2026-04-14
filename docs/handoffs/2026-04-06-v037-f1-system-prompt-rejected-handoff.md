# Curriculum Matcher Handoff v037

- Date: `2026-04-06`
- Milestone: `F1` system prompt tightening measured and rejected
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

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## What This Session Did

- read the latest handoff, live tracker, and next-experiment definition
- lifted `F1` and changed only `SYSTEM_PROMPT` in `src/curriculum_matcher/llm_rerank.py`
- rebuilt the top-10 error-row prompt pack and batch requests
- ran a fresh batch rerank experiment with `gpt-5.4-mini`, reasoning `medium`
- compared the candidate against the accepted baseline on the `424` shared rows present in both scored outputs
- rejected `F1`
- restored `src/curriculum_matcher/llm_rerank.py` to the accepted prompt baseline after measurement

Primary review note:

- `docs/analysis/2026-04-06-system-prompt-tightening-review.md`

Tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## F1 Decision Summary

Shared-row comparison versus the accepted baseline:

- selection rate `+0.0920`
- all-row top-1 `+0.0330`
- selected-row top-1 `+0.0015`

Why `F1` was rejected:

- the standing rule requires at least `+0.01` on both all-row and selected-row top-1
- `F1` did not clear the selected-row threshold
- `F1` also regressed selection rate materially

Important artifact note:

- the accepted historical scored artifact and the current error-row prompt pack do not align one-to-one
- the measured keep / reject decision therefore used the `424` shared `selection_identifier` rows instead of pretending the full historical artifact was row-matched to the new prompt pack

## Current Roadmap State

- `E1` remains rejected
- `E2` remains rejected
- `F1` is now rejected
- `F2`, `F3`, and `F4` are defined and not started
- no experiment is currently in progress

## Recommended Next Step

If work continues in this roadmap, lift exactly `F2` and test raw matcher-score exposure only. Keep the accepted matcher, retrieval, shortlist, model, and reasoning baselines fixed.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-06-v037-f1-system-prompt-rejected-handoff.md`.
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
>
> Current roadmap state:
> - `E1` is rejected after measured cross-encoder benchmarking
> - `E2` remains rejected after measured higher-shortlist comparisons
> - `F1` is rejected after measured system-prompt benchmarking
> - `F2` through `F4` are defined and not started
> - no experiment is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, the `F1` review note, and `docs/roadmap/2026-04-06-next-experiment-set.md`,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, and rerank baselines,
> 4. do not reopen `E1`, `E2`, or `F1` unless new measured evidence justifies it,
> 5. lift exactly `F2` and run only that prompt-contract experiment,
> 6. keep experiment-changing work serial and use subagents only for non-competing sidecar work,
> 7. keep the workflow benchmark-driven and one-variable-at-a-time,
> 8. leave a tracker update, a review note, and a fresh handoff before ending.
