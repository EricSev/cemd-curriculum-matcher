# Curriculum Matcher Handoff v036

- Date: `2026-04-06`
- Milestone: next experiment set defined after `E1` / `E2` rejection
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

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## What This Session Did

- read the latest handoff and the live tracker
- confirmed there was no active experiment left in the current roadmap
- kept the Tkinter operator surface untouched in `src/curriculum_matcher/app.py`
- did not reopen `E1` or `E2`
- defined the next experiment set as a new `F` series focused on the LLM rerank prompt contract
- updated the live tracker with new `F1` through `F4` tasks

Primary roadmap update:

- `docs/roadmap/2026-04-06-next-experiment-set.md`

Live tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## New F-Series Experiment Set

The next serial experiment sequence is:

- `F1` system prompt tightening
- `F2` raw matcher-score exposure
- `F3` candidate disambiguation block
- `F4` abstention wording calibration

All four are defined as prompt-contract-only experiments so they can be benchmarked without changing the accepted matcher, retrieval, shortlist, or model baseline.

## Recommended Next Step

Lift exactly `F1` from `not started` to `in progress`, change only `SYSTEM_PROMPT` in `src/curriculum_matcher/llm_rerank.py`, run the batch rerank comparison against the accepted `historical-top10-gpt54mini-medium-batch` baseline, and decide keep / reject before touching any later `F` task.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-06-v036-next-experiment-set-defined-handoff.md`.
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
> - `F1` through `F4` are defined and not started
> - no experiment is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, and `docs/roadmap/2026-04-06-next-experiment-set.md`,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, and rerank baselines,
> 4. do not reopen `E1` or `E2` unless new measured evidence justifies it,
> 5. lift exactly `F1` and run only that prompt-contract experiment,
> 6. keep experiment-changing work serial and use subagents only for non-competing sidecar work,
> 7. keep the workflow benchmark-driven and one-variable-at-a-time,
> 8. leave a tracker update, a review note, and a fresh handoff before ending.
