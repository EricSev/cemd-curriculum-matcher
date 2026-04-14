# Curriculum Matcher Handoff v035

- Date: `2026-04-06`
- Milestone: session closeout after `E1` rejection
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

## Session Closeout State

- `E1` has been benchmarked and rejected
- `E2` remains rejected
- no active experiment remains in the roadmap
- the baton is parked at a clean decision boundary

Primary source files for the next session:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`
- `docs/analysis/2026-04-06-cross-encoder-rerank-review.md`
- `docs/handoffs/2026-04-06-v034-e1-cross-encoder-rejected-handoff.md`

## What The Next Session Should Do

Do not resume implementation from an in-flight experiment, because there is no in-flight experiment now.

Start by defining the next experiment set.

That next session may use:

- `Curriculum Worker Canonical` for one task unit
- `Curriculum Worker Batch` for up to `3` compatible task units

If batch mode is used:

- keep experiment-changing work serial
- use subagents and parallel agents only for non-competing sidecar work
- do not run competing experiment changes in parallel

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-06-v035-session-closeout-handoff.md`.
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
> - no active experiment remains in the current roadmap
>
> Please:
> 1. read the latest handoff and the live tracker,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, and rerank baselines,
> 4. do not reopen `E1` or `E2` unless new measured evidence justifies it,
> 5. define the next experiment set before making further behavioral changes,
> 6. if batch mode is appropriate, keep experiment-changing work serial and use subagents only for non-competing sidecar work,
> 7. keep the workflow benchmark-driven and one-variable-at-a-time,
> 8. leave a tracker update and a fresh handoff before ending.
