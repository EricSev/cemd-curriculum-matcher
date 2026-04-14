# Curriculum Matcher Handoff v022

- Date: `2026-04-06`
- Milestone: validation-only checkpoint after `D4`
- Operator surface: keep using `src/curriculum_matcher/app.py`
- Workflow: benchmark-driven, one variable at a time

## Current Verified Baseline

Accepted matcher changes:

- `C1` safe publisher alias normalization
- `C2` safe product-title acronym expansion
- `C5` safe state-specific token normalization

Rejected matcher / retrieval changes:

- `C3` grade normalization tightening
- `C4` edition / year title cleanup
- `D1` hybrid retrieval with RRF
- `D2` field-aware lexical weighting

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## What This Session Did

This session did not start a new experiment.

Reason:

- the newest handoff (`v021`) already closed `D4`
- the live tracker still shows no `in progress` task
- the only remaining roadmap items, `E1` and `E2`, are both explicitly `deferred`
- per the roadmap rules, the next session must first choose exactly one deferred task to lift before changing code or running a new benchmark

## Validation Completed

Source confirmed:

- `src/curriculum_matcher/app.py` still defaults `CURRICULUM_MATCHER_RETRIEVAL_EXPERIMENT` to `char_ngram`
- `tests/test_matcher_core.py` still covers the char n-gram default and explicit direct override

Validation command completed:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./.venv/bin/python -m unittest tests.test_matcher_core tests.test_evaluation -q`

Validation result:

- `Ran 28 tests`
- `OK`

## Tracker State

Use `docs/roadmap/2026-04-05-next-phase-review-task-list.md` as the live tracker.

After this checkpoint, it still shows:

- `D4` accepted
- `E1` deferred
- `E2` deferred

## Decision

Decision: stop here without opening a new experiment.

Why:

- no active experiment was in flight
- no pending non-deferred experiment exists in the current roadmap
- starting `E1` or `E2` without explicitly lifting that deferral would violate the handoff and tracker rules

## Next Task

Before editing code or launching another benchmark, the next session must do exactly one of these:

- lift `E1` from `deferred` to `in progress` and run the cross-encoder pre-LLM reranker experiment
- lift `E2` from `deferred` to `in progress` and run shortlist-size retuning
- open a new roadmap if next-phase priorities changed

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-06-v022-validation-only-awaiting-next-experiment-handoff.md`.
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
> Please:
> 1. read the latest handoff, the `D4` decision note, and the live tracker,
> 2. do not reopen the `D` retrieval series unless new measured evidence requires it,
> 3. choose exactly one next task by explicitly lifting either `E1` or `E2` from `deferred` to `in progress` before editing code,
> 4. keep the workflow benchmark-driven and one-variable-at-a-time,
> 5. preserve the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 6. keep the accepted matcher and rerank baselines fixed unless a measured experiment clearly replaces them,
> 7. use `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for benchmark work in this environment,
> 8. leave a tracker update, a review note if a new experiment is measured, and a fresh handoff before ending.
