# Curriculum Matcher Handoff v021

- Date: `2026-04-05`
- Milestone: `D4` retrieval baseline checkpoint accepted
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

## D4 Result

Decision note:

- `docs/analysis/2026-04-05-retrieval-baseline-decision.md`

Decision:

- `D4` is `accepted`

Reason:

- the accepted char n-gram retrieval blend still beats the direct blend on the fixed representative benchmark
- current code in `src/curriculum_matcher/app.py` already matches that accepted baseline
- shortlist `10` stays justified, so no retrieval rollback or shortlist retune is warranted inside this checkpoint

## Source And Validation Completed

Code path validated:

- `src/curriculum_matcher/app.py`
- `tests/test_matcher_core.py`

Validation completed:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./.venv/bin/python -m unittest tests.test_matcher_core tests.test_evaluation -q`

## Current Tracker State

Use `docs/roadmap/2026-04-05-next-phase-review-task-list.md` as the live tracker.

After this handoff, it shows:

- `D4` accepted
- `E1` deferred
- `E2` deferred

## Next Task

No active experiment is open now that the `D` retrieval series is closed.

If the next session continues this roadmap, it should do one of these explicitly before changing code:

- decide to lift `E1` deferral and start the cross-encoder pre-LLM reranker experiment
- decide to lift `E2` deferral and start shortlist-size retuning
- open a new roadmap if the next phase priorities changed

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-05-v021-d4-retrieval-baseline-confirmed-handoff.md`.
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
> 3. if continuing this roadmap, choose exactly one next task by explicitly lifting either `E1` or `E2` from deferred status before editing code,
> 4. keep the workflow benchmark-driven and one-variable-at-a-time,
> 5. preserve the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 6. keep the accepted matcher and rerank baselines fixed unless a measured experiment clearly replaces them,
> 7. use `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for benchmark work in this environment,
> 8. leave a tracker update, a review note if a new experiment is measured, and a fresh handoff before ending.
