# Curriculum Matcher Handoff v033

- Date: `2026-04-06`
- Milestone: `E1` cross-encoder blocker cleared and ready to resume
- Operator surface: keep using `src/curriculum_matcher/app.py`
- Workflow: benchmark-driven, one variable at a time

## Current Verified Baseline

Accepted matcher changes:

- `C1` safe publisher alias normalization
- `C2` safe product-title acronym expansion
- `C5` safe state-specific token normalization

Rejected matcher / retrieval / shortlist changes:

- `C3` grade normalization tightening
- `C4` edition / year title cleanup
- `D1` hybrid retrieval with RRF
- `D2` field-aware lexical weighting
- `E2` shortlist-size retuning

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

Current experiment readiness:

- `E1` is no longer blocked because offline mode can now load the approved cross-encoder model

## What This Session Did

- confirmed again that the stripped automation worktree does not contain the roadmap, handoffs, or `src/curriculum_matcher/app.py`
- kept the active project path on the live OneDrive-backed repo that contains the roadmap and handoffs
- verified the default Hugging Face cache previously had no staged `cross-encoder/*` artifacts
- loaded `cross-encoder/ms-marco-MiniLM-L-6-v2` online from the live project root to stage it into the local cache
- re-ran the exact offline probe with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`

Direct unblock verification:

- command run:
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python - <<'PY'`
  - `from sentence_transformers import CrossEncoder`
  - `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
  - `PY`
- result:
  - offline load succeeded
  - cached model path now exists under `~/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2`

## Decision

Decision: stop here after unblocking the environment and hand off with `E1` ready to benchmark.

Why:

- the exact approved cross-encoder dependency is now staged offline
- no benchmark or code change has been run yet, so the experimental baseline remains clean
- the next run can start directly at the `E1` cascade evaluation instead of spending another cycle on cache verification

## Tracker Update

Updated live tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

Current roadmap state after this run:

- `E1` ready to resume
- `E2` rejected
- no active experiment is in progress yet

## Validation

No source files were changed.

No benchmark was run in this session.

Validated environment change:

- online load of `cross-encoder/ms-marco-MiniLM-L-6-v2` succeeded
- offline load of `cross-encoder/ms-marco-MiniLM-L-6-v2` succeeded with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`

## Next Task

Do this next:

- reopen `E1` and run the exact cross-encoder cascade benchmark against the locked retrieval baseline and shortlist `10`

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-06-v033-e1-unblocked-handoff.md`.
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
> - `E1` is now unblocked because offline cross-encoder weights are staged locally
> - `E2` remains rejected after measured higher-shortlist comparisons
> - no active experiment is in progress yet
>
> Please:
> 1. read the latest handoff and the live tracker,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. do not reopen the accepted `D4` retrieval baseline or the rejected `E2` shortlist change unless new measured evidence requires it,
> 4. reopen `E1` and benchmark the exact `cross-encoder/ms-marco-MiniLM-L-6-v2` cascade experiment against the locked baseline,
> 5. keep the workflow benchmark-driven and one-variable-at-a-time,
> 6. preserve the accepted matcher and rerank baselines unless a measured experiment clearly replaces them,
> 7. use `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for benchmark work in this environment,
> 8. leave a tracker update, a review note if `E1` is measured, and a fresh handoff before ending.
