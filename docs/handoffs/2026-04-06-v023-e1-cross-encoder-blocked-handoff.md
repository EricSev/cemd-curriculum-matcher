# Curriculum Matcher Handoff v023

- Date: `2026-04-06`
- Milestone: `E1` cross-encoder pre-LLM reranker attempt blocked
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

This session explicitly lifted `E1` from deferred and checked whether the environment could support the cross-encoder experiment offline before changing code.

Confirmed:

- Hugging Face cache contains `sentence-transformers/all-MiniLM-L6-v2`
- Hugging Face cache contains `sentence-transformers/all-mpnet-base-v2`
- no cached `cross-encoder/*` model is present

Direct blocker verification:

- command run:
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python - <<'PY'`
  - `from sentence_transformers import CrossEncoder`
  - `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
  - `PY`
- result:
  - `OSError`
  - offline mode could not find the model in local cache

## Decision

Decision: stop here and mark `E1` blocked.

Why:

- `E1` specifically requires a cross-encoder cascade experiment
- substituting the existing dual-encoder embeddings would change the experiment definition
- benchmarking without the actual cross-encoder would violate the one-variable and benchmark-driven rules
- no code changes are justified until the required model is locally available

## Tracker Update

Updated live tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`
- latest checkpoint now records the offline cache blocker
- `E1` now shows `blocked`

## Validation

No source changes were made.

No tests were run because the run stopped before implementation.

## Next Task

Do exactly one of these next:

- stage a specific offline cross-encoder model in the local Hugging Face cache, then resume `E1`
- if offline model staging is not possible, move `E1` back to `deferred` and lift `E2` instead

Recommended first choice:

- cache `cross-encoder/ms-marco-MiniLM-L-6-v2` or another explicitly approved cross-encoder model, then run the cascade experiment against the locked `D4` baseline

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-06-v023-e1-cross-encoder-blocked-handoff.md`.
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
> `E1` is currently blocked because this offline environment does not have a cached cross-encoder model. Please:
> 1. read the latest handoff and the live tracker,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. do not reopen the `D` retrieval series unless new measured evidence requires it,
> 4. if a cross-encoder model has now been staged offline, keep `E1` active and run exactly that cascade experiment,
> 5. if a cross-encoder model still is not available offline, change no code and decide whether to return `E1` to `deferred` and lift `E2` instead,
> 6. keep the workflow benchmark-driven and one-variable-at-a-time,
> 7. preserve the accepted matcher and rerank baselines unless a measured experiment clearly replaces them,
> 8. use `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for benchmark work in this environment,
> 9. leave a tracker update, a review note if a new experiment is measured, and a fresh handoff before ending.
