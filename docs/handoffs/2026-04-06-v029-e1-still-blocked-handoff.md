# Curriculum Matcher Handoff v029

- Date: `2026-04-06`
- Milestone: `E1` cross-encoder blocker rechecked and still blocked
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

Open blocker:

- `E1` cross-encoder pre-LLM reranker remains blocked because offline mode still cannot load a cached cross-encoder model

## What This Session Did

- confirmed the overnight run window was active before resuming work
- checked the stripped automation worktree first and found it did not contain the expected roadmap, handoffs, or `src/curriculum_matcher/app.py`
- resolved the active project path to the live OneDrive-backed repo that contains the roadmap and handoffs
- read the latest handoff and live tracker before touching the roadmap
- verified the default Hugging Face cache still has no staged `cross-encoder/*` artifacts
- re-ran the explicit offline load probe for `cross-encoder/ms-marco-MiniLM-L-6-v2` from the live project root

Direct blocker verification:

- command run:
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python - <<'PY'`
  - `from sentence_transformers import CrossEncoder`
  - `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
  - `PY`
- result:
  - `LocalEntryNotFoundError`
  - `OSError`
  - `We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.`

## Decision

Decision: stop here and keep `E1` blocked.

Why:

- the required cross-encoder is still unavailable offline
- substituting a different reranker would change the experiment definition
- the live roadmap still has no other active or pending experiment that can be run without redefining the roadmap
- no benchmark or code change is justified until the blocker is cleared or a new roadmap is explicitly defined

## Tracker Update

Updated live tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

Current roadmap state after this run:

- `E1` blocked
- `E2` rejected
- no active experiment remains in this roadmap

## Validation

No source files were changed.

No benchmark was run because the run stopped at blocker verification before implementation.

## Next Task

Do exactly one of these next:

- stage `cross-encoder/ms-marco-MiniLM-L-6-v2` or another explicitly approved cross-encoder model in the local offline cache, then reopen `E1`
- if offline staging is still not possible, define a new next-phase roadmap before changing matcher, retrieval, or rerank behavior

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-06-v029-e1-still-blocked-handoff.md`.
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
> - `E1` blocked because offline cross-encoder weights are still unavailable
> - `E2` rejected after measured higher-shortlist comparisons
> - no active experiment remains in the current roadmap
>
> Please:
> 1. read the latest handoff and the live tracker,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. do not reopen the accepted `D4` retrieval baseline or the rejected `E2` shortlist change unless new measured evidence requires it,
> 4. if an approved cross-encoder model is now staged offline, reopen `E1` and benchmark that exact cascade experiment,
> 5. if offline cross-encoder staging is still unavailable, do not substitute another reranker and do not silently invent a new experiment; define or request the next roadmap first,
> 6. keep the workflow benchmark-driven and one-variable-at-a-time,
> 7. preserve the accepted matcher and rerank baselines unless a measured experiment clearly replaces them,
> 8. use `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for benchmark work in this environment,
> 9. leave a tracker update, a review note if a new experiment is measured, and a fresh handoff before ending.
