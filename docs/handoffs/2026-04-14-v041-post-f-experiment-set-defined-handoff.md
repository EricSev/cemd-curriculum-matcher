# Curriculum Matcher Handoff v041

- Date: `2026-04-14`
- Milestone: post-`F` experiment set defined after repo cleanup checkpoint
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

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline: `F2` no-score candidate payload

## What This Session Did

- cleaned and checkpointed the repo reorganization into a commitable packaged workspace
- removed machine-specific path state from `matcher_settings.json`
- kept local automation runtime state out of version control
- reviewed the closed `F` series and confirmed the accepted baseline is still `F2`
- defined the next serial experiment set as a new `G` series focused on prompt input contract rather than more top-level wording changes
- updated the live tracker with new `G1` through `G4` tasks

Primary roadmap update:

- `docs/roadmap/2026-04-14-post-f-experiment-set.md`

Live tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## New G-Series Experiment Set

The next serial sequence is:

- `G1` canonical prompt-pack alignment checkpoint
- `G2` matcher-internal confidence label suppression
- `G3` derived ambiguity label suppression
- `G4` candidate series exposure experiment

The first task is intentionally a comparison-contract checkpoint rather than a behavior change. It should freeze one accepted `F2` prompt pack and scored output as the official row-matched baseline for future rerank comparisons.

## Recommended Next Step

Lift exactly `G1` from `not started` to `in progress`, create the canonical accepted `F2` row-matched baseline artifact set, and stop there before changing rerank behavior.

If `G1` completes cleanly, then lift exactly `G2` next and change only the `DISTRICT_ROW` payload in `src/curriculum_matcher/llm_rerank.py` by removing `confidence_band` and `match_selected_strategy`.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-14-v041-post-f-experiment-set-defined-handoff.md`.
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
> - rerank prompt baseline is `F2` no-score exposure
>
> Current roadmap state:
> - `F1` rejected
> - `F2` accepted
> - `F3` rejected
> - `F4` rejected
> - `G1` through `G4` are defined and not started
> - no experiment is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, and `docs/roadmap/2026-04-14-post-f-experiment-set.md`,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, shortlist, model, reasoning, and `F2` rerank prompt baselines,
> 4. do not reopen `E1`, `E2`, or rejected `F` tasks unless new measured evidence justifies it,
> 5. lift exactly `G1` first and run only that comparison-contract checkpoint,
> 6. keep experiment-changing work serial and one-variable-at-a-time,
> 7. leave a tracker update, a review note, and a fresh handoff before ending.
