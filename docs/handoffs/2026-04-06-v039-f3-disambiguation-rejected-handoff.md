# Curriculum Matcher Handoff v039

- Date: `2026-04-06`
- Milestone: `F3` candidate disambiguation block measured and rejected
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

Accepted prompt-contract change:

- `F2` suppress raw matcher scores in the rerank prompt payload

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## What This Session Did

- lifted `F3` and changed only prompt structure in `src/curriculum_matcher/llm_rerank.py`
- added a compact `CANDIDATE_DISAMBIGUATION` block using existing row and candidate metadata
- rebuilt the top-10 error-row prompt pack and batch requests
- ran a fresh batch rerank experiment with `gpt-5.4-mini`, reasoning `medium`
- compared the candidate against the accepted `F2` no-score baseline on the same `455` scored rows
- rejected `F3`
- restored `src/curriculum_matcher/llm_rerank.py` to the accepted `F2` no-score prompt baseline after measurement

Primary review note:

- `docs/analysis/2026-04-06-disambiguation-block-review.md`

Tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## F3 Decision Summary

Shared-row comparison versus the accepted `F2` baseline:

- selection rate `-0.0022`
- all-row top-1 `-0.0022`
- selected-row top-1 `-0.0027`
- invalid selected ids `+1`

Why `F3` was rejected:

- both top-1 metrics regressed versus the accepted `F2` baseline
- the state-specific target slice stayed flat instead of improving
- `Assessment`, `assessment_short_or_acronym_title`, and wrong-top1-but-gold-in-shortlist rows regressed
- the candidate introduced one invalid selected id

Important artifact note:

- unlike the earlier historical-baseline comparisons, `F2` and `F3` aligned cleanly on the same `455` prompt rows
- the rejection therefore comes from a direct row-matched comparison against the accepted `F2` baseline

## Current Roadmap State

- `E1` remains rejected
- `E2` remains rejected
- `F1` remains rejected
- `F2` remains accepted
- `F3` is now rejected
- `F4` is defined and not started
- no experiment is currently in progress

## Recommended Next Step

If work continues in this roadmap, lift exactly `F4` and test abstention wording calibration only, using the accepted `F2` no-score prompt baseline as the comparison point.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-06-v039-f3-disambiguation-rejected-handoff.md`.
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
> - `F3` is rejected after measured disambiguation-block benchmarking
> - `F4` is defined and not started
> - no experiment is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, the `F2` and `F3` review notes, and `docs/roadmap/2026-04-06-next-experiment-set.md`,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, and rerank baselines,
> 4. do not reopen `E1`, `E2`, `F1`, or `F3` unless new measured evidence justifies it,
> 5. use the accepted `F2` no-score prompt baseline and lift exactly `F4`,
> 6. keep experiment-changing work serial and use subagents only for non-competing sidecar work,
> 7. keep the workflow benchmark-driven and one-variable-at-a-time,
> 8. leave a tracker update, a review note, and a fresh handoff before ending.
