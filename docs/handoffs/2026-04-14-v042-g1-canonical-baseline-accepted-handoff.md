# Curriculum Matcher Handoff v042

- Date: `2026-04-14`
- Milestone: `G1` canonical prompt-pack alignment checkpoint accepted
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

Accepted comparison-contract baseline:

- use the row-matched `F2` artifact set recorded in:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## What This Session Did

- executed `G1` as a comparison-contract checkpoint only
- verified that the accepted `F2` prompt pack and scored output align exactly one-to-one
- confirmed:
  - prompt rows `455`
  - scored rows `455`
  - prompt-minus-scored `0`
  - scored-minus-prompt `0`
- recorded the accepted `F2` row-matched artifact set as the official rerank comparison baseline for the `G` series
- did not change matcher, retrieval, rerank behavior, shortlist, model, reasoning, or the Tkinter operator surface

Primary review note:

- `docs/analysis/2026-04-14-canonical-prompt-pack-alignment-review.md`

Tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## G1 Decision Summary

- `G1` is `accepted`
- this was not a behavior-change task
- the accepted `F2` no-score run is now the default direct comparison base for future `G`-series rerank experiments

Why `G1` was accepted:

- it removed avoidable ambiguity caused by comparing against the older pre-`F2` historical scored artifact
- it produced a stable row-matched baseline artifact set without changing model behavior

## Current Roadmap State

- `G1` is accepted
- `G2`, `G3`, and `G4` are defined and not started
- no experiment is currently in progress

## Recommended Next Step

Lift exactly `G2` and change only the `DISTRICT_ROW` payload in `src/curriculum_matcher/llm_rerank.py` by removing:

- `confidence_band`
- `match_selected_strategy`

Measure `G2` against the canonical accepted `F2` baseline artifact set, not the older pre-`F2` historical baseline.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-14-v042-g1-canonical-baseline-accepted-handoff.md`.
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
> Accepted comparison-contract baseline:
> - use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`
>
> Current roadmap state:
> - `G1` is accepted
> - `G2`, `G3`, and `G4` are defined and not started
> - no experiment is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, the `G1` review note, and `docs/roadmap/2026-04-14-post-f-experiment-set.md`,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, shortlist, model, reasoning, and `F2` rerank prompt baselines,
> 4. use the canonical accepted `F2` row-matched artifact set as the direct comparison baseline,
> 5. lift exactly `G2` and remove only `confidence_band` and `match_selected_strategy` from `DISTRICT_ROW`,
> 6. keep experiment-changing work serial and one-variable-at-a-time,
> 7. leave a tracker update, a review note, and a fresh handoff before ending.
