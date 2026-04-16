# Curriculum Matcher Handoff v049

- Date: `2026-04-16`
- Milestone: `H` series defined after `G4` closeout
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
- `G2` matcher-internal confidence label suppression
- `G3` derived ambiguity label suppression
- `G4` candidate series exposure experiment

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline: `F2` no-score candidate payload

Accepted comparison-contract baseline:

- use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## What This Session Did

- pushed the completed `G4` closeout commit to `origin/codex/g-series-closeout`
- confirmed the working tree was clean after the push
- read the latest `G4` handoff, the live tracker, the post-`F` experiment definition, the `G1` canonical baseline note, and the shortlist bottleneck snapshot
- defined the next `H` experiment set without changing matcher, retrieval, shortlist, rerank prompt, model, or reasoning behavior

New roadmap definition:

- `docs/roadmap/2026-04-16-next-experiment-set.md`

Updated tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## H-Series Decision Summary

The `G` series showed that more rerank prompt input-contract trimming did not beat the accepted `F2` no-score baseline. The next phase should focus on candidate recall and shortlist quality before the LLM reranker sees the row.

Defined sequence:

- `H1` shortlist failure audit checkpoint
- `H2` assessment-aware candidate recall experiment
- `H3` catalog-unspecified recovery experiment
- `H4` state-specific candidate balancing experiment

Current roadmap state:

- `G1` is accepted
- `G2` is rejected
- `G3` is rejected
- `G4` is rejected
- `H1` is ready
- no experiment-changing task is currently in progress

## Recommended Next Step

Start `H1` only.

`H1` should be a reporting / diagnostic checkpoint that separates hard rows into:

- gold catalog row absent from top-10, meaning candidate recall failed
- gold catalog row present in top-10 but not selected, meaning rerank selection failed

Report those buckets for:

- `Assessment`
- `catalog_unspecified`
- `adoption_state_high_risk`
- `catalog_state_specific_expected`

Do not change behavior until the `H1` audit points to the next single-variable experiment.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-16-v049-h-series-defined-handoff.md`.
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
> - `G2` is rejected
> - `G3` is rejected
> - `G4` is rejected
> - `H1` is ready
> - no experiment-changing task is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, and the `H` experiment-set definition,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, shortlist, model, reasoning, and `F2` rerank prompt baselines,
> 4. start exactly `H1`,
> 5. make `H1` diagnostic/reporting-only,
> 6. separate top-10 recall failures from rerank selection failures for the required hard slices,
> 7. leave a tracker update, a review note, and a fresh handoff before ending.
