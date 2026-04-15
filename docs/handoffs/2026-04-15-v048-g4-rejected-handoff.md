# Curriculum Matcher Handoff v048

- Date: `2026-04-15`
- Milestone: `G4` measured and rejected
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

- resumed the live `G4` batch after syncing the branch to GitHub
- downloaded the completed output and error artifacts
- scored the run
- compared `G4` against the canonical accepted `F2` row-matched baseline on the shared rows
- rejected `G4`
- restored `src/curriculum_matcher/llm_rerank.py` and `tests/test_llm_rerank.py` to the accepted `F2` baseline after measurement

Primary review note:

- `docs/analysis/2026-04-14-series-exposure-review.md`

Tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## G4 Decision Summary

Candidate run:

- `historical-top10-gpt54mini-medium-g4-no-series-batch`

Shared-row comparison versus the canonical accepted `F2` baseline:

- shared rows: `454`
- baseline-only rows: `1`
- selection rate `+0.0055`
- all-row top-1 `-0.0018`
- selected-row top-1 `-0.0077`

Targeted slice movement:

- `catalog_state_specific_expected`: top-1 `-0.1154`
- `catalog_unspecified`: top-1 `+0.0000`
- assessment rows: top-1 `+0.0000`
- wrong-top1-but-gold-in-shortlist rows: top-1 `-0.0071`

Operational notes:

- the candidate introduced `2` repaired selected ids
- the single failed request was a `503` `server_is_overloaded` response

Why `G4` was rejected:

- all-row top-1 regressed
- selected-row top-1 regressed
- the family-sensitive target slices did not improve
- the candidate added repaired-id risk without delivering compensating accuracy gains

## Current Roadmap State

- `G1` is accepted
- `G2` is rejected
- `G3` is rejected
- `G4` is rejected
- no experiment is currently in progress

## Recommended Next Step

No active `G`-series experiment remains. If work continues, define a new experiment set before changing rerank behavior again.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-15-v048-g4-rejected-handoff.md`.
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
> - no experiment is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, the `G4` review note, and the most recent experiment-set definition,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, shortlist, model, reasoning, and `F2` rerank prompt baselines,
> 4. define the next experiment set before making any new behavioral change,
> 5. keep experiment-changing work serial and one-variable-at-a-time,
> 6. leave a tracker update, a review note, and a fresh handoff before ending.
