# Curriculum Matcher Handoff v053

- Date: `2026-04-16`
- Milestone: `H4` measured and rejected; `H` series closed
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
- `H2` assessment-aware candidate recall
- `H3` catalog-unspecified recovery
- `H4` state-specific candidate balancing

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline: `F2` no-score candidate payload

Accepted comparison-contract baseline:

- use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## What This Session Did

- started exactly `H4`
- changed only an opt-in retrieval experiment path named `state_balance`
- verified focused unit tests passed during measurement:
  - `PYTHONPATH=src ./.venv/bin/python -m unittest tests/test_matcher_core.py`
  - result: `19` tests passed
- ran the representative benchmark before any LLM batch
- compared H4 against the accepted top-10 char n-gram baseline
- rejected H4
- restored the accepted code baseline after measurement

Primary review note:

- `docs/analysis/2026-04-16-state-balance-review.md`

Candidate artifacts:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h4_state_balance_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h4_state_balance_records.csv`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h4_state_balance_comparison.json`

Tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## H4 Decision Summary

Candidate:

- `state_balance`
- for `CA`, `FL`, and `TX` rows, added extra stage-1 candidates from catalog rows where `state_specific_version` is true and product or series text matches the row state

Overall deltas versus accepted char n-gram top-10 baseline:

- top-1: `-0.0010`
- top-3: `-0.0010`
- hit@10: `+0.0000`
- MRR: `-0.0006`
- nDCG@10: `-0.0005`

Row movement:

- changed shortlist rows: `13`
- changed metric rows: `4`
- top-1 gains / losses: `0 / 1`
- top-3 gains / losses: `0 / 1`
- top-10 gains / losses: `0 / 0`

Required slice movement:

- `adoption_state_high_risk` top-1 `-0.0028`, top-3 `-0.0029`, hit@10 `+0.0000`
- `catalog_state_specific_expected` top-1 `+0.0000`, top-3 `+0.0000`, hit@10 `+0.0000`
- assessment rows inside adoption states changed `4` shortlists but produced no top-10 gains
- `FL` had one top-1 loss
- `TX` had one top-3 loss

Why `H4` was rejected:

- no top-10 recovery gains appeared
- target state-specific slices did not improve
- overall top-1, top-3, MRR, and nDCG regressed slightly
- no LLM batch was justified

## Current Roadmap State

- `G1` is accepted
- `G2` is rejected
- `G3` is rejected
- `G4` is rejected
- `H1` is accepted
- `H2` is rejected
- `H3` is rejected
- `H4` is rejected
- no experiment-changing task is currently in progress

## Recommended Next Step

Define a new experiment set before changing behavior again.

The `H` series suggests that simple metadata-gated stage-1 expansion is not enough. A next phase should probably focus on error taxonomy or catalog-label structure before another retrieval tweak.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-16-v053-h4-rejected-h-series-closed-handoff.md`.
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
> - `H1` is accepted
> - `H2` is rejected
> - `H3` is rejected
> - `H4` is rejected
> - no experiment-changing task is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, the `H` experiment-set definition, and the H1-H4 review notes,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, shortlist, model, reasoning, and `F2` rerank prompt baselines,
> 4. define a new experiment set before making any new behavior change,
> 5. keep future experiment-changing work serial and one-variable-at-a-time,
> 6. leave a tracker update, a review note, and a fresh handoff before ending.
