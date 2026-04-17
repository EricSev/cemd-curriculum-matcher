# Curriculum Matcher Handoff v056

- Date: `2026-04-17`
- Milestone: `I` series closed
- Operator surface: keep using `src/curriculum_matcher/app.py`
- Workflow: diagnosis first, then benchmark-driven one-variable experiments

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
- `I4` taxonomy-informed retrieval candidate

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline: `F2` no-score candidate payload

Accepted comparison-contract baseline:

- use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## What This Session Did

- completed the full `I` series
- kept `I1` and `I2` diagnostic-only
- kept `I3` as a diagnostic label-normalization candidate
- measured `I4` as an opt-in retrieval candidate, rejected it, and restored the accepted code baseline after measurement
- wrote the I-series status report:
  - `docs/analysis/2026-04-17-i-series-status-report.md`
- updated the live tracker:
  - `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## I-Series Decisions

- `I1`: accepted diagnostic taxonomy audit
- `I2`: accepted diagnostic catalog-label consistency audit
- `I3`: accepted diagnostic label-normalization candidate
- `I4`: rejected taxonomy-informed retrieval candidate

## I4 Decision Summary

Candidate:

- `i4_taxonomy_recall`
- opt-in retrieval experiment for a small set of clean assessment retrieval-miss patterns

Overall deltas versus accepted char n-gram top-10 baseline:

- top-1: `+0.0000`
- top-3: `+0.0000`
- hit@10: `+0.0000`
- MRR: `+0.0000`
- nDCG@10: `+0.0000`

Row movement:

- changed shortlist rows: `0`
- top-1 gains / losses: `0 / 0`
- top-10 gains / losses: `0 / 0`

Why `I4` was rejected:

- the opt-in retrieval candidate produced no shortlist movement
- no target slice improved
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
- `I1` is accepted
- `I2` is accepted
- `I3` is accepted as diagnostic
- `I4` is rejected
- no experiment-changing task is currently in progress

## Recommended Next Step

Define a new experiment set before changing behavior again.

The best next focus is not broad retrieval expansion. Use the `I3` split to make reporting/prompt-context terminology clearer, and separately review ambiguous assessment historical gold labels before new runtime changes.
