# Curriculum Matcher Handoff v052

- Date: `2026-04-16`
- Milestone: `H3` measured and rejected
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

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline: `F2` no-score candidate payload

Accepted comparison-contract baseline:

- use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## What This Session Did

- started exactly `H3`
- changed only an opt-in retrieval experiment path named `unspecified_recall`
- verified focused unit tests passed during measurement:
  - `PYTHONPATH=src ./.venv/bin/python -m unittest tests/test_matcher_core.py`
  - result: `19` tests passed
- ran the representative benchmark before any LLM batch
- compared H3 against the accepted top-10 char n-gram baseline
- rejected H3
- restored the accepted code baseline after measurement

Primary review note:

- `docs/analysis/2026-04-16-unspecified-recall-review.md`

Candidate artifacts:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h3_unspecified_recall_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h3_unspecified_recall_records.csv`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h3_unspecified_recall_comparison.json`

Tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## H3 Decision Summary

Candidate:

- `unspecified_recall`
- added extra stage-1 candidates from catalog rows whose product or series looked `Unspecified`, no-information, or district-created before the existing scorer produced the final top-10

Overall deltas versus accepted char n-gram top-10 baseline:

- top-1: `-0.0030`
- top-3: `+0.0000`
- hit@10: `+0.0000`
- MRR: `-0.0016`
- nDCG@10: `-0.0012`

Row movement:

- changed shortlist rows: `35`
- changed metric rows: `11`
- top-1 gains / losses: `2 / 5`
- top-3 gains / losses: `0 / 0`
- top-10 gains / losses: `0 / 0`

Required slice movement:

- `catalog_unspecified` top-1 `+0.0000`, hit@10 `+0.0000`
- `sparse` top-1 `-0.0076`, hit@10 `+0.0000`
- `medium` top-1 `+0.0000`, hit@10 `+0.0000`
- `adoption_state_high_risk` top-1 `-0.0028`, hit@10 `+0.0000`

Why `H3` was rejected:

- the target `catalog_unspecified` slice did not improve
- overall top-1, MRR, and nDCG regressed
- top-1 losses outnumbered gains
- no top-10 recovery gains appeared, so no LLM batch was justified

## Current Roadmap State

- `G1` is accepted
- `G2` is rejected
- `G3` is rejected
- `G4` is rejected
- `H1` is accepted
- `H2` is rejected
- `H3` is rejected
- `H4` is not started
- no experiment-changing task is currently in progress

## Recommended Next Step

Start `H4` only if continuing the `H` series.

`H4` should focus on state-specific candidate balancing. Avoid broad candidate expansion, because both `H2` and `H3` moved shortlists without producing top-10 recovery.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-16-v052-h3-rejected-handoff.md`.
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
> - `H4` is not started
> - no experiment-changing task is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, the `H` experiment-set definition, and the H1-H3 review notes,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, shortlist, model, reasoning, and `F2` rerank prompt baselines except for the single H4 candidate-recall variable,
> 4. start exactly `H4`,
> 5. run the representative benchmark before any LLM batch,
> 6. decide whether H4 produces enough shortlist movement to justify a rerank batch,
> 7. leave a tracker update, a review note, and a fresh handoff before ending.
