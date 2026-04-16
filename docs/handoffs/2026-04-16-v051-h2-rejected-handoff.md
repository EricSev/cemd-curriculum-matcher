# Curriculum Matcher Handoff v051

- Date: `2026-04-16`
- Milestone: `H2` measured and rejected
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

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline: `F2` no-score candidate payload

Accepted comparison-contract baseline:

- use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## What This Session Did

- committed the `H` series definition and `H1` diagnostic checkpoint:
  - commit `7631500`
- started exactly `H2`
- changed only an opt-in retrieval experiment path named `assessment_recall`
- verified focused unit tests passed during measurement:
  - `PYTHONPATH=src ./.venv/bin/python -m unittest tests/test_matcher_core.py`
  - result: `19` tests passed
- ran the representative benchmark before any LLM batch
- compared H2 against the accepted top-10 char n-gram baseline
- rejected H2
- restored the accepted code baseline after measurement

Primary review note:

- `docs/analysis/2026-04-16-assessment-recall-review.md`

Candidate artifacts:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h2_assessment_recall_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h2_assessment_recall_records.csv`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h2_assessment_recall_comparison.json`

Tracker:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

## H2 Decision Summary

Candidate:

- `assessment_recall`
- for rows where `product_type_usage == Assessment`, added extra stage-1 candidates from assessment-like catalog rows before the existing scorer produced the final top-10

Overall deltas versus accepted char n-gram top-10 baseline:

- top-1: `+0.0000`
- top-3: `+0.0000`
- hit@10: `+0.0000`
- MRR: `+0.0000`
- nDCG@10: `+0.0000`

Row movement:

- changed shortlist rows: `51`
- changed metric rows: `16`
- top-1 gains / losses: `0 / 0`
- top-3 gains / losses: `0 / 0`
- top-10 gains / losses: `0 / 0`

Required slice movement:

- `Assessment` top-1 `+0.0000`, hit@10 `+0.0000`
- `assessment_short_or_acronym_title` top-1 `+0.0000`, hit@10 `+0.0000`
- `assessment_state_specific_expected` top-1 `+0.0000`, hit@10 `+0.0000`
- `assessment_publisher_missing` top-1 `+0.0000`, hit@10 `+0.0000`
- `catalog_unspecified` changed shortlist rows: `0`

Why `H2` was rejected:

- no overall shortlist metric improved
- no required assessment slice improved
- shortlist movement did not recover any additional gold rows into top-10
- no LLM batch was justified

## Current Roadmap State

- `G1` is accepted
- `G2` is rejected
- `G3` is rejected
- `G4` is rejected
- `H1` is accepted
- `H2` is rejected
- `H3` is not started
- no experiment-changing task is currently in progress

## Recommended Next Step

Start `H3` only if continuing the `H` series.

`H3` should focus on `catalog_unspecified` recovery. H1 showed `37 / 56` `catalog_unspecified` rows absent from top-10, and H2 did not move that slice at all.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-16-v051-h2-rejected-handoff.md`.
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
> - `H3` is not started
> - no experiment-changing task is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, the `H` experiment-set definition, the `H1` shortlist failure audit, and the `H2` rejection note,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, shortlist, model, reasoning, and `F2` rerank prompt baselines except for the single H3 candidate-recall variable,
> 4. start exactly `H3`,
> 5. run the representative benchmark before any LLM batch,
> 6. decide whether H3 produces enough shortlist movement to justify a rerank batch,
> 7. leave a tracker update, a review note, and a fresh handoff before ending.
