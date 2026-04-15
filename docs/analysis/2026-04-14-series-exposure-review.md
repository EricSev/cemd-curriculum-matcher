# G4 Candidate Series Exposure Review

- Date: `2026-04-15`
- Task: `G4` candidate series exposure experiment
- Decision: `rejected`
- Accepted comparison baseline: `historical-top10-gpt54mini-medium-f2-no-score-canonical`
- Candidate run: `historical-top10-gpt54mini-medium-g4-no-series-batch`
- Candidate prompt summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g4-no-series-prompts-summary.json`
- Candidate scored summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g4-no-series-batch-scored-summary.json`
- Candidate pipeline summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g4-no-series-batch-pipeline-summary.json`
- Comparison artifact: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g4-no-series-vs-f2-comparison.json`
- Slice artifact: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g4-no-series-slice-deltas.json`

## Single Variable

- removed only candidate `series` from `SHORTLIST_CANDIDATES` in `src/curriculum_matcher/llm_rerank.py`

Held fixed:

- matcher baseline `C1 + C2 + C5`
- retrieval baseline `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline `F2` no-score candidate payload

## Comparison Scope

The `G4` batch completed with `454` scored rows and `1` failed request, so the clean comparison was done on the `454` shared `selection_identifier` rows present in both scored CSVs.

Row-set summary:

- baseline rows: `455`
- candidate rows: `454`
- shared rows: `454`
- baseline-only rows: `1`
- candidate-only rows: `0`

The failed request was a `503` `server_is_overloaded` response, not a prompt-format or validation failure.

## Overall Delta On Shared Rows

| Metric | Baseline | G4 Candidate | Delta |
| --- | --- | --- | --- |
| row_count | 454 | 454 | 0 |
| llm_selection_rate | 0.5055 | 0.5110 | +0.0055 |
| llm_top1_accuracy_on_all_rows | 0.1978 | 0.1960 | -0.0018 |
| llm_top1_accuracy_on_selected_rows | 0.3913 | 0.3836 | -0.0077 |
| repaired_selected_id_count | 0 | 2 | +2.0 |
| invalid_selected_id_count | 0 | 0 | +0.0 |

Interpretation:

- removing `series` made the model select slightly more often
- that extra selection rate did not convert into better quality, because both protected top-1 metrics regressed
- the candidate also introduced `2` repaired selected ids, which is a new operational liability versus the accepted baseline

## Required Slice Readout

- assessment rows: top-1 `+0.0000`, selection rate `+0.0147`
- `assessment_short_or_acronym_title`: top-1 `+0.0000`, selection rate `+0.0370`
- `catalog_state_specific_expected`: top-1 `-0.1154`, selection rate `-0.0385`
- `catalog_unspecified`: top-1 `+0.0000`, selection rate `+0.0417`
- wrong-top1-but-gold-in-shortlist rows: top-1 `-0.0071`, selection rate `+0.0071`

What did not improve:

- the key near-family state-specific slice regressed sharply
- wrong-top1-but-gold-in-shortlist recovery regressed slightly instead of improving
- assessment-heavy rows became a little more selective but did not gain any top-1 recovery

Why that matters:

- `G4` was meant to test whether removing family-level `series` context would reduce near-neighbor anchoring
- instead, the candidate did not create a measurable recovery lift and weakened one of the most important disambiguation slices
- the repaired-id regression makes the simpler payload less robust in practice

## Keep / Reject Decision

- reject `G4`
- keep the accepted `F2` no-score rerank prompt baseline unchanged

Why:

- all-row top-1 regressed by `-0.0018`
- selected-row top-1 regressed by `-0.0077`
- the family-sensitive target slices did not improve
- the candidate introduced `2` repaired selected ids and also lost `1` row to a server overload

## Baseline Update

- no behavioral baseline change
- `src/curriculum_matcher/llm_rerank.py` and `tests/test_llm_rerank.py` have been restored to the accepted `F2` prompt baseline after measurement
- accepted rerank baseline remains the `F2` no-score candidate payload
- accepted comparison-contract baseline remains:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## Recommended Next Step

No active `G`-series experiment remains after the `G4` closeout. The next step should be to define a new experiment set before making further behavioral changes.
