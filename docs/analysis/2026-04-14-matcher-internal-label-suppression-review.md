# G2 Matcher-Internal Label Suppression Review

- Date: `2026-04-14`
- Task: `G2` matcher-internal confidence label suppression
- Decision: `rejected`
- Accepted comparison baseline: `historical-top10-gpt54mini-medium-f2-no-score-canonical`
- Candidate run: `historical-top10-gpt54mini-medium-g2-internal-labels-batch`
- Candidate prompt summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g2-internal-labels-prompts-summary.json`
- Candidate scored summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g2-internal-labels-batch-scored-summary.json`
- Candidate pipeline summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g2-internal-labels-batch-pipeline-summary.json`
- Comparison artifact: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g2-internal-labels-vs-f2-comparison.json`

## Single Variable

- removed only `confidence_band` and `match_selected_strategy` from `DISTRICT_ROW` in `src/curriculum_matcher/llm_rerank.py`

Held fixed:

- matcher baseline `C1 + C2 + C5`
- retrieval baseline `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline `F2` no-score candidate payload

## Comparison Scope

The canonical accepted `F2` baseline has `455` prompt/scored rows.

The `G2` batch completed with `450` scored rows and `5` failed requests, so the clean comparison was done on the `450` shared `selection_identifier` rows present in both scored CSVs.

Row-set summary:

- baseline rows: `455`
- candidate rows: `450`
- shared rows: `450`
- baseline-only rows: `5`
- candidate-only rows: `0`

## Overall Delta On Shared Rows

| Metric | Baseline | G2 Candidate | Delta |
| --- | --- | --- | --- |
| row_count | 450 | 450 | 0 |
| llm_selection_rate | 0.5022 | 0.4911 | -0.0111 |
| llm_top1_accuracy_on_all_rows | 0.1978 | 0.1978 | +0.0000 |
| llm_top1_accuracy_on_selected_rows | 0.3938 | 0.4027 | +0.0089 |
| repaired_selected_id_count | 0 | 0 | +0 |
| invalid_selected_id_count | 0 | 0 | +0 |

Interpretation:

- suppressing matcher-internal labels made the model slightly more selective
- selected-row accuracy improved a little on shared rows, but not enough to clear the `+0.01` promotion threshold
- all-row top-1 stayed flat
- the selection-rate regression violates the current cost-aware promotion rule

## Required Slice Readout

- `catalog_state_specific_expected`: top-1 `-0.0090`, selection rate `-0.0090`
- wrong-top1-but-gold-in-shortlist rows: top-1 `+0.0119`, selection rate `-0.0119`
- assessment rows: top-1 `+0.0000`, selection rate `-0.0189`

What improved:

- wrong-top1-but-gold-in-shortlist rows improved slightly once selected
- selected-row precision moved in the right direction overall

Why that still was not enough:

- all-row recovery did not improve at all on shared rows
- selected-row precision improved by only `+0.0089`, still below the standing `+0.01` threshold
- `catalog_state_specific_expected` regressed
- the candidate batch also lost `5` rows to failed requests, which weakens the operating contract versus the canonical accepted baseline

## Keep / Reject Decision

- reject `G2`
- keep the accepted `F2` no-score rerank prompt baseline unchanged

Why:

- the candidate did not improve all-row top-1
- the selected-row lift did not clear the promotion threshold
- selection rate regressed
- row coverage also regressed because the completed batch returned `450` scored rows instead of the baseline `455`

## Baseline Update

- no behavioral baseline change
- accepted rerank baseline remains the `F2` no-score candidate payload
- accepted comparison-contract baseline remains:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## Recommended Next Step

If work continues in this roadmap, restore `src/curriculum_matcher/llm_rerank.py` to the accepted `F2` prompt baseline and then lift exactly `G3`.
