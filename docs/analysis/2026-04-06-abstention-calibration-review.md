# F4 Abstention Wording Calibration Review

- Date: `2026-04-06`
- Task: `F4` abstention wording calibration
- Decision: `rejected`
- Accepted baseline run before this task: `historical-top10-gpt54mini-medium-f2-no-score-batch`
- Candidate run: `historical-top10-gpt54mini-medium-f4-abstention-batch`
- Candidate prompt summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f4-abstention-prompts-summary.json`
- Candidate scored summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f4-abstention-batch-scored-summary.json`
- Comparison artifact: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f4-abstention-vs-f2-common424-comparison.json`
- Slice artifact: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f4-abstention-slice-report.json`

## Single Variable

- changed only the abstention guidance sentence in `SYSTEM_PROMPT`
- held fixed:
  - accepted `F2` no-score prompt baseline
  - matcher baseline `C1 + C2 + C5`
  - retrieval baseline `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
  - shortlist `10`
  - model `gpt-5.4-mini`
  - reasoning `medium`

## Comparison Scope

The accepted `F2` scored output and the `F4` candidate scored output aligned cleanly on the same `455` prompt rows.

Row-set alignment:

- shared rows: `455`
- baseline-only rows: `0`
- candidate-only rows: `0`

## Overall Delta On Shared Rows

| Metric | Baseline | F4 Candidate | Delta |
| --- | --- | --- | --- |
| row_count | 455 | 455 | 0 |
| llm_selection_rate | 0.5055 | 0.4308 | -0.0747 |
| llm_top1_accuracy_on_all_rows | 0.1978 | 0.1912 | -0.0066 |
| llm_top1_accuracy_on_selected_rows | 0.3913 | 0.4439 | +0.0526 |
| repaired_selected_id_count | 0 | 0 | +0.0 |
| invalid_selected_id_count | 0 | 0 | +0.0 |

Interpretation:

- the stricter abstention wording made the model materially more selective
- once the model did select, it was more accurate
- but it gave back too much overall recovery, so all-row top-1 regressed

## Required Slice Readout

- `Assessment`: top-1 `-0.0074`, selection rate `-0.0368`
- `catalog_state_specific_expected`: top-1 `-0.0769`, selection rate `-0.0769`
- `adoption_state_high_risk`: top-1 `-0.0061`, selection rate `-0.0485`
- `catalog_unspecified`: top-1 `+0.0208`, selection rate `-0.1458`
- `assessment_short_or_acronym_title`: top-1 `-0.0123`, selection rate `-0.0370`
- wrong-top1-but-gold-in-shortlist rows: top-1 `-0.0214`, selection rate `-0.0786`

What improved:

- `catalog_unspecified` improved modestly
- selected-row precision improved substantially overall

Why that still was not enough:

- the main state-specific target slice regressed sharply
- wrong-top1-but-gold-in-shortlist recovery regressed, which means the stricter wording abstained away too many recoverable wins
- roadmap acceptance rule requires improvement on both all-row and selected-row top-1 without a selection-rate regression

## Keep / Reject Decision

- reject `F4`
- keep the accepted `F2` no-score prompt baseline

Why:

- all-row top-1 regressed by `-0.0066`
- selection rate regressed materially by `-0.0747`
- the stricter abstention wording was too conservative for the accepted top-10 rerank operating point

## Baseline Update

- no baseline change
- `src/curriculum_matcher/llm_rerank.py` has been restored to the accepted `F2` no-score prompt baseline after measurement
- accepted rerank baseline remains:
  - suppress raw matcher scores in the rerank prompt payload
  - shortlist `10`
  - model `gpt-5.4-mini`
  - reasoning `medium`

## Recommended Next Step

No active `F`-series experiment remains after the `F4` closeout. The next session should define a new experiment set before making further behavioral changes.
