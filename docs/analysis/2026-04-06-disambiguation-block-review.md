# F3 Candidate Disambiguation Block Review

- Date: `2026-04-06`
- Task: `F3` candidate disambiguation block experiment
- Decision: `rejected`
- Accepted baseline run before this task: `historical-top10-gpt54mini-medium-f2-no-score-batch`
- Candidate run: `historical-top10-gpt54mini-medium-f3-disambiguation-batch`
- Candidate prompt summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f3-disambiguation-prompts-summary.json`
- Candidate scored summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f3-disambiguation-batch-scored-summary.json`
- Comparison artifact: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f3-disambiguation-vs-f2-common424-comparison.json`
- Slice artifact: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f3-disambiguation-slice-report.json`

## Single Variable

- added a compact `CANDIDATE_DISAMBIGUATION` block to the rerank prompt using only already-available row context and candidate metadata
- held fixed:
  - accepted `F2` no-score prompt baseline
  - matcher baseline `C1 + C2 + C5`
  - retrieval baseline `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
  - shortlist `10`
  - model `gpt-5.4-mini`
  - reasoning `medium`

## Comparison Scope

Unlike the earlier historical baseline comparison, the accepted `F2` scored output and the `F3` candidate scored output aligned cleanly on the same `455` prompt rows.

Row-set alignment:

- shared rows: `455`
- baseline-only rows: `0`
- candidate-only rows: `0`

## Overall Delta On Shared Rows

| Metric | Baseline | F3 Candidate | Delta |
| --- | --- | --- | --- |
| row_count | 455 | 455 | 0 |
| llm_selection_rate | 0.5055 | 0.5033 | -0.0022 |
| llm_top1_accuracy_on_all_rows | 0.1978 | 0.1956 | -0.0022 |
| llm_top1_accuracy_on_selected_rows | 0.3913 | 0.3886 | -0.0027 |
| repaired_selected_id_count | 0 | 0 | +0.0 |
| invalid_selected_id_count | 0 | 1 | +1.0 |

Interpretation:

- the added block did not produce a measurable accuracy gain
- both all-row and selected-row top-1 slipped slightly
- the candidate also introduced one invalid selected id, which is a new failure mode versus the accepted `F2` baseline

## Required Slice Readout

- `Assessment`: top-1 `-0.0147`, selection rate `-0.0147`
- `catalog_state_specific_expected`: top-1 `+0.0000`, selection rate `+0.0000`
- `adoption_state_high_risk`: top-1 `-0.0061`, selection rate `+0.0121`
- `catalog_unspecified`: top-1 `+0.0000`, selection rate `+0.0208`
- `assessment_short_or_acronym_title`: top-1 `-0.0123`, selection rate `+0.0247`
- wrong-top1-but-gold-in-shortlist rows: top-1 `-0.0071`, selection rate `+0.0000`

What did not improve:

- the targeted state-specific slice stayed flat instead of improving further
- the assessment-heavy and acronym-heavy slices regressed
- the main `wrong_top1_but_gold_in_shortlist` recovery slice also regressed

Why that matters:

- `F3` was supposed to sharpen ambiguity decisions after the accepted `F2` no-score change
- instead, it added prompt complexity without creating a measurable recovery benefit
- the invalid-id regression makes the added block less safe than the simpler accepted baseline

## Keep / Reject Decision

- reject `F3`
- keep the accepted `F2` no-score prompt baseline

Why:

- both top-1 metrics regressed
- the candidate did not improve the required ambiguity slices overall
- it introduced one invalid selected id
- roadmap default says mixed or degrading results should be rejected

## Baseline Update

- no baseline change
- `src/curriculum_matcher/llm_rerank.py` has been restored to the accepted `F2` no-score prompt baseline after measurement
- accepted rerank baseline remains:
  - suppress raw matcher scores in the rerank prompt payload
  - shortlist `10`
  - model `gpt-5.4-mini`
  - reasoning `medium`

## Recommended Next Step

If the roadmap continues, the next serial experiment should be `F4` abstention wording calibration, measured against the unchanged accepted `F2` no-score prompt baseline.
