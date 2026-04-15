# G3 Derived Ambiguity Label Suppression Review

- Date: `2026-04-14`
- Task: `G3` derived ambiguity label suppression
- Decision: `rejected`
- Accepted comparison baseline: `historical-top10-gpt54mini-medium-f2-no-score-canonical`
- Candidate run: `historical-top10-gpt54mini-medium-g3-derived-labels-batch`
- Candidate prompt summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-prompts-summary.json`
- Candidate scored summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-batch-scored-summary.json`
- Candidate pipeline summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-batch-pipeline-summary.json`
- Comparison artifact: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-vs-f2-comparison.json`
- Slice artifact: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-slice-deltas.json`

## Single Variable

- removed only `usage_ambiguity`, `state_specific_risk`, `placeholder_mapping`, and `assessment_slice` from `DISTRICT_ROW` in `src/curriculum_matcher/llm_rerank.py`

Held fixed:

- matcher baseline `C1 + C2 + C5`
- retrieval baseline `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline `F2` no-score candidate payload

## Comparison Scope

The original `G3` batch looked stuck while it was still processing, but it eventually completed cleanly with full row coverage.

Row-set alignment:

- shared rows: `455`
- baseline-only rows: `0`
- candidate-only rows: `0`

## Overall Delta On Shared Rows

| Metric | Baseline | G3 Candidate | Delta |
| --- | --- | --- | --- |
| row_count | 455 | 455 | 0 |
| llm_selection_rate | 0.5055 | 0.5099 | +0.0044 |
| llm_top1_accuracy_on_all_rows | 0.1978 | 0.1846 | -0.0132 |
| llm_top1_accuracy_on_selected_rows | 0.3913 | 0.3621 | -0.0292 |
| repaired_selected_id_count | 0 | 0 | +0.0 |
| invalid_selected_id_count | 0 | 0 | +0.0 |

Interpretation:

- removing the derived ambiguity labels made the model slightly more likely to select
- despite that higher selection rate, both protected quality metrics regressed materially
- the candidate did not create a new invalid-id failure mode, but it also did not recover more correct matches

## Required Slice Readout

- `catalog_state_specific_expected`: top-1 `-0.1154`, selection rate `+0.0000`
- `adoption_state_high_risk`: top-1 `-0.0182`, selection rate `+0.0121`
- `catalog_unspecified`: top-1 `-0.0208`, selection rate `+0.0625`
- assessment rows: top-1 `+0.0000`, selection rate `-0.0074`
- `assessment_short_or_acronym_title`: top-1 `+0.0000`, selection rate `+0.0247`
- wrong-top1-but-gold-in-shortlist rows: top-1 `-0.0429`, selection rate `+0.0071`

What did not improve:

- the main state-specific target slice regressed sharply
- the placeholder target slice also regressed while becoming more selective in the wrong direction
- assessment-heavy rows stayed flat on top-1, so removing the labels did not unlock extra assessment recovery
- wrong-top1-but-gold-in-shortlist recovery regressed, which is especially costly because that slice is where reranking can still add value

Why that matters:

- `G3` was intended to test whether analysis-layer ambiguity labels were over-steering the model
- instead, removing them reduced top-1 recovery on the rows where the accepted baseline was already doing useful ambiguity work
- the extra selection rate did not translate into better outcomes

## Keep / Reject Decision

- reject `G3`
- keep the accepted `F2` no-score rerank prompt baseline unchanged

Why:

- all-row top-1 regressed by `-0.0132`
- selected-row top-1 regressed by `-0.0292`
- the key state-specific, placeholder, and recovery slices did not improve
- roadmap default says mixed or degrading results should be rejected

## Baseline Update

- no behavioral baseline change
- `src/curriculum_matcher/llm_rerank.py` and `tests/test_llm_rerank.py` have been restored to the accepted `F2` prompt baseline after measurement
- accepted rerank baseline remains the `F2` no-score candidate payload
- accepted comparison-contract baseline remains:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## Recommended Next Step

If work continues in this roadmap, lift exactly `G4` next and remove only candidate `series` from `SHORTLIST_CANDIDATES`.
