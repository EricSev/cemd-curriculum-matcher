# F2 Raw Matcher-Score Exposure Review

- Date: `2026-04-06`
- Task: `F2` raw matcher-score exposure experiment
- Decision: `accepted`
- Accepted baseline run before this task: `historical-top10-gpt54mini-medium-batch`
- Candidate run: `historical-top10-gpt54mini-medium-f2-no-score-batch`
- Candidate prompt summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-prompts-summary.json`
- Candidate scored summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-batch-scored-summary.json`
- Comparison artifact: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-vs-baseline-common424-comparison.json`
- Slice artifact: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-slice-report.json`

## Single Variable

- removed raw candidate `score` from the `SHORTLIST_CANDIDATES` payload in `src/curriculum_matcher/llm_rerank.py`
- held fixed:
  - matcher baseline `C1 + C2 + C5`
  - retrieval baseline `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
  - shortlist `10`
  - model `gpt-5.4-mini`
  - reasoning `medium`
  - candidate order by rank

## Comparison Scope

The candidate prompt pack used the current error-row workflow and produced `455` prompts.

As with `F1`, the accepted historical baseline scored artifact does not align one-to-one with that prompt set, so the clean comparison was done on the `424` shared `selection_identifier` rows present in both scored outputs.

Shared-row artifact summaries:

- baseline overlap summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-batch-common424-scored-summary.json`
- candidate overlap summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-batch-common424-scored-summary.json`

Row-set mismatch:

- baseline-only rows: `48`
- candidate-only rows: `31`

## Overall Delta On Shared Rows

| Metric | Baseline | F2 Candidate | Delta |
| --- | --- | --- | --- |
| row_count | 424 | 424 | 0 |
| llm_selection_rate | 0.4858 | 0.5236 | +0.0378 |
| llm_top1_accuracy_on_all_rows | 0.1698 | 0.2028 | +0.0330 |
| llm_top1_accuracy_on_selected_rows | 0.3495 | 0.3874 | +0.0379 |
| repaired_selected_id_count | 0 | 0 | +0.0 |
| invalid_selected_id_count | 0 | 0 | +0.0 |

Interpretation:

- removing the raw score anchor improved both all-row and selected-row top-1 materially
- the candidate selected somewhat more often, but the larger gain came with clearly better precision once it selected
- the candidate cleared the standing promotion rule used by the comparison helper

## Required Slice Readout

- `Assessment`: top-1 `+0.0583`, selection rate `+0.0917`
- `catalog_state_specific_expected`: top-1 `+0.3333`, selection rate `+0.2500`
- `adoption_state_high_risk`: top-1 `+0.0255`, selection rate `+0.0191`
- `catalog_unspecified`: top-1 `+0.0213`, selection rate `-0.0426`
- `assessment_short_or_acronym_title`: top-1 `+0.0000`, selection rate `+0.0000`
- wrong-top1-but-gold-in-shortlist rows: top-1 `+0.1111`, selection rate `+0.0889`

What improved:

- state-specific ambiguity rows improved sharply
- wrong-top1-but-gold-in-shortlist recovery improved strongly, which was the main `F2` target
- selected-row accuracy increased by `+0.0379`, showing that the model was not only selecting more often, but also selecting better

Residual caution:

- `assessment_short_or_acronym_title` stayed flat
- some of the overall lift still came with a modest rise in selection rate, so later prompt experiments should avoid stacking further encouragement to select unless they also preserve precision

## Keep / Accept Decision

- accept `F2`
- promote the no-score prompt policy to the rerank prompt baseline

Why:

- all-row top-1 improved by `+0.0330`
- selected-row top-1 improved by `+0.0379`
- invalid and repaired id counts stayed flat at `0`
- the candidate produced the intended recovery win on rows where the gold was already in the shortlist

## Baseline Update

- prompt baseline now suppresses raw matcher scores in the rerank prompt payload
- `src/curriculum_matcher/llm_rerank.py` should keep the no-score candidate payload
- accepted rerank baseline remains:
  - shortlist `10`
  - model `gpt-5.4-mini`
  - reasoning `medium`

## Recommended Next Step

If the roadmap continues, the next serial experiment should be `F3` candidate disambiguation block, measured against the newly accepted `F2` prompt baseline.
