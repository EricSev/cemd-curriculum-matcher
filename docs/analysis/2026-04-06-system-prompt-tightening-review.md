# F1 System Prompt Tightening Review

- Date: `2026-04-06`
- Task: `F1` system prompt tightening
- Decision: `rejected`
- Accepted baseline run: `historical-top10-gpt54mini-medium-batch`
- Candidate run: `historical-top10-gpt54mini-medium-f1-system-prompt-batch`
- Candidate prompt summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f1-system-prompt-prompts-summary.json`
- Candidate scored summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f1-system-prompt-batch-scored-summary.json`
- Comparison artifact: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f1-system-prompt-vs-baseline-common424-comparison.json`
- Slice artifact: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f1-system-prompt-slice-report.json`

## Single Variable

- changed `SYSTEM_PROMPT` wording only in `src/curriculum_matcher/llm_rerank.py`
- held fixed:
  - matcher baseline `C1 + C2 + C5`
  - retrieval baseline `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
  - shortlist `10`
  - model `gpt-5.4-mini`
  - reasoning `medium`

## Comparison Scope

The candidate prompt pack used the current error-row workflow and produced `455` prompts.

The accepted historical baseline scored artifact does not align one-to-one with that prompt set, so the clean comparison was done on the `424` shared `selection_identifier` rows present in both scored outputs.

Shared-row artifact summaries:

- baseline overlap summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-batch-common424-scored-summary.json`
- candidate overlap summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f1-system-prompt-batch-common424-scored-summary.json`

Row-set mismatch:

- baseline-only rows: `48`
- candidate-only rows: `31`

## Overall Delta On Shared Rows

| Metric | Baseline | F1 Candidate | Delta |
| --- | --- | --- | --- |
| row_count | 424 | 424 | 0 |
| llm_selection_rate | 0.4858 | 0.5778 | +0.0920 |
| llm_top1_accuracy_on_all_rows | 0.1698 | 0.2028 | +0.0330 |
| llm_top1_accuracy_on_selected_rows | 0.3495 | 0.3510 | +0.0015 |
| repaired_selected_id_count | 0 | 0 | +0.0 |
| invalid_selected_id_count | 0 | 0 | +0.0 |

Interpretation:

- the tighter prompt made the model choose more often
- all-row top-1 improved on the shared error-row set
- selected-row accuracy barely moved and did not clear the promotion threshold
- selection rate regressed materially, so the candidate failed the cost-aware acceptance rule

## Required Slice Readout

- `Assessment`: top-1 `+0.0667`, selection rate `+0.1500`
- `catalog_state_specific_expected`: top-1 `+0.2917`, selection rate `+0.2500`
- `adoption_state_high_risk`: top-1 `+0.0191`, selection rate `+0.0828`
- `catalog_unspecified`: top-1 `+0.0213`, selection rate `+0.0638`
- `assessment_short_or_acronym_title`: top-1 `+0.0000`, selection rate `+0.0857`
- wrong-top1-but-gold-in-shortlist rows: top-1 `+0.1111`, selection rate `+0.1111`

What improved:

- difficult state-specific and assessment-heavy rows benefited meaningfully
- the candidate was noticeably more willing to overturn a wrong top-1 when the gold was already in the shortlist

Why that still was not enough:

- the prompt mainly improved by selecting more often, not by becoming materially sharper once it selected
- selected-row accuracy moved only `+0.0015`, far below the `+0.01` promotion threshold
- the larger selection rate jump weakens the current cost and caution contract for the accepted rerank baseline

## Keep / Reject Decision

- reject `F1`
- keep the current rerank prompt baseline

Why:

- the standing comparison rule requires at least `+0.01` on both all-row and selected-row top-1 with no selection-rate regression
- `F1` improved all-row top-1 but did not improve selected-row top-1 enough
- `F1` also increased selection rate substantially, which fails the no-regression rule
- roadmap default says mixed results should be rejected

## Baseline Update

- no baseline change
- `src/curriculum_matcher/llm_rerank.py` has been restored to the accepted prompt baseline after measurement
- accepted rerank baseline remains:
  - shortlist `10`
  - model `gpt-5.4-mini`
  - reasoning `medium`

## Recommended Next Step

If the roadmap continues, the next serial experiment should be `F2` raw matcher-score exposure, measured against the unchanged accepted baseline.
