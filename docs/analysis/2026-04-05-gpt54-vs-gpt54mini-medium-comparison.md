# OpenAI Batch Rerank Comparison

- Baseline: `historical-top10-gpt54mini-medium-batch`
- Candidate: `historical-top10-gpt54-medium-batch`
- Decision: `keep_baseline`
- Rule: Promote only when top-1 on all rows and on selected rows each improve by at least 0.01, without a selection-rate regression.
- Reason: Candidate did not clear the cost-aware promotion thresholds.

## Metrics

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| row_count | 472 | 472 | 0 |
| llm_selection_rate | 0.4788 | 0.3093 | -0.1695 |
| llm_top1_accuracy_on_all_rows | 0.1843 | 0.1716 | -0.0127 |
| llm_top1_accuracy_on_selected_rows | 0.385 | 0.5548 | 0.1698 |
| repaired_selected_id_count | 1 | 0 | -1.0 |
| invalid_selected_id_count | 0 | 0 | 0.0 |
