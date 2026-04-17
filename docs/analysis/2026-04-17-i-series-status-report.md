# I-Series Status Report

- Date: `2026-04-17`
- Status: `I` series closed
- Runtime baseline changed: `no`
- Operator surface changed: `no`

## Final Decisions

| ID | Decision | Scope | Result |
| --- | --- | --- | --- |
| `I1` | accepted diagnostic | Assessment and `catalog_unspecified` taxonomy audit | Most assessment misses are ambiguity / historical-ground-truth cases; `catalog_unspecified` is mostly catalog-label / placeholder structure. |
| `I2` | accepted diagnostic | Catalog-label consistency audit | `catalog_unspecified` splits into unresolved structural placeholders and retrievable named-series rows. |
| `I3` | accepted diagnostic candidate | split `catalog_unspecified` derived label | `56` rows split into `46` named-series rows and `10` structural-placeholder rows. |
| `I4` | rejected | taxonomy-informed retrieval candidate | No shortlist movement and no metric gains. |

## Key Findings

- `Assessment` is not primarily a clean retrieval-expansion problem: only `6 / 260` assessment rows landed in the clean `likely_retrieval_miss` bucket in `I1`.
- `catalog_unspecified` is not one failure type: `I3` split it into `46` named-series rows and `10` structural-placeholder rows.
- The `I4` retrieval candidate did not move any accepted top-10 shortlists, so no LLM rerank batch was justified.
- The accepted runtime baseline remains matcher `C1 + C2 + C5`, retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, and `F2` no-score rerank prompt.

## Artifacts

- `docs/analysis/2026-04-17-i1-taxonomy-audit.md`
- `docs/analysis/2026-04-17-i2-catalog-label-consistency.md`
- `docs/analysis/2026-04-17-i3-label-normalization-review.md`
- `docs/analysis/2026-04-17-i4-taxonomy-recall-review.md`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_i1_taxonomy_audit.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_i1_taxonomy_audit_rows.csv`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_i2_catalog_label_consistency.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_i2_catalog_label_consistency_rows.csv`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_i3_label_normalization.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_i3_label_normalization_rows.csv`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_i4_taxonomy_recall_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_i4_taxonomy_recall_records.csv`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_i4_taxonomy_recall_comparison.json`

## Recommended Next Phase

Define a new experiment set before changing behavior again.

Recommended focus:

- promote the `I3` split into stable reporting / prompt-context terminology
- review ambiguous assessment gold labels before another retrieval experiment
- avoid broad candidate expansion unless a new diagnostic identifies a narrow recoverable class
