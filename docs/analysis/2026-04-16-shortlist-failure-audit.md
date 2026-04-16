# H1 Shortlist Failure Audit

- Task: `H1` diagnostic checkpoint
- records rows: `1000`
- LLM joined rows: `455`
- behavior changed: `no`

## Retrieval / Candidate Recall Buckets

| slice | row_count | gold_absent_from_top10 | gold_absent_from_top10_rate | gold_present_in_top10 | ranker_top1_failure_with_gold_present |
| --- | --- | --- | --- | --- | --- |
| Assessment | 260 | 142 | 0.5462 | 118 | 23 |
| catalog_unspecified | 56 | 37 | 0.6607 | 19 | 14 |
| adoption_state_high_risk | 349 | 148 | 0.4241 | 201 | 32 |
| catalog_state_specific_expected | 53 | 16 | 0.3019 | 37 | 12 |

## LLM Selection Buckets On Joined Rows

| slice | llm_row_count | llm_gold_absent_from_top10 | llm_gold_present_in_top10 | llm_selection_failure_with_gold_present | llm_abstained_with_gold_present | llm_top1_success |
| --- | --- | --- | --- | --- | --- | --- |
| Assessment | 136 | 113 | 23 | 4 | 2 | 19 |
| catalog_unspecified | 48 | 34 | 14 | 6 | 0 | 8 |
| adoption_state_high_risk | 165 | 133 | 32 | 13 | 0 | 19 |
| catalog_state_specific_expected | 26 | 14 | 12 | 0 | 0 | 12 |

## Interpretation

- `Assessment` has `142` top-10 recall failures and `23` ranker-selection failures with the gold row present.
- `catalog_unspecified` remains recall-limited: `37` of `56` rows do not have the gold row in the top-10.
- `catalog_state_specific_expected` has a mixed failure shape: `16` absent-from-top10 rows and `12` gold-present ranker failures.
- `adoption_state_high_risk` is split across both failure modes: `148` recall misses and `32` gold-present ranker failures.

## Decision

- accept `H1` as a diagnostic checkpoint
- no behavioral baseline change
- use this audit to prioritize `H2` assessment-aware candidate recall next
