# Historical Sample Representativeness Report

## Input Files

- population file: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Curriculum Matching Data - 07122025.csv`
- matched sample file: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/tbl_MatchedDataSample750.csv`
- raw-view sample file: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/tbl_UnMatchedDataSample750.csv`
- historical results file: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/Matcher_Results_20250712_10K.csv`

## Routine Compatibility

- matched sample shares required matcher input columns: `['product_name_raw', 'publisher_raw', 'grade']`
- raw-view sample shares required matcher input columns: `['product_name_raw', 'publisher_raw', 'grade']`
- matched sample matcher-compatible: `True`
- raw-view sample matcher-compatible: `True`
- Conclusion: the same matcher input routine can run on both 750-row files because both retain `product_name_raw`, `publisher_raw`, and `grade`.

## Sample Relationship

- matched sample rows: `750`
- raw-view sample rows: `750`
- overlapping `selection_identifier` rows: `750`
- matched-only rows: `0`
- raw-only rows: `0`
- Interpretation: the two 750-row files are two schema views of the same records, not separate matched and unmatched populations.

## Population Notes

- full population row count: `532163`
- full population nonblank `product_identifier` rate: `True`
- full population product-type mix: `{'Core Curriculum': 0.4238, 'Supplemental': 0.3222, 'Assessment': 0.2541}`

## Representativeness Comparison

| dataset | row_count | publisher_missing_rate | product_type_usage_top | state_tvd | subject_tvd | product_type_usage_tvd | publisher_missing_abs_diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matched_750 | 750 | 0.6093 | {"Supplemental": 0.544, "Core Curriculum": 0.456} | 0.3677 | 0.0146 | 0.2541 | 0.0018 |
| raw_view_750 | 750 | 0.6093 | {"Supplemental": 0.544, "Core Curriculum": 0.456} | 0.3677 | 0.0146 | 0.2541 | 0.0018 |
| representative_sample | 1000 | 0.62 | {"Core Curriculum": 0.412, "Supplemental": 0.328, "Assessment": 0.26} | 0.0308 | 0.0083 | 0.0118 | 0.0125 |
| historical_results_10k | 10000 | 0.5365 | {"Core Curriculum": 0.5595, "Supplemental": 0.4405} | 0.0459 | 0.0666 | 0.2541 | 0.071 |

Interpretation:

- Lower total-variation distance is better.
- The 750-row sample materially diverges from the 532K population, especially because it omits the `Assessment` segment entirely.
- The 10K results file is directionally closer than the 750-row sample on state mix, but it still misses `Assessment` and should not be treated as fully representative.

## Representative Sample Output

- generated sample rows: `1000`
- generated sample product-type mix: `{'Core Curriculum': 0.412, 'Supplemental': 0.328, 'Assessment': 0.26}`
- Sampling method: deterministic proportional stratification over `product_type_usage`, `subject`, `state`, and publisher presence.
- This sample is a better benchmark starting point than the existing 750-row slice because it preserves the full-population `Assessment` share.

## Recommendation

- Keep the two 750-row files only as a routine-compatibility check, not as the main evidence for representativeness.
- Use the full 532K file as the source population for benchmark sampling.
- Use the generated representative sample for future benchmark expansion, then add targeted slices for hard failure families on top of it.

