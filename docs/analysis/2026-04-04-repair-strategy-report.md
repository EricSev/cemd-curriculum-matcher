# Repair Strategy Report

- row count: 100
- repair attempted rate: 0.5100
- fallback usage rate: 0.2700
- repaired rows: 27

## Repair Outcome Summary

- repair rescued no-prediction rows: 24
- repair improved top-1 correctness: 6
- repair hurt top-1 correctness: 0
- repair left top-1 unchanged: 0

## By Selected Strategy

| match_selected_strategy | row_count | top1_accuracy | top3_recall | mean_top1_score |
| --- | --- | --- | --- | --- |
| primary | 73 | 0.5616 | 0.7123 | 0.6235 |
| title_plus_publisher | 20 | 0.3 | 0.3 | 0.5243 |
| publisher_as_title | 7 | 0.0 | 0.0 | 0.6577 |

## By Strategy And Confidence Band

| match_selected_strategy | confidence_band | row_count | top1_accuracy | top3_recall |
| --- | --- | --- | --- | --- |
| primary | high | 13 | 0.7692 | 0.8462 |
| primary | low | 8 | 0.5 | 0.625 |
| primary | medium | 49 | 0.551 | 0.7347 |
| primary | no_prediction | 2 | 0.0 | 0.0 |
| primary | very_low | 1 | 0.0 | 0.0 |
| publisher_as_title | medium | 7 | 0.0 | 0.0 |
| title_plus_publisher | low | 6 | 0.0 | 0.0 |
| title_plus_publisher | medium | 13 | 0.3846 | 0.3846 |
| title_plus_publisher | very_low | 1 | 1.0 | 1.0 |

## Recommendation

- Keep repair selection score-based in general, but retain a heuristic gate on `title_plus_publisher` when it tries to replace an existing primary match.
- In this benchmark slice, fallback repairs still rescued 24 no-prediction rows while reducing repair hurt rows to 0.
- `publisher_as_title` remains a low-trust strategy in this slice with top-1 accuracy 0.0000; keep it out of any looser replacement policy.

## Example Rows

### Helped
| product_name_raw | publisher_raw | grade | expected_match_id_norm | primary_top1_id | predicted_match_id_norm | match_selected_strategy |
| --- | --- | --- | --- | --- | --- | --- |
| Get Ready | Vista Higher Learning | 7 | 69fd26ffdecc9e2739c3d47f6ce23fb2936a8d5b49bcb2d7d3b378b0e3f60408 |  | 69fd26ffdecc9e2739c3d47f6ce23fb2936a8d5b49bcb2d7d3b378b0e3f60408 | title_plus_publisher |
| Ready to Advance | Benchmark Education | TK | 7f90d8edd14992157fe6f6a455f35ce9c82b9d10c0098e050ae52546cea64acf |  | 7f90d8edd14992157fe6f6a455f35ce9c82b9d10c0098e050ae52546cea64acf | title_plus_publisher |
| Raz Kids | nan | 7 | 48bf52e4216b58b9b2f698c128378c8f4ba2f662821486f84b962a73ebc36d78 |  | 48bf52e4216b58b9b2f698c128378c8f4ba2f662821486f84b962a73ebc36d78 | title_plus_publisher |
| Lexia Learning online literacy program | nan | 11 | 5fb10758249670e6952d5f6ce2be127c3d9e0ce34c4fa86b96769758cbf60b33 |  | 5fb10758249670e6952d5f6ce2be127c3d9e0ce34c4fa86b96769758cbf60b33 | title_plus_publisher |
| Lexia Learning | nan | 5 | d499119780cdb98b9b6a2a91f1bab8d6286296d5d83c487c7dfe9e970680c2d5 | 2c28f535599947c449e83c970f3fd40b151c7b3c862691019a51772f1404784b | d499119780cdb98b9b6a2a91f1bab8d6286296d5d83c487c7dfe9e970680c2d5 | title_plus_publisher |
| Inside the USA | nan | 11 | 3dfb2ac36ecf42a79e805a3249f666fd9b240b250fe201a0b4ccbad1e6d94d7c |  | 3dfb2ac36ecf42a79e805a3249f666fd9b240b250fe201a0b4ccbad1e6d94d7c | title_plus_publisher |
