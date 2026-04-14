# Human QA Tuning Report

- row count: 261
- overall challenge precision: 0.8209
- overall challenge recall: 0.3416
- overall challenge rate: 0.2567

## By Review Priority

| human_match_review_priority | row_count | incorrect_rate | challenge_rate | precision | recall |
| --- | --- | --- | --- | --- | --- |
| low | 194 | 0.5464 | 0.0 | 0.0 | 0.0 |
| medium | 38 | 0.7368 | 1.0 | 0.7368 | 1.0 |
| high | 29 | 0.931 | 1.0 | 0.931 | 1.0 |

## By Benchmark Slice

| qa_benchmark_slice | row_count | incorrect_rate | challenge_rate | precision | recall |
| --- | --- | --- | --- | --- | --- |
| has_adopted_url | 156 | 0.5962 | 0.0962 | 0.7333 | 0.1183 |
| fallback_selected | 105 | 0.6476 | 0.4952 | 0.8462 | 0.6471 |

## By Case Type

| qa_case_type | row_count | incorrect_rate | challenge_rate | precision | recall |
| --- | --- | --- | --- | --- | --- |
| expected_match | 100 | 0.0 | 0.12 | 0.0 | 0.0 |
| pred_rank_2_incorrect | 74 | 1.0 | 0.3919 | 1.0 | 0.3919 |
| pred_rank_1_incorrect | 53 | 1.0 | 0.3019 | 1.0 | 0.3019 |
| pred_rank_3_incorrect | 34 | 1.0 | 0.2941 | 1.0 | 0.2941 |

## By Label Strength

| qa_label_strength | row_count | incorrect_rate | challenge_rate | precision | recall |
| --- | --- | --- | --- | --- | --- |
| silver_negative | 108 | 1.0 | 0.3611 | 1.0 | 0.3611 |
| gold_positive | 100 | 0.0 | 0.12 | 0.0 | 0.0 |
| silver_strong_negative | 53 | 1.0 | 0.3019 | 1.0 | 0.3019 |

## Support Score Threshold Sweep

| support_score_lt | challenge_rate | precision | recall |
| --- | --- | --- | --- |
| 0.25 | 0.046 | 0.5833 | 0.0435 |
| 0.35 | 0.1073 | 0.6071 | 0.1056 |
| 0.45 | 0.3142 | 0.7195 | 0.3665 |
| 0.55 | 0.4828 | 0.6746 | 0.528 |
| 0.65 | 0.7165 | 0.6524 | 0.7578 |

## Most Informative Challenge Reasons

| reason | row_count | incorrect_rate |
| --- | --- | --- |
| repaired_input_changed_prediction | 41 | 0.8293 |
| low_support_score | 53 | 0.8113 |
| weak_title_evidence | 42 | 0.8095 |
| ai_prefers_alternative | 55 | 0.7818 |
| weak_publisher_evidence | 44 | 0.7727 |

## Reviewer Workflow Recommendation

- `high` priority currently yields precision 0.9310 on 29 rows; use this bucket as the default must-review queue.
- `medium` priority currently yields precision 0.7368; treat this bucket as overflow review or sampled audit rather than the default queue.
- `low` priority currently yields precision 0.0000; treat this bucket as pass-through with periodic calibration sampling only.
- Simple support-score thresholds remain useful as a reference, but the current operating point should stay compound: strong AI disagreement first, then low-support rows only when title evidence is weak.
