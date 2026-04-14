# E2 Shortlist-Size Retuning Review

- Date: `2026-04-06`
- Task: `E2` shortlist-size retuning
- Decision: `rejected`
- Baseline summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_summary.json`
- Candidate summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top15_shortlist_retune_summary.json`
- Candidate records: `benchmarks/outputs/historical_07122025_representative_1000_fast_top15_shortlist_retune_records.csv`
- Prompt-pack baseline summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-batch-prompts-summary.json`
- Prompt-pack candidate summary: `benchmarks/outputs/openai_batch_runs/historical-top15-gpt54mini-medium-batch-prompts-summary.json`

Representative benchmark: `historical_07122025_representative_1000`, profile `fast`, retrieval baseline `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`

Single variable:

- shortlist size `10` -> `15`

## Overall Delta

| Metric | Baseline top-10 | Candidate top-15 | Delta |
| --- | --- | --- | --- |
| top-1 accuracy | 0.4880 | 0.4880 | +0.0000 |
| top-3 recall | 0.5740 | 0.5740 | +0.0000 |
| shortlist hit@10 | 0.6280 | 0.6280 | +0.0000 |
| shortlist MRR | 0.5374 | 0.5377 | +0.0003 |
| shortlist nDCG@10 | 0.5596 | 0.5596 | +0.0000 |
| shortlist hit@15 | 0.6280 | 0.6320 | +0.0040 |

## Prompt-Pack Coverage Delta

- error-row prompt count stayed `455`
- average candidates per review row rose from `4.1714` to `4.6022` (`+0.4308`)
- `59` review rows expanded beyond `10` candidates
- candidate-count tail under top-15:
  - `11`: `12`
  - `12`: `6`
  - `13`: `9`
  - `14`: `15`
  - `15`: `17`

## Row-Level Movement

- `top-15` added `4` new gold-shortlist rescues, all at ranks `11-15`
- no existing gold-shortlist rows changed rank inside the top-15
- the `4` rescue rows were:
  - `TX / ELA / Assessment / medium / adoption_state_high_risk / assessment_short_or_acronym_title` at rank `14`
  - `TX / Math / Assessment / medium / adoption_state_high_risk / assessment_short_or_acronym_title` at rank `11`
  - `CA / Science / Core Curriculum / medium / adoption_state_high_risk / not_assessment` at rank `11`
  - `NJ / Science / Core Curriculum / sparse / standard_state / not_assessment` at rank `11`

## Required Slice Readout

- `Assessment` improved from `0.4538` at top-10 to `0.4615` at top-15
- `assessment_short_or_acronym_title` improved from `0.4720` at top-10 to `0.4845` at top-15
- `adoption_state_high_risk` improved from `0.5759` at top-10 to `0.5845` at top-15
- those gains do not change the protected top-10 quality contract for the accepted rerank baseline

## Keep / Reject Decision

- reject `E2`
- keep shortlist `10`

## Baseline Update

- no baseline change
- locked rerank baseline remains:
  - shortlist `10`
  - model `gpt-5.4-mini`
  - reasoning `medium`
