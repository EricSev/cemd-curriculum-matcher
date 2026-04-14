# Shortlist Bottleneck Snapshot

- benchmark: `benchmarks/gold/historical_07122025_representative_1000.csv`
- profile: `fast`
- topn_final: `10`
- hit@1: `0.46`
- hit@3: `0.561`
- hit@10: `0.599`
- MRR: `0.5126`
- nDCG@10: `0.5341`

## Worst recall-limiting slices

- `product_type_usage` worst slice: `Assessment` | count `260` | top3 `0.3577` | top10 `0.3731`
- `evidence_richness` worst slice: `policy_placeholder_no_info` | count `1` | top3 `0.0` | top10 `0.0`
- `placeholder_mapping` worst slice: `no_information_available` | count `1` | top3 `0.0` | top10 `0.0`
- `state_specific_risk` worst slice: `adoption_state_high_risk` | count `349` | top3 `0.5272` | top10 `0.5473`
- `assessment_slice` worst slice: `assessment_state_specific_expected` | count `7` | top3 `0.0` | top10 `0.0`
