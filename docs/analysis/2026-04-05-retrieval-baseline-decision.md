# Retrieval Baseline Decision Checkpoint

- Date: `2026-04-05`
- Task: `D4` retrieval baseline decision checkpoint
- Variable changed: none
- Scope: compare accepted retrieval candidates only
- Decision note uses existing measured artifacts; no new benchmark variant was introduced

## Candidates Compared

Baseline retrieval candidate before `D3` acceptance:

- `C1 + C2 + C5` matcher baseline with direct stage-1 blend
- artifact: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_state_normalization_summary.json`

Accepted retrieval candidate from `D3`:

- `C1 + C2 + C5` matcher baseline with `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- artifact: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_summary.json`

Shortlist and rerank settings held fixed across both:

- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## Code State Validation

Current source confirms the accepted retrieval baseline is what the app runs by default:

- `src/curriculum_matcher/app.py` defaults `CURRICULUM_MATCHER_RETRIEVAL_EXPERIMENT` to `char_ngram`
- `_blend_stage1_scores()` keeps the accepted blend at `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- `tests/test_matcher_core.py` includes coverage for the char n-gram default and the explicit direct-blend override

Validation completed:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./.venv/bin/python -m unittest tests.test_matcher_core tests.test_evaluation -q`

## Measured Comparison

Representative benchmark: `historical_07122025_representative_1000`, profile `fast`, shortlist `10`

| Metric | Direct Blend | Char N-Gram Blend | Delta |
| --- | ---: | ---: | ---: |
| top-1 accuracy | 0.4760 | 0.4880 | +0.0120 |
| top-3 recall | 0.5740 | 0.5740 | +0.0000 |
| prediction rate | 0.9370 | 0.9430 | +0.0060 |
| shortlist hit@10 | 0.6190 | 0.6280 | +0.0090 |
| shortlist MRR | 0.5285 | 0.5374 | +0.0089 |
| shortlist nDCG@10 | 0.5508 | 0.5596 | +0.0088 |

Interpretation:

- the accepted char n-gram blend remains the stronger retrieval default on the fixed representative benchmark
- the gain is not isolated to one metric; top-1, prediction rate, hit@10, MRR, and nDCG@10 all moved in the right direction
- top-3 recall stayed flat, so keeping shortlist `10` does not require another retrieval change here

## Required Slice Check

Key protected slices from the accepted `D3` comparison still support the same decision:

- `Assessment`: top-1 `+0.0269`, top-10 `+0.0076`, MRR `+0.0160`
- `assessment_short_or_acronym_title`: top-1 `+0.0435`, MRR `+0.0248`
- `adoption_state_high_risk`: top-1 `+0.0114`, top-10 `+0.0086`, MRR `+0.0099`
- `sparse` evidence: top-1 `+0.0176`, top-10 `+0.0075`, MRR `+0.0131`
- `catalog_state_specific_expected`: flat across protected metrics

Known carried risk:

- `catalog_unspecified` remains weak and lost top-3 recall `-0.0357`
- `assessment_publisher_missing` lost top-3 recall `-0.0123`
- those losses did not create a top-10 regression large enough to overturn the accepted retrieval baseline

## Decision

Decision: `accepted`

Why:

- `D3` already established the char n-gram blend as the only accepted retrieval variant
- this checkpoint confirms there is no remaining evidence-based reason to revert to the direct blend
- shortlist `10` stays justified because the accepted retrieval baseline improved shortlist quality without requiring a higher shortlist to realize the gain

## Baseline Update

Retrieval baseline remains:

- `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`

Rerank baseline remains:

- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

Matcher baseline remains:

- `C1` publisher alias normalization
- `C2` acronym expansion
- `C5` state-specific token normalization

## Next Task

The next roadmap items are still deferred:

- `E1` cross-encoder pre-LLM reranker experiment
- `E2` shortlist-size retuning

Do not start either unless the next session explicitly decides to lift that deferral or open the next phase.
