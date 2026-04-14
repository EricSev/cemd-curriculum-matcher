# Next-Phase Build/Test Iterations

- Date: 2026-04-05
- Status: `in progress`
- Program posture: rerank-first, benchmark-driven, one variable at a time

## Locked priorities for the next phase

Test now:

1. `gpt-5.4` vs `gpt-5.4-mini` at `top-10`, `medium`
2. shortlist recall / MRR / `nDCG@10` instrumentation
3. publisher alias normalization
4. acronym / abbreviation expansion
5. grade normalization tightening
6. edition / year extraction improvements
7. hybrid retrieval with Reciprocal Rank Fusion
8. field-aware lexical weighting
9. character n-gram retrieval

Test after the above:

- cross-encoder reranker before the LLM
- shortlist size retuning only after retrieval experiments are measured

Defer for later:

- core matcher scoring logic rewrites
- learned rerankers / LambdaMART
- model fine-tuning / Ditto-style matching
- ColBERT / SPLADE / larger retrieval architecture changes
- broad UI refactors unrelated to benchmark flow

## Iteration sequence

### Iteration 1: Lock the rerank benchmark baseline

- keep shortlist size fixed at `10`
- keep reasoning effort fixed at `medium`
- compare `gpt-5.4-mini` vs `gpt-5.4`
- use the scored-summary comparison script to decide whether the larger model earns promotion

Acceptance rule:

- only promote the larger model if top-1 on all reviewed rows and top-1 on selected rows each improve by at least `0.01` with no selection-rate regression

Primary artifacts:

- `benchmarks/outputs/openai_batch_runs/*-scored-summary.json`
- `benchmarks/outputs/openai_batch_runs/*-pipeline-summary.json`
- `benchmarks/outputs/openai_batch_runs/*-comparison.json`
- `docs/analysis/*-comparison.md`

### Iteration 2: Measure shortlist quality directly

- keep matcher scoring logic unchanged
- require evaluation outputs to include:
  - hit@1
  - hit@3
  - hit@10
  - MRR
  - `nDCG@10`
- require these metrics in:
  - overall summary
  - confidence bands
  - required benchmark slices

Required slices:

- `product_type_usage`
- `evidence_richness`
- `placeholder_mapping`
- `state_specific_risk`
- `assessment_slice`

Primary artifacts:

- updated representative benchmark summary JSON
- updated representative benchmark records CSV
- shortlist bottleneck snapshot markdown

### Iteration 3: Test normalization one family at a time

- publisher aliases
- acronym expansion
- grade normalization
- edition/year extraction
- state-variant token normalization

Rule:

- each normalization family must be benchmarked independently and must not regress overall top-1 materially

Acceptance focus:

- `Assessment`
- `catalog_unspecified`
- state-risk slices
- publisher-missing and acronym-heavy rows

### Iteration 4: Test retrieval upgrades before scorer changes

- hybrid retrieval with RRF
- field-aware lexical weighting
- character n-gram retrieval

Primary success criteria:

- shortlist recall@10 improves
- no unacceptable downstream precision regression
- visible gains in sparse-evidence / assessment / abbreviation-heavy slices

### Iteration 5: Add a deterministic pre-LLM reranker if needed

- test a cross-encoder over retrieved candidates before the LLM
- compare retrieval-only + LLM vs retrieval + cross-encoder + LLM
- treat this as the first non-LLM cascade experiment, not a broad architecture rewrite

## Current implementation status

Completed in this pass:

- evaluation summary contract now includes shortlist metrics:
  - `hit_rate_at_1`
  - `hit_rate_at_3`
  - `hit_rate_at_10`
  - `mrr`
  - `ndcg_at_10`
- slice and cross-slice summaries now carry those rank metrics
- representative top-10 benchmark artifacts have been regenerated with the richer metric contract
- shortlist bottleneck report script exists and has been run on the representative top-10 benchmark
- scored-summary comparison script exists for the rerank model comparison checkpoint
- `gpt-5.4` vs `gpt-5.4-mini` comparison is complete and the mini baseline remains the default
- live review task tracker now exists for the next work cycle
- standing experiment comparison template now exists
- weekly checkpoint template now includes the fixed worst-bottleneck review section

Open at end of this pass:

- start the first normalization experiment from the live review task tracker

Primary tracking docs:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`
- `docs/roadmap/experiment-comparison-template.md`
