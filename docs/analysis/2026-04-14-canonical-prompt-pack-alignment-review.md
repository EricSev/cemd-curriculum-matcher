# G1 Canonical Prompt-Pack Alignment Review

- Date: `2026-04-14`
- Task: `G1` canonical prompt-pack alignment checkpoint
- Decision: `accepted`
- Accepted rerank baseline before this task: `F2` no-score candidate payload
- Canonical baseline manifest: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## Single Variable

- reporting / comparison contract only
- no rerank behavior change
- no matcher, retrieval, shortlist, model, reasoning, or prompt payload field changes

Held fixed:

- matcher baseline `C1 + C2 + C5`
- retrieval baseline `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline `F2` no-score candidate payload

## What Was Checked

The purpose of `G1` was to stop relying on overlap-only comparisons against the older pre-`F2` historical scored artifact and to freeze one row-matched accepted baseline for future rerank experiments.

Validated accepted `F2` artifact set:

- prompt JSONL: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-prompts.jsonl`
- prompt summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-prompts-summary.json`
- batch output: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-batch-batch-output.jsonl`
- scored CSV: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-batch-scored.csv`
- scored summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-batch-scored-summary.json`
- pipeline summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-batch-pipeline-summary.json`

## Row-Match Validation

Accepted `F2` run parity:

| Check | Value |
| --- | ---: |
| prompt row count | 455 |
| prompt unique ids | 455 |
| scored row count | 455 |
| scored unique ids | 455 |
| prompt minus scored | 0 |
| scored minus prompt | 0 |
| duplicate prompt ids | 0 |
| duplicate scored ids | 0 |

Interpretation:

- the accepted `F2` prompt pack and scored output align exactly one-to-one
- the accepted `F2` run is already suitable to serve as the canonical row-matched baseline for future rerank comparisons

## Why This Checkpoint Was Needed

The older pre-`F2` historical baseline artifact still does not align with the current error-row prompt pack:

| Check | Value |
| --- | ---: |
| prompt row count | 455 |
| scored row count | 472 |
| prompt minus scored | 31 |
| scored minus prompt | 48 |

That mismatch is why `F1` through `F4` decisions depended on shared-row comparisons.

## Keep / Accept Decision

- accept `G1`
- freeze the accepted `F2` row-matched artifact set as the official rerank comparison baseline for the `G` series

Why:

- `G1` achieved exact row-count parity without changing model behavior
- it removes avoidable comparison ambiguity for future prompt-contract experiments
- it gives later `G` tasks a single accepted baseline artifact set to compare against directly

## Baseline Update

- no behavioral baseline change
- comparison-contract baseline is now the accepted `F2` row-matched artifact set recorded in:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## Recommended Next Step

If work continues in this roadmap, lift exactly `G2` next and change only the `DISTRICT_ROW` payload in `src/curriculum_matcher/llm_rerank.py` by removing:

- `confidence_band`
- `match_selected_strategy`
