# Curriculum Matcher Handoff v043

- Date: `2026-04-14`
- Milestone: `G2` started; live batch still in progress
- Operator surface: keep using `src/curriculum_matcher/app.py`
- Workflow: benchmark-driven, one variable at a time

## Current Verified Baseline

Accepted matcher changes:

- `C1` safe publisher alias normalization
- `C2` safe product-title acronym expansion
- `C5` safe state-specific token normalization

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline: `F2` no-score candidate payload

Accepted comparison-contract baseline:

- use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## What This Session Did

- lifted `G2`
- changed only the `DISTRICT_ROW` payload in `src/curriculum_matcher/llm_rerank.py`
- removed only:
  - `confidence_band`
  - `match_selected_strategy`
- updated `tests/test_llm_rerank.py` to assert that both fields are absent from the emitted `DISTRICT_ROW`
- verified locally:
  - `PYTHONPATH=src ./.venv/bin/python -m unittest tests.test_llm_rerank`
  - result: `5` tests passed
- regenerated `G2` prompt and request artifacts:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g2-internal-labels-prompts.jsonl`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g2-internal-labels-prompts-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g2-internal-labels-requests.jsonl`
- started the live OpenAI batch run

## Live G2 Batch State

- run name: `historical-top10-gpt54mini-medium-g2-internal-labels-batch`
- batch id: `batch_69de7b254d408190b7bbd31563c9d421`
- status at checkpoint: `in_progress`
- request counts at checkpoint:
  - total `455`
  - completed `413`
  - failed `0`

No scored output, comparison artifact, or keep / reject decision exists yet because the batch has not reached a terminal state.

## Recommended Next Step

Resume `G2` from the existing live batch instead of rebuilding anything.

Once the batch reaches a terminal state:

1. download the output through the existing pipeline tooling or direct batch retrieval
2. score the run into:
   - `historical-top10-gpt54mini-medium-g2-internal-labels-batch-scored.csv`
   - `historical-top10-gpt54mini-medium-g2-internal-labels-batch-scored-summary.json`
   - `historical-top10-gpt54mini-medium-g2-internal-labels-batch-pipeline-summary.json`
3. compare the scored summary against the canonical accepted `F2` baseline scored summary
4. write the `G2` review note
5. update the tracker to `accepted` or `rejected`
6. leave a fresh handoff before touching `G3`

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-14-v043-g2-batch-in-progress-handoff.md`.
>
> Current verified matcher baseline:
> - `C1` safe publisher alias normalization accepted
> - `C2` safe product-title acronym expansion accepted
> - `C5` safe state-specific token normalization accepted
>
> Current verified retrieval and rerank baseline:
> - stage-1 recall `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
> - shortlist `10`
> - model `gpt-5.4-mini`
> - reasoning `medium`
> - rerank prompt baseline is `F2` no-score exposure
>
> Accepted comparison-contract baseline:
> - use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`
>
> Current roadmap state:
> - `G1` is accepted
> - `G2` is in progress
> - `G3` and `G4` are not started
>
> Live `G2` batch:
> - run name `historical-top10-gpt54mini-medium-g2-internal-labels-batch`
> - batch id `batch_69de7b254d408190b7bbd31563c9d421`
> - checkpoint status `in_progress` with `413 / 455` completed and `0` failed
>
> Please:
> 1. resume from the existing `G2` batch instead of rebuilding prompts or requests,
> 2. wait for terminal status,
> 3. download and score the completed batch output,
> 4. compare it against the canonical accepted `F2` baseline,
> 5. decide accept / reject for `G2`,
> 6. update the tracker, write the review note, and leave a fresh handoff before ending.
