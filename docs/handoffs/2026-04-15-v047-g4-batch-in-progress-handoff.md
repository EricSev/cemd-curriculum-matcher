# Curriculum Matcher Handoff v047

- Date: `2026-04-15`
- Milestone: `G4` started; live batch still in progress
- Operator surface: keep using `src/curriculum_matcher/app.py`
- Workflow: benchmark-driven, one variable at a time

## Current Verified Baseline

Accepted matcher changes:

- `C1` safe publisher alias normalization
- `C2` safe product-title acronym expansion
- `C5` safe state-specific token normalization

Rejected matcher / retrieval / rerank / shortlist changes:

- `C3` grade normalization tightening
- `C4` edition / year title cleanup
- `D1` hybrid retrieval with RRF
- `D2` field-aware lexical weighting
- `E1` cross-encoder pre-LLM reranker
- `E2` shortlist-size retuning
- `F1` system prompt tightening
- `F3` candidate disambiguation block
- `F4` abstention wording calibration
- `G2` matcher-internal confidence label suppression
- `G3` derived ambiguity label suppression

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline: `F2` no-score candidate payload

Accepted comparison-contract baseline:

- use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## What This Session Did

- lifted `G4`
- changed only the `SHORTLIST_CANDIDATES` payload in `src/curriculum_matcher/llm_rerank.py`
- removed only:
  - `series`
- updated `tests/test_llm_rerank.py` to assert `series` is absent from emitted shortlist candidates while the accepted `F2` row-context fields and remaining candidate metadata stay present
- verified locally:
  - `PYTHONPATH=src ./.venv/bin/python -m unittest tests/test_llm_rerank.py`
  - result: `5` tests passed
- regenerated `G4` prompt and request artifacts:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g4-no-series-prompts.jsonl`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g4-no-series-prompts-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g4-no-series-requests.jsonl`
- started the live OpenAI batch run

## Live G4 Batch State

- run name: `historical-top10-gpt54mini-medium-g4-no-series-batch`
- batch id: `batch_69df8418282c8190bb455bf74b88e118`
- status at checkpoint: `in_progress`
- request counts at checkpoint:
  - total `455`
  - completed `400`
  - failed `0`

No scored output, comparison artifact, slice artifact, review note, or keep / reject decision exists yet because the batch has not reached a terminal state.

## Recommended Next Step

Resume `G4` from the existing live batch instead of rebuilding anything.

Once the batch reaches a terminal state:

1. download the output through the existing pipeline tooling or direct batch retrieval
2. score the run into:
   - `historical-top10-gpt54mini-medium-g4-no-series-batch-scored.csv`
   - `historical-top10-gpt54mini-medium-g4-no-series-batch-scored-summary.json`
   - `historical-top10-gpt54mini-medium-g4-no-series-batch-pipeline-summary.json`
3. compare the scored output against the canonical accepted `F2` baseline
4. write:
   - `docs/analysis/2026-04-14-series-exposure-review.md`
   - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g4-no-series-vs-f2-comparison.json`
5. update the tracker to `accepted` or `rejected`
6. leave a fresh handoff before touching any new experiment

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-15-v047-g4-batch-in-progress-handoff.md`.
>
> Current verified matcher baseline:
> - `C1` safe publisher alias normalization accepted
> - `C2` safe product-title acronym expansion accepted
> - `C5` safe state-specific token normalization accepted
> - `C3` grade normalization rejected
> - `C4` edition / year cleanup rejected
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
> - `G2` is rejected
> - `G3` is rejected
> - `G4` is in progress
> - no experiment is currently in progress besides the live `G4` batch
>
> Live `G4` batch:
> - run name `historical-top10-gpt54mini-medium-g4-no-series-batch`
> - batch id `batch_69df8418282c8190bb455bf74b88e118`
> - checkpoint status `in_progress` with `400 / 455` completed and `0` failed
>
> Please:
> 1. resume from the existing `G4` batch instead of rebuilding prompts or requests,
> 2. wait for terminal status,
> 3. download and score the completed batch output,
> 4. compare it against the canonical accepted `F2` baseline,
> 5. decide accept / reject for `G4`,
> 6. update the tracker, write the review note, and leave a fresh handoff before ending.
