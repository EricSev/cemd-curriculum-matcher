# Curriculum Matcher Handoff v045

- Date: `2026-04-14`
- Milestone: `G3` started; live batch still in progress
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

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline: `F2` no-score candidate payload

Accepted comparison-contract baseline:

- use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## What This Session Did

- lifted `G3`
- changed only the `DISTRICT_ROW` payload in `src/curriculum_matcher/llm_rerank.py`
- removed only:
  - `usage_ambiguity`
  - `state_specific_risk`
  - `placeholder_mapping`
  - `assessment_slice`
- updated `tests/test_llm_rerank.py` to assert those four fields are absent from the emitted `DISTRICT_ROW` while `confidence_band` and `match_selected_strategy` remain present
- verified locally:
  - `PYTHONPATH=src ./.venv/bin/python -m unittest tests/test_llm_rerank.py`
  - result: `5` tests passed
- regenerated `G3` prompt and request artifacts:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-prompts.jsonl`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-prompts-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-requests.jsonl`
- started the live OpenAI batch run

## Live G3 Batch State

- run name: `historical-top10-gpt54mini-medium-g3-derived-labels-batch`
- batch id: `batch_69debc7911b08190b55cccfdbba7d76c`
- status at checkpoint: `in_progress`
- request counts at checkpoint:
  - total `455`
  - completed `0`
  - failed `0`

No scored output, comparison artifact, slice artifact, review note, or keep / reject decision exists yet because the batch has not reached a terminal state.

## Recommended Next Step

Resume `G3` from the existing live batch instead of rebuilding anything.

Once the batch reaches a terminal state:

1. download the output through the existing pipeline tooling or direct batch retrieval
2. score the run into:
   - `historical-top10-gpt54mini-medium-g3-derived-labels-batch-scored.csv`
   - `historical-top10-gpt54mini-medium-g3-derived-labels-batch-scored-summary.json`
   - `historical-top10-gpt54mini-medium-g3-derived-labels-batch-pipeline-summary.json`
3. compare the scored output against the canonical accepted `F2` baseline on the shared `selection_identifier` row set if row coverage differs
4. write:
   - `docs/analysis/2026-04-14-derived-ambiguity-label-suppression-review.md`
   - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-vs-f2-comparison.json`
5. update the tracker to `accepted` or `rejected`
6. leave a fresh handoff before touching `G4`

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-14-v045-g3-batch-in-progress-handoff.md`.
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
> - `G3` is in progress
> - `G4` is not started
>
> Live `G3` batch:
> - run name `historical-top10-gpt54mini-medium-g3-derived-labels-batch`
> - batch id `batch_69debc7911b08190b55cccfdbba7d76c`
> - checkpoint status `in_progress` with `0 / 455` completed and `0` failed
>
> Please:
> 1. resume from the existing `G3` batch instead of rebuilding prompts or requests,
> 2. wait for terminal status,
> 3. download and score the completed batch output,
> 4. compare it against the canonical accepted `F2` baseline,
> 5. decide accept / reject for `G3`,
> 6. update the tracker, write the review note, and leave a fresh handoff before ending.
