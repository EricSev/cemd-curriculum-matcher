# Next Experiment Set After E-Series Closeout

- Date: `2026-04-06`
- Status: `ready`
- Scope: define the next benchmarked experiment set before any further matcher, retrieval, or rerank behavior changes
- Operator surface: keep using `src/curriculum_matcher/app.py`

## Why A New Experiment Set Exists

The `E` series is closed:

- `E1` cross-encoder pre-LLM reranker is `rejected`
- `E2` shortlist-size retuning is `rejected`

That leaves no active experiment in the current roadmap. The next phase should not reopen retrieval architecture, shortlist size, or accepted matcher normalization work unless new measured evidence justifies it.

The highest-leverage remaining seam that still fits the current Tkinter batch workflow is the LLM rerank prompt contract in `src/curriculum_matcher/llm_rerank.py`.

## Locked Baseline To Preserve

Hold these fixed for every `F`-series experiment unless the tracker explicitly says otherwise:

- matcher baseline: accepted `C1`, `C2`, `C5`; rejected `C3`, `C4`
- retrieval baseline: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist size: `10`
- rerank model: `gpt-5.4-mini`
- reasoning effort: `medium`
- operator surface: `src/curriculum_matcher/app.py`

Do not reopen `E1` or `E2` inside this phase.

## F-Series Goal

Test prompt-contract changes one variable at a time before touching broader architecture again.

These experiments should run against the saved top-10 baseline prompt-pack flow and be compared against the current accepted batch rerank baseline:

- baseline run: `historical-top10-gpt54mini-medium-batch`
- baseline scored summary: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-batch-scored-summary.json`

## Experiment Sequence

### `F1` System prompt tightening

Single variable:

- rewrite `SYSTEM_PROMPT` only

Intent:

- make shortlist-only selection, district-usage context, and ambiguity handling more explicit without changing prompt payload fields

Acceptance focus:

- `llm_top1_accuracy_on_all_rows`
- `llm_top1_accuracy_on_selected_rows`
- `llm_selection_rate`
- ambiguity-heavy slices, especially state-specific and assessment-heavy rows

### `F2` Raw matcher-score exposure

Single variable:

- remove or suppress raw candidate `score` values from the prompt while keeping candidate order unchanged

Intent:

- test whether numeric score anchoring is limiting the LLM's ability to overturn a wrong top-1 candidate when the gold is already inside the shortlist

Acceptance focus:

- rows where baseline top-1 is wrong but shortlist contains the gold
- selection-rate stability
- invalid-id / repaired-id counts

### `F3` Disambiguation block

Single variable:

- add one compact derived comparison block that highlights candidate differentiators already present in metadata

Candidate differentiators may include:

- state-specific version
- district-created flag
- embedded assessment flag
- publisher presence vs missingness
- placeholder / unspecified risk

Acceptance focus:

- `catalog_state_specific_expected`
- `adoption_state_high_risk`
- `catalog_unspecified`
- `assessment_short_or_acronym_title`

### `F4` Abstention wording calibration

Single variable:

- adjust abstention guidance only while leaving schema and candidate payload unchanged

Intent:

- measure whether the current abstention wording is too permissive or too conservative for top-10 rerank recovery

Acceptance focus:

- `llm_selection_rate`
- `llm_top1_accuracy_on_all_rows`
- `llm_top1_accuracy_on_selected_rows`
- primary-reason mix for abstained rows

## Review Rules

For each `F` task:

1. mark exactly one task `in progress`
2. change only the named variable
3. run the batch rerank comparison against the accepted baseline
4. save scored summary, pipeline summary, and comparison artifact
5. write one short review note with keep / reject decision
6. update the baseline only if the task is explicitly accepted

Default rule:

- if results are mixed, reject the candidate and keep the current baseline

## What The Next Session Should Do

Start with `F1`.

Do not stack `F1` and `F2` together. If `F1` is rejected, compare `F2` against the still-accepted prompt baseline. If `F1` is accepted, then `F2` must compare against the newly accepted `F1` baseline.
