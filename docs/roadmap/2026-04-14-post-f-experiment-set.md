# Post-F Experiment Set

- Date: `2026-04-14`
- Status: `ready`
- Scope: define the next benchmarked experiment set after the `F` series closeout
- Operator surface: keep using `src/curriculum_matcher/app.py`

## Why A New Experiment Set Exists

The `F` series is closed:

- `F1` system prompt tightening is `rejected`
- `F2` raw matcher-score exposure is `accepted`
- `F3` candidate disambiguation block is `rejected`
- `F4` abstention wording calibration is `rejected`

That means the accepted rerank prompt baseline is now the `F2` no-score payload policy in `src/curriculum_matcher/llm_rerank.py`.

The next phase should not keep churning on high-level prompt wording. That seam has already been tested directly. The cleaner remaining surface is the prompt input contract itself:

- which row-context fields should be exposed to the LLM
- which candidate metadata fields actually help
- whether the accepted baseline comparison workflow should be anchored to a canonical prompt-pack row set before more behavioral experiments

## Locked Baseline To Preserve

Hold these fixed for every `G`-series task unless the tracker explicitly says otherwise:

- matcher baseline: accepted `C1`, `C2`, `C5`; rejected `C3`, `C4`
- retrieval baseline: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist size: `10`
- rerank model: `gpt-5.4-mini`
- reasoning effort: `medium`
- rerank prompt baseline: `F2` no-score candidate payload
- operator surface: `src/curriculum_matcher/app.py`

Do not reopen `E1`, `E2`, or rejected `F` tasks inside this phase unless new measured evidence justifies it.

## Why The G-Series Focuses On Prompt Inputs

The accepted `F2` change improved recovery by removing raw matcher-score anchoring, which suggests the model is sensitive to what evidence shape it sees, not just how it is instructed.

Two unresolved issues remain:

1. The accepted historical scored artifact and the current error-row prompt pack still do not align one-to-one, so recent decisions depended on shared-row comparisons instead of a fully row-matched baseline.
2. The current `DISTRICT_ROW` payload includes several derived matcher or analysis labels such as `confidence_band`, `match_selected_strategy`, `usage_ambiguity`, `state_specific_risk`, `placeholder_mapping`, and `assessment_slice`. Those fields may be helping, hurting, or leaking matcher-internal framing into the rerank step.

## G-Series Goal

Test prompt input-contract changes one variable at a time before returning to larger architecture or model questions.

These tasks should run against the accepted `F2` no-score rerank baseline and preserve the locked matcher, retrieval, shortlist, model, and reasoning settings.

## Experiment Sequence

### `G1` Canonical prompt-pack alignment checkpoint

Single variable:

- reporting / comparison contract only; no prompt behavior change

Intent:

- freeze one accepted `F2` prompt pack and its scored output as the official row-matched comparison base for subsequent rerank experiments
- remove avoidable ambiguity caused by overlap-only comparisons against older historical artifacts

Acceptance focus:

- row-count parity between accepted baseline prompt pack, batch output, and scored summary
- a stable baseline artifact set that later `G` tasks can compare against directly
- no rerank behavior change in source code

### `G2` Matcher-internal confidence label suppression

Single variable:

- remove `confidence_band` and `match_selected_strategy` from `DISTRICT_ROW`

Intent:

- test whether matcher-internal confidence and repair-path labels are anchoring the LLM too strongly to the baseline matcher behavior

Acceptance focus:

- `llm_top1_accuracy_on_all_rows`
- `llm_top1_accuracy_on_selected_rows`
- wrong-top1-but-gold-in-shortlist recovery
- selection-rate stability

### `G3` Derived ambiguity label suppression

Single variable:

- remove derived row labels `usage_ambiguity`, `state_specific_risk`, `placeholder_mapping`, and `assessment_slice` while keeping raw district evidence unchanged

Intent:

- test whether analysis-layer taxonomy labels are helping the model reason or over-steering it toward canned ambiguity narratives

Acceptance focus:

- `catalog_state_specific_expected`
- `adoption_state_high_risk`
- `catalog_unspecified`
- `assessment_short_or_acronym_title`

### `G4` Candidate series exposure experiment

Single variable:

- remove candidate `series` from `SHORTLIST_CANDIDATES` while keeping product name, publisher, grades, year, product type, and current boolean metadata unchanged

Intent:

- test whether `series` is acting as useful disambiguation context or as noisy family-level anchoring that pulls the model toward near-neighbor false positives

Acceptance focus:

- near-duplicate family rows
- wrong-top1-but-gold-in-shortlist recovery
- selected-row precision
- assessment-heavy slices where family naming is often repetitive

## Review Rules

For each `G` task:

1. mark exactly one task `in progress`
2. change only the named variable
3. run the batch rerank comparison against the accepted `F2` baseline
4. save prompt summary, scored summary, pipeline summary, and comparison artifact
5. write one short review note with keep / reject decision
6. update the baseline only if the task is explicitly accepted

Default rule:

- if results are mixed, reject the candidate and keep the current baseline

## Recommended First Move

Start with `G1`.

`G1` is not a new behavior change. It is a comparison-contract checkpoint that should make every later rerank decision easier to interpret. Once the canonical `F2` row-matched baseline is saved, lift exactly `G2` first among behavioral experiments.
