# Post-H Experiment Set

- Date: `2026-04-17`
- Status: `ready`
- Scope: define the next benchmarked experiment set after the `H` series closeout
- Operator surface: keep using `src/curriculum_matcher/app.py`
- Workflow: diagnosis first, then benchmark-driven one-variable experiments

## Why A New Experiment Set Exists

The `H` series is closed:

- `H1` shortlist failure audit checkpoint is `accepted`
- `H2` assessment-aware candidate recall is `rejected`
- `H3` catalog-unspecified recovery is `rejected`
- `H4` state-specific candidate balancing is `rejected`

The rejected `H2` through `H4` candidates all changed shortlist composition but produced no top-10 gold-row recovery. That is useful negative evidence: simple metadata-gated stage-1 expansion is not enough. The next phase should stop broadening retrieval and instead inspect whether the hard rows are being labeled, grouped, or judged against ambiguous historical ground truth in a way that hides the true failure mode.

## Locked Baseline To Preserve

Hold these fixed unless a later task is explicitly accepted:

- matcher baseline: accepted `C1`, `C2`, `C5`; rejected `C3`, `C4`
- retrieval baseline: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist size: `10`
- rerank model: `gpt-5.4-mini`
- reasoning effort: `medium`
- rerank prompt baseline: `F2` no-score candidate payload
- comparison baseline: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`
- operator surface: `src/curriculum_matcher/app.py`

Do not reopen rejected `H` retrieval-expansion tasks unless a taxonomy audit shows a narrower, testable reason they failed.

## Why The I-Series Focuses On Taxonomy

The current hard slices may be mixing several different problems under the same labels:

- `Assessment` rows include very short titles, missing publishers, state-specific test names, and placeholder-like product names.
- `catalog_unspecified` rows may represent valid catalog structure, policy placeholder rows, district-created rows, or historical ground-truth ambiguity.
- `adoption_state_high_risk` and `catalog_state_specific_expected` overlap with assessment and unspecified cases, but the `H4` state-specific retrieval rule did not improve them.
- `G3` showed that removing derived ambiguity labels from the rerank payload hurt downstream selection, so those labels may be useful for the LLM while still being too coarse for experiment design.

The next set should separate error taxonomy from behavior change. First determine which rows are true retrieval misses, catalog-structure gaps, ambiguous labels, or rerank-choice failures. Then test only one narrow behavior at a time.

## I-Series Goal

Produce a cleaner error taxonomy for assessment and placeholder/catalog-unspecified cases, then use it to choose a narrowly scoped candidate experiment.

The promotion bar stays conservative:

- diagnostic tasks must produce reusable artifacts and no runtime behavior change
- behavior-changing tasks must improve the targeted slice without material overall top-1 regression
- no LLM rerank batch should run unless shortlist movement recovers gold rows or isolates a gold-present selection problem
- accepted baseline changes must remain compatible with the Tkinter operator surface

## Experiment Sequence

### `I1` Assessment and catalog-unspecified taxonomy audit

Single variable:

- reporting / diagnosis only; no matcher, retrieval, shortlist, prompt, model, or reasoning change

Intent:

- review rows in `Assessment` and `catalog_unspecified` slices across the accepted representative top-10 records
- split failures into taxonomy buckets:
  - likely retrieval miss
  - likely catalog-label / placeholder-structure issue
  - likely ambiguous historical ground truth
  - gold present but selected ranker or LLM chose another candidate
- report overlaps among `assessment_slice`, `placeholder_mapping`, `state_specific_risk`, `evidence_richness`, publisher presence, and top-10 status

Acceptance focus:

- produces a stable diagnostic artifact for deciding whether `I2` should target labels, catalog structure, retrieval scoring, or rerank behavior
- no behavioral baseline change

### `I2` Catalog-label consistency audit

Single variable:

- catalog/benchmark label audit only; no runtime matcher behavior change

Intent:

- inspect whether `Unspecified`, no-information, district-created, and assessment-like catalog rows are consistently represented in the catalog and benchmark labels
- identify cases where the gold catalog row appears structurally valid but the derived label collapses distinct meanings into `catalog_unspecified`
- identify cases where the historical gold row itself may be too ambiguous to use as a clean retrieval target

Acceptance focus:

- produces row-level examples and count summaries
- recommends either a label-normalization experiment, a benchmark-ground-truth review, or no action

### `I3` Narrow label-normalization candidate

Single variable:

- one derived-label normalization rule only, chosen from `I1` / `I2`

Intent:

- test whether a single taxonomy correction improves downstream diagnosis or rerank context without changing candidate generation
- do not change stage-1 retrieval, shortlist size, or candidate scoring

Acceptance focus:

- targeted hard-slice interpretability improves
- if sent through LLM rerank, all-row top-1 and selected-row top-1 must not regress
- no increase in invalid or repaired selected ids

### `I4` Taxonomy-informed retrieval candidate

Single variable:

- one retrieval or shortlist rule only, chosen from accepted `I1` / `I2` evidence

Intent:

- only after taxonomy evidence exists, test one narrow retrieval adjustment aimed at a proven failure class
- avoid broad stage-1 expansion patterns from `H2`, `H3`, and `H4`

Acceptance focus:

- targeted hit@10 recovery for the audited failure class
- overall top-1 stability
- downstream LLM batch only if top-10 recovery or gold-present selection evidence justifies it

## Review Rules

For each `I` task:

1. mark exactly one task `in progress`
2. change only the named variable
3. preserve the locked baseline unless the task is explicitly accepted
4. for diagnostic tasks, save Markdown and JSON or CSV artifacts
5. for behavior-changing tasks, run the representative benchmark before any LLM batch
6. send an LLM rerank batch only if shortlist evidence justifies the cost
7. write one short review note with keep / reject / diagnostic-accepted decision

Default rule:

- if results are mixed, reject the candidate and keep the current baseline

## Recommended First Move

Start with `I1`.

`I1` is a no-behavior diagnostic checkpoint. It should determine whether the next real experiment belongs in retrieval, catalog-label cleanup, benchmark ground-truth review, or the accepted LLM reranker.
