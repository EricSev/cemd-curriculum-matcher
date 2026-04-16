# Post-G Experiment Set

- Date: `2026-04-16`
- Status: `ready`
- Scope: define the next benchmarked experiment set after the `G` series closeout
- Operator surface: keep using `src/curriculum_matcher/app.py`
- Workflow: benchmark-driven, one variable at a time

## Why A New Experiment Set Exists

The `G` series is closed:

- `G1` canonical prompt-pack alignment is `accepted`
- `G2` matcher-internal confidence label suppression is `rejected`
- `G3` derived ambiguity label suppression is `rejected`
- `G4` candidate series exposure is `rejected`

The `G` results show that more prompt input-contract trimming is not currently beating the accepted `F2` no-score rerank baseline. The next useful surface is candidate recall and shortlist quality before the LLM sees the rows.

## Locked Baseline To Preserve

Hold these fixed unless a task is explicitly accepted:

- matcher baseline: accepted `C1`, `C2`, `C5`; rejected `C3`, `C4`
- retrieval baseline: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist size: `10`
- rerank model: `gpt-5.4-mini`
- reasoning effort: `medium`
- rerank prompt baseline: `F2` no-score candidate payload
- comparison baseline: `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`
- operator surface: `src/curriculum_matcher/app.py`

Do not reopen rejected `E`, `F`, or `G` tasks inside this phase unless new measured evidence justifies it.

## Why The H-Series Focuses On Candidate Recall

The strongest remaining bottlenecks are not broad prompt failures:

- `Assessment` remains the worst usage-family slice in the representative benchmark
- `catalog_unspecified` remains a major warning slice
- `adoption_state_high_risk` and `catalog_state_specific_expected` continue to behave differently from standard rows
- `G2` through `G4` did not improve these areas by hiding or exposing different prompt fields

That suggests the next set should test whether the right catalog rows are entering the top-10 shortlist reliably enough for the accepted LLM reranker to recover them.

## H-Series Goal

Improve shortlist quality for the known hard slices without changing the accepted LLM prompt baseline.

The promotion bar stays conservative:

- improve `hit@10` or targeted slice top-10 coverage without reducing overall top-1 materially
- preserve or improve downstream LLM all-row top-1 and selected-row top-1 when a rerank batch is required
- avoid adding invalid or repaired selected-id risk

## Experiment Sequence

### `H1` Shortlist failure audit checkpoint

Single variable:

- reporting / diagnosis only; no matcher, retrieval, shortlist, prompt, model, or reasoning change

Intent:

- split current hard rows into two buckets:
  - gold catalog row absent from top-10, which is a retrieval / candidate-recall failure
  - gold catalog row present in top-10 but not selected, which is a rerank selection failure
- report those buckets for `Assessment`, `catalog_unspecified`, `adoption_state_high_risk`, and `catalog_state_specific_expected`

Acceptance focus:

- produces a stable diagnostic artifact for deciding whether `H2` should target candidate generation, scoring, or rerank behavior
- no behavioral baseline change

### `H2` Assessment-aware candidate recall experiment

Single variable:

- assessment-row candidate recall policy only

Intent:

- test whether rows whose district-facing usage is `Assessment` benefit from a candidate recall adjustment that gives assessment-like catalog rows a better chance to enter the top-10
- keep non-assessment rows on the locked retrieval baseline

Acceptance focus:

- `Assessment` hit@10
- `assessment_short_or_acronym_title` hit@10 and top-1
- `assessment_state_specific_expected`
- overall hit@10 and top-1 must not materially regress

### `H3` Catalog-unspecified recovery experiment

Single variable:

- conservative candidate handling for `Unspecified` / placeholder-like catalog families only

Intent:

- test whether sparse district rows that map to `Unspecified` catalog entries need a separate candidate-recall rule rather than more LLM prompt context
- do not use gold labels at runtime; derive any candidate policy from district row evidence and catalog metadata only

Acceptance focus:

- `catalog_unspecified` hit@10 and top-1
- sparse and medium evidence rows
- invalid / repaired selected-id risk if sent through the LLM reranker

### `H4` State-specific candidate balancing experiment

Single variable:

- candidate balancing for state-specific catalog rows only

Intent:

- test whether adoption-state and state-specific expected rows need a targeted candidate recall rule so state-specific and non-state-specific near-neighbors are both visible to the final ranker

Acceptance focus:

- `adoption_state_high_risk`
- `catalog_state_specific_expected`
- assessment rows inside adoption states
- overall top-1, hit@10, and downstream LLM selection quality

## Review Rules

For each `H` task:

1. mark exactly one task `in progress`
2. change only the named variable
3. run the representative benchmark first when the task affects retrieval or candidate generation
4. send an LLM rerank batch only if the shortlist movement is strong enough to justify the cost
5. save summary, records, comparison, and slice artifacts
6. write one short review note with keep / reject decision
7. update the baseline only if the task is explicitly accepted

Default rule:

- if results are mixed, reject the candidate and keep the current baseline

## Recommended First Move

Start with `H1`.

`H1` is a no-behavior diagnostic checkpoint. It should clarify whether the next real experiment should prioritize candidate recall, candidate scoring, or the accepted LLM reranker.
