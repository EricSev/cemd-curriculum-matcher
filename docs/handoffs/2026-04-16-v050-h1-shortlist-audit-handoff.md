# Curriculum Matcher Handoff v050

- Date: `2026-04-16`
- Milestone: `H1` measured and accepted as a diagnostic checkpoint
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
- `G4` candidate series exposure experiment

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline: `F2` no-score candidate payload

Accepted comparison-contract baseline:

- use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## What This Session Did

- pushed the completed `G4` closeout commit to `origin/codex/g-series-closeout`
- defined the `H` experiment set:
  - `docs/roadmap/2026-04-16-next-experiment-set.md`
- added a reusable H1 audit script:
  - `scripts/report_shortlist_failure_audit.py`
- ran `H1` against the accepted representative top-10 char n-gram records and canonical `F2` LLM scored rows
- wrote the H1 review note:
  - `docs/analysis/2026-04-16-shortlist-failure-audit.md`
- saved the H1 JSON artifact:
  - `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_h1_shortlist_failure_audit.json`
- updated the live tracker:
  - `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

No matcher, retrieval, shortlist, rerank prompt, model, or reasoning behavior changed.

## H1 Decision Summary

`H1` separated top-10 recall failures from selection failures for the required hard slices.

Representative records:

- rows: `1000`
- overall gold absent from top-10: `372`
- overall gold present in top-10: `628`
- overall ranker top-1 failure with gold present: `140`

Required slice readout:

- `Assessment`: `142 / 260` rows absent from top-10; `23` gold-present ranker failures
- `catalog_unspecified`: `37 / 56` rows absent from top-10; `14` gold-present ranker failures
- `adoption_state_high_risk`: `148 / 349` rows absent from top-10; `32` gold-present ranker failures
- `catalog_state_specific_expected`: `16 / 53` rows absent from top-10; `12` gold-present ranker failures

Joined canonical `F2` LLM rows:

- joined rows: `455`
- overall gold absent from top-10: `315`
- overall gold present in top-10: `140`
- LLM selection failures with gold present: `50`

Decision:

- accept `H1` as a diagnostic checkpoint
- no behavioral baseline change
- prioritize `H2` assessment-aware candidate recall next

Why:

- `Assessment` is heavily recall-limited: more than half of representative assessment rows do not have the gold catalog row in the top-10
- `catalog_unspecified` is also recall-limited, but the `H` sequence starts with assessment because it is larger and was already the worst usage-family slice
- state-specific rows have a mixed failure shape and can wait until `H4`

## Current Roadmap State

- `G1` is accepted
- `G2` is rejected
- `G3` is rejected
- `G4` is rejected
- `H1` is accepted
- `H2` is not started
- no experiment-changing task is currently in progress

## Recommended Next Step

Start `H2` only.

`H2` should test one assessment-row candidate recall policy while preserving:

- accepted matcher baseline `C1 + C2 + C5`
- accepted retrieval blend unless the single H2 variable explicitly wraps assessment candidate recall
- shortlist `10`
- accepted `F2` no-score rerank prompt
- model `gpt-5.4-mini`
- reasoning `medium`

Run the representative benchmark before considering an LLM batch. Send a rerank batch only if the assessment shortlist movement is strong enough to justify cost.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-16-v050-h1-shortlist-audit-handoff.md`.
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
> - `G4` is rejected
> - `H1` is accepted
> - `H2` is not started
> - no experiment-changing task is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, the `H` experiment-set definition, and the `H1` shortlist failure audit,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, shortlist, model, reasoning, and `F2` rerank prompt baselines except for the single H2 candidate-recall variable,
> 4. start exactly `H2`,
> 5. run the representative benchmark before any LLM batch,
> 6. decide whether H2 produces enough shortlist movement to justify a rerank batch,
> 7. leave a tracker update, a review note, and a fresh handoff before ending.
