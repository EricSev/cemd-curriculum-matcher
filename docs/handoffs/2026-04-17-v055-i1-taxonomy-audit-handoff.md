# Curriculum Matcher Handoff v055

- Date: `2026-04-17`
- Milestone: `I1` measured and accepted as a diagnostic checkpoint
- Operator surface: keep using `src/curriculum_matcher/app.py`
- Workflow: diagnosis first, then benchmark-driven one-variable experiments

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
- `H2` assessment-aware candidate recall
- `H3` catalog-unspecified recovery
- `H4` state-specific candidate balancing

Accepted retrieval and rerank baseline:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`
- rerank prompt baseline: `F2` no-score candidate payload

Accepted comparison-contract baseline:

- use `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`

## What This Session Did

- started exactly `I1`
- kept `I1` diagnostic-only
- preserved the accepted matcher, retrieval, shortlist, rerank prompt, model, reasoning, and Tkinter operator baselines
- added a reusable taxonomy audit script:
  - `scripts/report_i1_taxonomy_audit.py`
- ran the audit against:
  - accepted top-10 char n-gram records
  - canonical `F2` scored CSV
  - the `07122025` product catalog used by the current benchmark workflow
- saved the `I1` artifacts:
  - `docs/analysis/2026-04-17-i1-taxonomy-audit.md`
  - `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_i1_taxonomy_audit.json`
  - `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_i1_taxonomy_audit_rows.csv`
- updated the live tracker:
  - `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

No matcher, retrieval, shortlist, rerank prompt, model, reasoning, or UI behavior changed.

## I1 Decision Summary

The focal audit covered `316` rows across `Assessment` and `catalog_unspecified`.

Primary bucket counts:

- `already_correct_or_recovered`: `127`
- `ambiguous_historical_ground_truth`: `135`
- `catalog_label_or_placeholder_structure`: `38`
- `gold_present_selection_failure`: `10`
- `likely_retrieval_miss`: `6`

Required slice readout:

- `Assessment`: `260` rows
  - `135` primary `ambiguous_historical_ground_truth`
  - `114` `already_correct_or_recovered`
  - `6` clean `likely_retrieval_miss`
  - `4` primary `gold_present_selection_failure`
  - `1` `catalog_label_or_placeholder_structure`
- `catalog_unspecified`: `56` rows
  - `37` primary `catalog_label_or_placeholder_structure`
  - `13` `already_correct_or_recovered`
  - `6` primary `gold_present_selection_failure`

Why `I1` was accepted:

- it produced reusable JSON, Markdown, and row-level CSV artifacts
- it clarified that most assessment misses are not clean retrieval-expansion targets
- it clarified that `catalog_unspecified` should be inspected as catalog-label / placeholder structure before another retrieval tweak
- it changed no runtime behavior

## Current Roadmap State

- `G1` is accepted
- `G2` is rejected
- `G3` is rejected
- `G4` is rejected
- `H1` is accepted
- `H2` is rejected
- `H3` is rejected
- `H4` is rejected
- `I1` is accepted
- `I2` is not started
- no experiment-changing task is currently in progress

## Recommended Next Step

Start exactly `I2`.

`I2` should remain diagnostic-only and inspect catalog-label consistency for `Unspecified`, no-information, district-created, and assessment-like catalog rows. Do not change retrieval or rerank behavior until `I2` determines whether the issue is catalog structure, label normalization, or benchmark ground-truth ambiguity.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-17-v055-i1-taxonomy-audit-handoff.md`.
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
> - `G2` / `G3` / `G4` are rejected
> - `H1` is accepted as diagnostic
> - `H2` / `H3` / `H4` are rejected
> - `I1` is accepted as diagnostic
> - `I2` is not started
> - no experiment-changing task is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, the post-`H` `I` experiment-set definition, and the `I1` taxonomy audit,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, shortlist, model, reasoning, and `F2` rerank prompt baselines,
> 4. start exactly `I2`,
> 5. keep `I2` diagnostic-only with no runtime behavior change,
> 6. inspect catalog-label consistency for `Unspecified`, no-information, district-created, and assessment-like catalog rows,
> 7. leave a tracker update, a review note, diagnostic artifact(s), and a fresh handoff before ending.
