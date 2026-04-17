# Curriculum Matcher Handoff v054

- Date: `2026-04-17`
- Milestone: post-`H` experiment set defined
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

- read the latest `H4` closeout handoff
- read the live tracker, the `H` experiment-set definition, and the H1-H4 review notes
- preserved the accepted matcher, retrieval, shortlist, rerank prompt, model, reasoning, and Tkinter operator baselines
- defined the post-`H` `I` experiment set:
  - `docs/roadmap/2026-04-17-post-h-experiment-set.md`
- wrote the experiment-set review note:
  - `docs/analysis/2026-04-17-post-h-experiment-set-review.md`
- updated the live tracker:
  - `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

No matcher, retrieval, shortlist, rerank prompt, model, reasoning, or UI behavior changed.

## Why The I-Series Exists

The `H` series showed that simple candidate expansion is not enough:

- `H2` moved assessment shortlists but produced no top-10 gains.
- `H3` moved unspecified shortlists but produced no top-10 gains and regressed top-1.
- `H4` moved state-specific shortlists but produced no top-10 gains and slightly regressed top-1 / top-3.

The next likely bottleneck is not broad recall. It is the error taxonomy around `Assessment`, `catalog_unspecified`, state-specific overlap, and ambiguous historical ground truth.

## I-Series Roadmap

- `I1`: Assessment and catalog-unspecified taxonomy audit
- `I2`: Catalog-label consistency audit
- `I3`: Narrow label-normalization candidate
- `I4`: Taxonomy-informed retrieval candidate

`I1` and `I2` are diagnostic only. Do not change behavior during those checkpoints.

## Current Roadmap State

- `G1` is accepted
- `G2` is rejected
- `G3` is rejected
- `G4` is rejected
- `H1` is accepted
- `H2` is rejected
- `H3` is rejected
- `H4` is rejected
- `I1` is not started
- no experiment-changing task is currently in progress

## Recommended Next Step

Start exactly `I1`.

`I1` should review `Assessment` and `catalog_unspecified` labels to determine whether the active failures are retrieval misses, catalog-structure issues, ambiguous historical ground truth, or gold-present selection failures.

## Exact Next-Session Prompt

> Resume work on the curriculum matcher from `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-17-v054-post-h-experiment-set-defined-handoff.md`.
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
> - `I1` is not started
> - no experiment-changing task is currently in progress
>
> Please:
> 1. read the latest handoff, the live tracker, and the post-`H` `I` experiment-set definition,
> 2. keep the Tkinter operator surface in `src/curriculum_matcher/app.py`,
> 3. preserve the accepted matcher, retrieval, shortlist, model, reasoning, and `F2` rerank prompt baselines,
> 4. start exactly `I1`,
> 5. keep `I1` diagnostic-only with no runtime behavior change,
> 6. review `Assessment` and `catalog_unspecified` labels to determine whether the issue is retrieval, catalog structure, or ambiguous historical ground truth,
> 7. leave a tracker update, a review note, diagnostic artifact(s), and a fresh handoff before ending.
