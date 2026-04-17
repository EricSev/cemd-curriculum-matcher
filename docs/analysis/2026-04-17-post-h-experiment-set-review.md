# Post-H Experiment Set Review

- Date: `2026-04-17`
- Task: define the next experiment set after the `H` series closeout
- Decision: `accepted as roadmap`
- Behavior changed: `no`
- Roadmap definition: `docs/roadmap/2026-04-17-post-h-experiment-set.md`

## Inputs Reviewed

- latest handoff: `docs/handoffs/2026-04-16-v053-h4-rejected-h-series-closed-handoff.md`
- live tracker: `docs/roadmap/2026-04-05-next-phase-review-task-list.md`
- H-series definition: `docs/roadmap/2026-04-16-next-experiment-set.md`
- H1 diagnostic: `docs/analysis/2026-04-16-shortlist-failure-audit.md`
- H2 rejection: `docs/analysis/2026-04-16-assessment-recall-review.md`
- H3 rejection: `docs/analysis/2026-04-16-unspecified-recall-review.md`
- H4 rejection: `docs/analysis/2026-04-16-state-balance-review.md`

## Rationale

The `H` series established that broad metadata-gated candidate expansion is not moving the accepted benchmark:

- `H2` changed `51` shortlists but recovered `0` additional top-10 gold rows.
- `H3` changed `35` shortlists, regressed overall top-1 by `-0.0030`, and recovered `0` additional top-10 gold rows.
- `H4` changed `13` shortlists, regressed overall top-1 by `-0.0010`, and recovered `0` additional top-10 gold rows.

That points away from another immediate retrieval expansion and toward a more careful taxonomy pass. The next phase should determine whether `Assessment` and `catalog_unspecified` failures are true retrieval misses, catalog-structure problems, ambiguous historical labels, or gold-present selection failures.

## Decision

Accept the new `I` series as the next roadmap:

- `I1`: Assessment and catalog-unspecified taxonomy audit
- `I2`: Catalog-label consistency audit
- `I3`: Narrow label-normalization candidate
- `I4`: Taxonomy-informed retrieval candidate

No matcher, retrieval, shortlist, rerank prompt, model, reasoning, or Tkinter operator behavior changed.

## Current Baseline

The accepted runtime baseline remains:

- matcher `C1 + C2 + C5`
- retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- rerank prompt `F2` no-score candidate payload
- model `gpt-5.4-mini`
- reasoning `medium`

## Recommended Next Step

Start exactly `I1`.

Do not change retrieval or rerank behavior until the taxonomy audit identifies a narrower failure class worth testing.
