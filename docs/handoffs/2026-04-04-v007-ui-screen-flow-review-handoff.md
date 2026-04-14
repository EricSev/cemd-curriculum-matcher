# UI Screen Flow Review Handoff

- Date: 2026-04-04
- Version: v007
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`
- Next-session focus: use the reviewed collection UI evidence to sharpen benchmark slicing, QA reasoning, and the plan for assessment-specific analysis

## What changed this session

- Reviewed the archived screenshot document:
  - `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/99-archive/superseded-docs/Sprint 1/Curriculum Collection Screen flow.docx`
- Extracted the embedded screenshots and inspected representative UI states.
- Confirmed that the collection application UI is broadly consistent with the training manuals on the following high-value points:
  - top-level record filters expose `Selection Subject`, `Selection Grade Level`, `Selection Type`, and `Record Status`
  - the dashboard visually separates `Find new curriculum` from `Validate existing curriculum`
  - workers edit records through a multi-tab modal with `Details`, `Product`, `Update Product`, and `Notes / Comments`
  - `Selection Type` is a first-class worker-entered field in the `Details` tab
  - `Product name from district`, `Publisher from district`, `Copyright year`, and `State Specific Curriculum Version` are entered separately from catalog selection
  - the chosen catalog match appears as a separate `Selected Product`
  - the `Update Product` tab uses a searchable catalog table with `product name`, `product type`, `series`, `subject`, `grades`, `copyright`, and `publisher`

## Important UI observations

### 1. The UI separates raw district evidence from normalized catalog mapping

The screenshots make the workflow explicit:

- workers first capture district-facing details
- then they map that evidence to a catalog item

This matters because historical labels may encode:

- valid raw district text plus an imperfect catalog selection
- sparse raw district evidence with a reasonable but ambiguous catalog match
- placeholder catalog choices like `Unspecified`

That separation reinforces the earlier recommendation to distinguish evidence richness from catalog-match quality in benchmark analysis.

### 2. Selection Type is prominent and editable

The UI confirms that `Selection Type` is not hidden metadata. It sits on the main `Details` tab alongside subject and grade level. That makes it even clearer that:

- workers are expected to consciously encode district usage
- usage-type mistakes are likely meaningful label behavior, not accidental backend defaults

### 3. Record Status appears operationally important

The screenshots show `Record Status` at the top of the modal and the dashboard grid. Visible statuses include:

- `Not yet started`
- `Ready for review`

The manuals also refer to `On Hold`, and the product tab includes curriculum-status choices such as:

- `Validated / In Use`
- `Curriculum / Product has changed`
- `No curriculum information available`

This suggests there are at least two layers of status:

- workflow/review progress status
- curriculum evidence status

That distinction is important for future QA interpretation. We should avoid conflating reviewer workflow state with evidence certainty.

### 4. The catalog-search screen likely contributes to label noise

The screenshot document itself opens with a written note on search pain points:

- exact-string sensitivity
- punctuation sensitivity
- identical titles across publishers
- publisher rebrand confusion
- hard copyright-year choices
- overuse of `Other` when the product is actually present

The screenshots support that diagnosis because the search UI looks like a plain table search over many near-duplicate rows. In the examples:

- `Wonders` appears in multiple copyright years
- state-specific variants are mixed into the same result list
- `Unspecified` appears as a separate selectable row

Implication:

- some benchmark error is likely UI-mediated selection difficulty rather than pure worker misunderstanding

### 5. State-specific variation is visibly part of worker choice

The UI shows both:

- a `State Specific Curriculum Version` toggle on the product-entry side
- state-specific rows inside catalog search results

This supports keeping or adding an explicit benchmark slice for state-specific variant risk, especially for CA, FL, and TX.

## Cautions

- The reviewed file lives under `superseded-docs`, so it should be treated as supporting context, not the canonical current product spec.
- The screenshots appear to reflect a May 20, 2025 workflow snapshot.
- Even so, the main structures line up well with the current manuals and strengthen confidence in the policy artifact already added this session.

## Artifacts created this and prior session

- policy artifact:
  - `docs/policy/data-collection-rules.md`
- policy implications report:
  - `docs/analysis/2026-04-04-collection-policy-implications-report.md`

## Recommended next-session deliverables

1. Convert the policy and UI findings into concrete benchmark-slice definitions.
2. Propose a revised QA reason taxonomy that separates:
   - evidence sparsity
   - usage ambiguity
   - catalog-search ambiguity
   - assessment-specific alias/subtype ambiguity
3. Decide whether to create:
   - a dedicated `Assessment` benchmark slice only, or
   - a separate evaluation/treatment path for assessment rows.
4. If needed, review assessment-specific training materials next, since current evidence strongly suggests assessment is governed by a distinct workflow.

## Recommended decisions for next session

- Keep one overall representative benchmark, but add policy-aware slices.
- Add a distinct slice for catalog-selection difficulty, especially:
  - duplicate-title multi-publisher products
  - multi-copyright rows
  - unspecified placeholders
  - state-specific variants
- Treat `Assessment` as first-class in reporting immediately.
- Strongly consider a separate matcher treatment path for `Assessment` after reviewing the dedicated assessment training.

## Agent-Optimized Kickoff Prompt

Resume work on the curriculum matcher from these artifacts:

- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-04-v007-ui-screen-flow-review-handoff.md`
- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/policy/data-collection-rules.md`
- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/analysis/2026-04-04-collection-policy-implications-report.md`

Current priority is to turn the reviewed collection policy and UI evidence into concrete evaluation recommendations without changing matcher logic yet.

Please:
1. review the handoff, policy artifact, and policy implications report,
2. use the representative historical benchmark artifacts as the reference benchmark:
   - `benchmarks/gold/historical_07122025_representative_1000.csv`
   - `benchmarks/outputs/historical_07122025_representative_1000_fast_summary.json`
   - `benchmarks/outputs/historical_07122025_representative_1000_fast_records.csv`
3. propose a concrete benchmark slicing plan based on policy and UI realities, including:
   - evidence richness
   - usage ambiguity
   - state-specific risk
   - placeholder / unspecified mapping
   - assessment-specific rows
4. propose a revised QA challenge-reason taxonomy that distinguishes:
   - likely matcher error
   - likely evidence sparsity
   - likely UI/catalog-selection ambiguity
   - likely assessment alias or subtype ambiguity
5. make a recommendation on whether `Assessment` should remain a first-class slice inside one benchmark framework or move toward a separate treatment path,
6. if necessary, identify the minimum additional assessment-specific training materials to review next.

Constraints:
- do not start by changing matcher logic
- keep the work artifact-driven
- keep recommendations benchmark- and QA-oriented
- treat the reviewed screenshot doc as supporting UI evidence, not the current source of truth

Expected outputs for the next session:
- a concrete benchmark slicing proposal
- a concrete QA reason taxonomy proposal
- a recommendation on `Assessment` handling
- a short note on any additional assessment-training review that is still needed
