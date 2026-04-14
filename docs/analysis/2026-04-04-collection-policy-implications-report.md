# Collection Policy Implications Report

## Purpose

This report links the worker training rules to current benchmark behavior, especially on the representative historical benchmark:

- `benchmarks/gold/historical_07122025_representative_1000.csv`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_records.csv`

## Current Benchmark Context

- overall top-1 accuracy: `0.46`
- overall top-3 recall: `0.561`
- overall prediction rate: `0.932`
- by product usage:
  - `Supplemental` top-1: `0.625`
  - `Core Curriculum` top-1: `0.432`
  - `Assessment` top-1: `0.2962`

Additional record-level context from the current output:

- `Assessment`
  - publisher missing rate: `0.8385`
  - no-prediction rate: `0.1231`
  - non-primary strategy rate: `0.5462`
  - low/very-low/no-prediction confidence rate: `0.8346`
- `Core Curriculum`
  - publisher missing rate: `0.3689`
  - no-prediction rate: `0.0413`
  - non-primary strategy rate: `0.1529`
  - low/very-low/no-prediction confidence rate: `0.2597`
- `Supplemental`
  - publisher missing rate: `0.7622`
  - no-prediction rate: `0.0579`
  - non-primary strategy rate: `0.2805`
  - low/very-low/no-prediction confidence rate: `0.5213`

## What The Collection Policy Changes In Interpretation

### 1. Benchmark labels encode district-reported usage, not just product identity

The manuals repeatedly instruct workers to tag products based on how the district says they are used. That means some apparent matcher errors are not simple retrieval failures. They may instead be failures to recover district usage context from weak raw text.

Implication:

- benchmark evaluation should keep `product_type_usage` and `selection_type` logic front and center
- false positives where the title is right but the usage context is wrong should be treated as meaningful errors, not near-misses

### 2. Human labels tolerate sparse metadata

Workers are allowed to save valid rows with missing publisher, missing copyright, missing grade detail, and sometimes publisher-only or district-created placeholders. This means low-information rows are not annotation mistakes by default.

Implication:

- benchmark slices should explicitly separate rich-evidence rows from sparse-evidence rows
- matcher misses on sparse rows should not be interpreted the same way as misses on rows with title, publisher, grade, and state-specific detail

### 3. The collection process is district-website bounded

Workers do not use general web evidence except through district-linked materials. `No curriculum information available` is a bounded-search policy outcome, not proof of product absence.

Implication:

- QA should distinguish between likely model error and likely evidence-bound ambiguity
- challenge reasons should be able to say that a row is weak because the district source itself is weak, not because the candidate set is obviously wrong

## Implications For Benchmark Design

### Recommended benchmark slices

Add or formalize the following slices:

- `usage_family`
  - `Core Curriculum`
  - `Supplemental`
  - `Assessment`
- `evidence_richness`
  - raw title only
  - title plus publisher
  - title plus publisher plus grade/variant
- `district_usage_clarity`
  - explicit core/supplemental wording
  - inferred from context
  - ambiguous usage wording
- `state_specific_risk`
  - CA/TX/FL
  - other adoption states
  - open territory
- `placeholder_or_fallback_label`
  - district-created
  - publisher-only unspecified
  - other/unspecified

Why:

- the manuals make these distinctions operationally real
- current topline benchmark metrics blend together policy-easy and policy-hard cases

### Recommended benchmark policy note

Every benchmark report should state that correctness is judged against district-reported usage labels collected under bounded website-only evidence rules. This will prevent over-reading some misses as pure model quality failures.

## Implications For Matcher Error Interpretation

### Core Curriculum

Core has better structure and stronger district documentation expectations. Errors here are more likely to be real ranking or normalization failures because the worker instructions assume relatively clearer evidence and fuller course structure.

### Supplemental

Supplemental is expected to be sparse and heterogeneous, but the benchmark still performs best here. That is consistent with many supplemental products being distinctive brand names even when metadata is thin.

Interpretation:

- stronger supplemental performance does not mean the task is easier in policy terms
- it likely means many supplemental labels are easier to identify lexically once found

### Assessment

Assessment should not be interpreted as just another low-information slice. Training materials explicitly say assessment collection is a separate process with separate training. `CEMD Data Elements 2025.md` also shows assessment-specific subtags that do not appear in the core/supplemental manuals.

Likely causes of current weakness:

- raw titles are often short acronyms or generic programs like `MAP`, `PSAT`, `CAASPP`, `ELPAC`, `FAST`, `STAR`
- publisher is usually missing
- district usage is often state-program-like or assessment-window-like rather than product-catalog-like
- assessment families appear to involve many aliases and umbrella-to-subassessment relationships
- the current benchmark likely mixes multiple assessment subtypes that should not be treated as one homogeneous family

In short, `Assessment` looks weaker partly because it is a different policy regime and a different naming regime.

## Implications For QA Triage And Challenge Reasons

Current QA reasons are mostly model-confidence oriented. The training review suggests adding policy-aware reasons.

### Recommended QA challenge reasons

- `usage_type_ambiguous_on_district_site`
- `district_source_is_static_or_context_thin`
- `raw_title_too_generic_for_catalog_mapping`
- `publisher_missing_but_allowed_by_collection_policy`
- `publisher_only_unspecified_candidate_needed`
- `district_created_placeholder_expected`
- `state_specific_variant_policy_risk`
- `assessment_subtype_or_alias_ambiguity`

### Recommended QA triage adjustments

- route `Assessment` rows with acronym-only titles or missing publisher to a dedicated review bucket
- distinguish `likely matcher error` from `likely policy/evidence ambiguity`
- sample `No curriculum information available` rows separately from ordinary mismatches because they reflect bounded search policy rather than a normal mapping problem

## Likely Failure Families

The manuals and benchmark together suggest these failure families:

- usage-type mismatch
  - right product family, wrong `Selection Type`
- sparse-evidence mapping
  - title-only or acronym-only rows
- state-specific variant confusion
  - especially CA, TX, FL
- placeholder mapping
  - district-created or publisher-only unspecified rows
- assessment alias hierarchy
  - umbrella test family vs specific form or subtype
- district-source context loss
  - static file, weak page context, or thin wording

## Why Assessment Is Much Weaker Than Supplemental Or Core

The best explanation from the reviewed materials is not simply that assessment rows are noisier. It is that they appear to be collected under a distinct policy and ontology that the current benchmark analysis is not modeling explicitly.

Signals supporting that conclusion:

- assessment has separate training, per transcript
- assessment has its own data elements and subtype taxonomy
- current assessment rows are far more likely to be acronymic, publisher-missing, low-confidence, and fallback-selected
- current sample failures look like program-family resolution problems, not just ordinary title matching problems

Recommendation:

- treat `Assessment` as a first-class analysis slice immediately
- plan for a separate matcher treatment path unless later assessment training review shows the process is simpler than it currently appears

## Recommendations

### Benchmark slices

- Yes, change benchmark slices.
- At minimum add explicit slices for:
  - `Assessment`
  - sparse-evidence rows
  - placeholder rows
  - state-specific variant rows

### QA categories

- Yes, change QA categories.
- Add policy-aware challenge reasons, especially for usage ambiguity, sparse evidence, and assessment alias ambiguity.

### Assessment handling

- Make `Assessment` a first-class analysis slice now.
- Treat it as a likely separate matcher treatment path next, pending review of the dedicated assessment training materials.

That recommendation is stronger than for any other family because the current materials explicitly indicate a separate collection process, and the representative benchmark already shows a materially different error profile.
