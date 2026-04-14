# Benchmark Slicing And QA Recommendations

## Purpose

Translate the reviewed collection-policy artifact, UI screen-flow evidence, and representative historical benchmark outputs into concrete evaluation recommendations without changing matcher logic.

Primary source artifacts:

- `docs/handoffs/2026-04-04-v007-ui-screen-flow-review-handoff.md`
- `docs/policy/data-collection-rules.md`
- `docs/analysis/2026-04-04-collection-policy-implications-report.md`
- `benchmarks/gold/historical_07122025_representative_1000.csv`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_records.csv`

Reference catalog used by the benchmark output summary:

- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv`

## Starting Point

Current representative benchmark metrics:

- top-1 accuracy: `0.46`
- top-3 recall: `0.561`
- prediction rate: `0.932`

Current usage-family metrics:

- `Supplemental`: top-1 `0.625`
- `Core Curriculum`: top-1 `0.432`
- `Assessment`: top-1 `0.2962`

Policy and UI evidence together imply:

- historical labels encode district-reported usage, not only product identity
- sparse metadata is policy-valid, so many low-information rows are not annotation defects
- workers enter district-facing evidence separately from the normalized catalog selection
- catalog search likely introduces duplicate-title, publisher, copyright, state-version, and `Unspecified` ambiguity
- assessment follows a distinct workflow and likely a distinct ontology

## Benchmark Slicing Proposal

Recommendation: keep one representative benchmark framework, but add required reporting slices that distinguish policy-hard cases from ordinary ranking failures.

### 1. Required primary slices

These should appear in every benchmark report.

#### `usage_family`

Definition:

- `Core Curriculum`
- `Supplemental`
- `Assessment`

Why:

- this is a first-class collection field
- labels encode district usage, not intrinsic market category
- current performance is materially different across these families

#### `evidence_richness`

Definition:

- `sparse`
  - raw title only or very short title
  - publisher absent
  - no visible year or variant clue
- `medium`
  - title plus one useful disambiguator
  - examples: publisher present, year in raw text, or meaningful variant wording
- `rich`
  - title plus publisher plus at least one additional disambiguator
  - examples: year, explicit variant wording, or long descriptive title
- `policy_placeholder_no_info`
  - explicit no-information or placeholder titles such as `Not Available`

Implementation note:

- this should be computed from benchmark row fields only, not matcher output
- start with deterministic heuristics over `product_name_raw`, `publisher_raw`, and obvious year/variant tokens
- allow later replacement with a cleaner feature extractor if needed

Observed benchmark signal from a first heuristic pass:

- `sparse`: `406` rows, top-1 `0.4631`
- `medium`: `452` rows, top-1 `0.4270`
- `rich`: `141` rows, top-1 `0.5603`
- `policy_placeholder_no_info`: `1` row, top-1 `0.0`

Interpretation:

- richer evidence clearly behaves differently and should not be blended with sparse policy-valid rows

#### `usage_ambiguity`

Definition:

- `explicit_usage`
  - raw district wording explicitly signals use context
  - examples: intervention, Tier II, benchmark, main curriculum, supplemental resource
- `implicit_usage`
  - row has a product label but no direct usage cue in the captured evidence
- `assessment_naming_regime`
  - assessment rows tracked separately because their ambiguity is often subtype- or acronym-based rather than core-vs-supplemental wording

Why:

- policy makes `Selection Type` meaningful
- UI confirms workers explicitly enter usage separately from product selection
- “right title, wrong usage” should remain a real error, not a near-miss

Implementation note:

- do not try to infer district usage from catalog `product_type`
- derive this slice from district-facing row evidence and row family

#### `state_specific_risk`

Definition:

- `catalog_state_specific_expected`
  - expected catalog row has `state_specific_version = true`
- `adoption_state_high_risk`
  - state in `CA`, `FL`, or `TX`
- `standard_state`
  - all others

Why:

- policy explicitly treats state-specific versions as meaningful
- UI shows both a state-version input and state-specific catalog rows
- catalog search ambiguity is visibly higher in adoption-state contexts

Observed benchmark signal:

- `state_specific_catalog`: `53` rows, top-1 `0.4151`, top-3 `0.5094`
- all `CA/FL/TX`: `402` rows, top-1 `0.4428`
- `Assessment` within `CA/FL/TX`: `87` rows, top-1 `0.1724`

Interpretation:

- adoption-state risk is especially important when combined with assessment naming ambiguity

#### `placeholder_mapping`

Definition:

- `catalog_unspecified`
  - expected catalog name or series contains `Unspecified`
- `district_created`
  - expected catalog row has `district_created = true`
- `no_information_available`
  - policy placeholder rows such as `Assessment Information Not Publicly Available`
- `standard_catalog_mapping`
  - none of the above

Why:

- policy allows fallback and placeholder selections in bounded-search conditions
- UI search results visibly include `Unspecified`
- this is exactly where catalog-selection ambiguity and label noise can accumulate

Observed benchmark signal:

- `unspecified`: `56` rows, top-1 `0.0893`, top-3 `0.2321`

Interpretation:

- `Unspecified` rows are the strongest single benchmark warning sign in the current representative sample
- they should be a mandatory headline slice, not buried inside the overall average

#### `assessment_specific`

Definition:

- `assessment_all`
- `assessment_short_or_acronym_title`
- `assessment_publisher_missing`
- `assessment_state_specific_expected`
- `assessment_non_assessment_catalog_edge`
  - optional review bucket for odd rows where expected catalog metadata suggests embedded or mixed-use products

Observed benchmark signal:

- `assessment`: `260` rows, top-1 `0.2962`, top-3 `0.3577`
- `assessment_short_title`: `174` rows, top-1 `0.2816`
- `assessment_publisher_missing`: `218` rows, top-1 `0.2844`

Interpretation:

- assessment should not be reported as a single opaque family
- acronymic and publisher-missing assessment rows deserve their own mandatory cut

### 2. Cross-slice matrix to report

For each benchmark run, report at least these intersections:

- `usage_family x evidence_richness`
- `usage_family x placeholder_mapping`
- `usage_family x state_specific_risk`
- `Assessment x assessment_specific`

Recommended minimum QA review sample per run:

- 20 incorrect rows from `catalog_unspecified`
- 20 incorrect rows from `Assessment`
- 10 incorrect rows from `Assessment x adoption_state_high_risk`
- 10 no-prediction rows from `policy_placeholder_no_info` or explicit no-information labels

### 3. What not to do

- do not collapse placeholder rows into ordinary “hard title match” error analysis
- do not treat sparse rows as defective labels by default
- do not use the screenshot document as the current system spec; use it only as supporting evidence for ambiguity modes already consistent with policy artifacts

## Revised QA Challenge-Reason Taxonomy

Recommendation: split challenge reasons into one primary reason plus optional secondary tags.

### Primary reasons

#### `likely_matcher_error`

Use when:

- evidence is reasonably rich
- expected row is not an obvious placeholder
- candidate confusion is ordinary ranking or normalization failure

Typical cues:

- title and publisher are present
- variant is explicit
- no strong sign of UI duplicate ambiguity

#### `likely_evidence_sparsity`

Use when:

- district-facing evidence is thin but policy-valid
- row is title-only, acronym-only, publisher-missing, or otherwise weakly specified
- the miss is plausibly driven by bounded district evidence rather than a clearly wrong ranking choice

Typical secondary tags:

- `title_only`
- `publisher_missing_allowed`
- `year_missing_allowed`
- `weak_district_context`

#### `likely_ui_catalog_selection_ambiguity`

Use when:

- the worker-entered raw evidence may be fine, but the normalized catalog choice is inherently ambiguous in the UI
- duplicate titles, near-duplicate publishers, copyright-year variants, state-specific variants, or `Unspecified` rows are plausible confusion drivers

Typical secondary tags:

- `duplicate_title`
- `publisher_rebrand_or_alias`
- `copyright_year_collision`
- `state_specific_variant_collision`
- `unspecified_row_present`

#### `likely_assessment_alias_or_subtype_ambiguity`

Use when:

- the row is assessment-related
- the ambiguity looks like family-vs-subtest, alias-vs-official-name, acronym expansion, or benchmark/interim/formative subtype confusion

Typical secondary tags:

- `acronym_only`
- `family_vs_subassessment`
- `state_test_alias`
- `benchmark_vs_interim`
- `publisher_missing_assessment`

### Secondary tags

These should be add-on tags, not primary reasons:

- `usage_type_mismatch`
- `no_prediction`
- `fallback_strategy_used`
- `expected_unspecified`
- `district_created_expected`
- `state_specific_expected`
- `assessment_workflow_gap`
- `reviewer_follow_up_needed`

### Triage guidance

Default QA routing:

- `likely_matcher_error` -> matcher error queue
- `likely_evidence_sparsity` -> evidence-limited queue
- `likely_ui_catalog_selection_ambiguity` -> catalog QA / labeling ambiguity queue
- `likely_assessment_alias_or_subtype_ambiguity` -> assessment review queue

This is better than a confidence-only taxonomy because it distinguishes true model misses from policy-shaped ambiguity.

## Recommendation On `Assessment`

Recommendation:

- keep `Assessment` as a first-class slice inside the single benchmark framework now
- move toward a separate treatment path for assessment evaluation and likely later matcher handling

Why not fully split the benchmark immediately:

- it is still valuable to preserve one representative benchmark and one topline scoreboard
- product stakeholders likely need to compare families in a single view
- the current benchmark already contains assessment rows, so removing them now would hide a real deployment problem

Why a separate treatment path is still warranted:

- assessment has separate training, per reviewed materials
- assessment has distinct subtags in `CEMD Data Elements 2025.md`
- current benchmark behavior is materially different, not just slightly worse
- assessment naming patterns are often acronymic, subtype-based, and state-program-like rather than catalog-title-like
- `Assessment` inside `CA/FL/TX` is especially weak

Practical next step:

- retain one benchmark framework for reporting
- add an assessment-specific evaluation appendix and QA queue now
- defer matcher-logic separation until after reviewing the minimum assessment training materials below

## Minimum Additional Assessment Review Needed

Minimum next review set:

1. `CEMD Data Elements 2025.md`
   - focus on all assessment-specific fields and subtags
   - this is the clearest artifact already known to encode the assessment ontology
2. `Video 2.1_ The Type of Curriculum Information to Look For.srt`
   - review the assessment-related transcript segment again to capture the exact “separate process” wording
3. Any dedicated assessment manual, FAQ, or transcript not yet reviewed
   - current training folder listing does not show an obvious assessment instructional manual alongside the core and supplemental manuals
   - if one exists elsewhere, this is the single highest-value missing artifact

If no dedicated assessment manual exists in the accessible materials, the minimum acceptable follow-up is:

- document that assessment policy is only partially observed
- keep `Assessment` in the benchmark as a flagged special family
- avoid committing to assessment-specific matcher logic until dedicated training is found or confirmed absent

## Recommended Next Session Deliverables

- formalize the slice definitions in benchmark-reporting code or benchmark post-processing only
- backfill the representative benchmark with the new slice columns
- apply the revised QA reason taxonomy to a targeted error sample
- review the minimum assessment-specific training artifacts before any assessment matcher changes

## Bottom Line

The immediate evaluation change should be to keep one benchmark but stop treating it as homogeneous.

Most important additions:

- mandatory `usage_family` reporting
- mandatory `evidence_richness` reporting
- mandatory `placeholder_mapping`, especially `Unspecified`
- mandatory assessment sub-slices
- QA reasons that separate matcher failure from sparse evidence, UI/catalog ambiguity, and assessment alias ambiguity
