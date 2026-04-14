# Data Collection Rules

Derived from:

- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/03-research/training/CEMD Core Curriculum Data Collection - Instructional Manual.md`
- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/03-research/training/2024-2025 AY CEMD Supplemental Curriculum Data Collection - Instructional Manual.md`
- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/03-research/training/FAQ - Core Curriculum Research & Data Collection.md`
- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/03-research/training/FAQ - Supplemental Curriculum Research & Data Collection.md`
- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/03-research/training/CEMD Data Elements 2025.md`
- Targeted transcript clarification from:
  - `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/03-research/training/Video 2.1_ The Type of Curriculum Information to Look For.srt`
  - `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/03-research/training/Video 2.2_ How to Research & Navigate School District Websites - SUPP.srt`

## Scope

- These rules describe how human workers are instructed to collect district curriculum selections.
- They are collection-policy rules, not matcher logic rules.
- They imply what the historical labels mean and where those labels are likely sparse, noisy, or policy-shaped.

## Core Principles

- District website evidence is the primary source of truth.
- Selection type records district usage, not intrinsic product type.
- Product title should be captured exactly as listed by the district.
- Researchers should complete a good-faith district search before concluding that no information is available.
- Missing metadata is often acceptable; workers are instructed to capture what is available rather than require every field.

## Evidence And Source Rules

- Count only information tied to the district website.
- Do not count school board documents unless they are linked from the district site.
- Static files are allowed as evidence only when reached from a district webpage.
- Record the live higher-level district webpage as the main curriculum URL, not the PDF or document itself.
- For supplemental, a direct document link can additionally be stored in `Source Document Link`.
- If an old referential URL points straight to a static file or a broken page, workers are told to re-find the district page that links to it.
- If district materials are outdated, unlabeled, or inconsistently dated, workers are told to look for the most recent posted information and assume posted materials reflect current use if nothing newer is available.

## Search And Effort Rules

- Spend at least 10-15 minutes per district on a thorough search.
- Search district menus and internal search first, then Google if needed.
- Expected alternate evidence sources include curriculum pages, instructional materials pages, curriculum maps, frameworks, pacing guides, teacher resources, student resources, and parent/family resources.
- Workers are told to finish one district at a time unless a case is put on hold.

## Inclusion Rules

- Capture every product the district indicates for the applicable subject and grade context.
- Add as many rows as needed when multiple products are listed for the same district, subject, and grade band.
- Core team may also capture supplemental products encountered in core documentation by duplicating the row and changing `Selection Type` to `Supplemental`.
- For core-only FAQ handling, if only the publisher is listed, workers may use a publisher-plus-subject unspecified catalog entry.
- District-created or teacher-created curriculum should be recorded as district-created curriculum when the catalog supports it.
- State-specific versions should be selected when the district indicates one, and FL state-specific variants should be assumed where a FL-specific catalog version exists.
- Course type should be recorded only when explicitly indicated.
- `General` must not be defaulted just because nothing more specific is listed.

## Exclusion Rules

- Do not include information that cannot be tied back to the district website.
- Do not include information when subject area cannot be determined.
- Do not collect novels or book lists for ELA even if they appear in curriculum documents.
- Do not collect excluded course families listed in the manuals, such as IB and various non-target electives.

## Selection Type Rules

- `Selection Type` reflects how the district says it uses the product.
- The same product title may be `Core Curriculum` in one district and `Supplemental` in another.
- Workers are explicitly told not to infer selection type from the product’s general market identity.
- Typical district language:
  - core: main instruction, primary text, primary resource
  - supplemental: resource, supplement, Tier II, Tier III, intervention, remedial
- Transcript clarification: workers are told that selection type describes district usage and should remain supplemental even when the product itself may be recognized as core-like elsewhere.

## Field Completeness Rules

- Complete as many fields as possible, but blank fields are allowed when the district does not provide the information.
- `Product name from district` is required and should be verbatim raw text.
- Publisher and copyright are entered when available.
- Copyright may be deduced from surrounding adoption information when that clearly narrows the catalog choice.
- `Other` should be used only as a last resort.
- Grade band and grades used may be inferred from district structure only in limited ways described by the manuals.
- If grade coverage is still unclear, workers are told to ask, comment, or place the record on hold.

## Record Status And Escalation Rules

- `Validated / In Use` means current district usage was confirmed.
- `Curriculum / product has changed` is used when a prior validated product is no longer current.
- `No curriculum information available` is used only after a thorough search.
- On blocked cases, workers are told to use comments and set status to `On Hold`.
- QA is expected to cross-check `No curriculum information available` rows.

## Policy Differences By Collection Type

### Core

- Broader subject scope: ELA, Math, Science, Social Studies.
- Core documentation often carries more explicit grade and course structure.
- Workers validate existing rows and also add newly found products.
- Core workflow explicitly allows capture of supplemental products found in the same source.

### Supplemental

- Narrower subject scope: ELA and Math only.
- Documentation is expected to be less complete and often found in different website locations than core.
- Workers are told that grade, course type, and adoption metadata are less likely to be available.
- Supplemental products may include practice, intervention, facilitation, adaptive tools, and resource banks.

### Assessment

- The markdown manuals reviewed here do not define the assessment workflow itself.
- `CEMD Data Elements 2025.md` shows assessment-specific fields and subtags:
  - benchmark
  - college readiness
  - diagnostic
  - formative
  - progress monitoring / interim
  - social emotional learning
  - universal screeners
- Transcript clarification says assessment collection is a separate process with separate training materials.
- This means assessment labels likely follow policy distinctions not present in the core and supplemental manuals.

## Implications For Historical Labels

- A row may be correct even when publisher, copyright, or exact variant details are missing.
- Short raw titles can still be valid labels if that is all the district reports.
- Some labels encode district usage choices rather than catalog taxonomy.
- `No curriculum information available` is a policy outcome after bounded search effort, not proof that no product exists.
- Assessment labels should be treated as partially out-of-context if only core and supplemental manuals are consulted, because the training explicitly says assessment uses a separate process.
