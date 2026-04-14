# Instructional Materials Review Handoff

- Date: 2026-04-04
- Version: v006
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`
- Next-session focus: review worker training and instructional materials, extract collection-policy insights, and determine how those policies should inform benchmark design, matcher evaluation, and QA

## What changed this session

- Re-tuned QA triage to reduce reviewer workload while preserving strong precision.
- Added a strategy-specific repair gate for `title_plus_publisher`.
- Regenerated QA and repair benchmark outputs and refreshed roadmap/status docs.
- Audited historical-run sample representativeness against the July 2025 full population.
- Added a reusable historical sample analysis script and generated a new representative benchmark sample from the 532K-row historical population.
- Ran the matcher on the new representative 1,000-row benchmark against the July 2025 catalog.

## Current benchmark state

### Canonical QA benchmark

Artifacts:

- `benchmarks/qa/starter_gold_human_qa_canonical.csv`
- `benchmarks/outputs/starter-gold-human-qa-canonical-summary.json`
- `docs/analysis/2026-04-04-human-qa-tuning-report.md`

Current metrics:

- row count: `261`
- challenge rate: `0.2567`
- precision: `0.8209`
- recall: `0.3416`

Reviewer workflow recommendation:

- `high`: must-review queue
- `medium`: overflow review or sampled audit queue
- `low`: pass-through with periodic calibration sampling

Interpretation:

- the QA operating point is now more practical operationally than the earlier `0.5287` challenge-rate setting
- `high` priority precision is now `0.9310` on `29` rows
- QA is now functioning as a triage layer rather than an exhaustive catch-all

### Gold benchmark with repair gating

Artifacts:

- `benchmarks/outputs/starter-gold-fast-summary.json`
- `benchmarks/outputs/starter-gold-fast-records.csv`
- `docs/analysis/2026-04-04-repair-strategy-report.md`

Current metrics:

- record count: `100`
- top-1 accuracy: `0.47`
- top-3 recall: `0.58`
- prediction rate: `0.98`
- repair attempted rate: `0.51`
- fallback usage rate: `0.27`

Repair findings:

- repair rescued no-prediction rows: `24`
- repair improved top-1 correctness: `6`
- repair hurt top-1 correctness: `0`

Interpretation:

- the `title_plus_publisher` gate appears justified on this slice
- `publisher_as_title` remains low-trust and should not be loosened

### Representative historical benchmark

Source population:

- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Curriculum Matching Data - 07122025.csv`

Related artifacts created this session:

- `scripts/analyze_historical_run_samples.py`
- `docs/analysis/2026-04-04-historical-sample-representativeness-report.md`
- `benchmarks/gold/historical_07122025_representative_1000.csv`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_summary.json`
- `benchmarks/outputs/historical_07122025_representative_1000_fast_records.csv`

Representativeness findings:

- the two 750-row files under historical runs are the same `750` `selection_identifier` rows in two different schema views
- both 750-row files are matcher-input compatible
- the 750-row sample is not representative of the full `532,163`-row population
- the 10K historical results file is closer than the 750 sample on state mix, but still not representative because it omits `Assessment`
- the new 1,000-row representative sample preserves the population product-type mix much better, including `Assessment`

Representative 1,000-row benchmark metrics:

- record count: `1000`
- top-1 accuracy: `0.46`
- top-3 recall: `0.561`
- prediction rate: `0.932`

Most important slice findings:

- `Supplemental`: top-1 accuracy `0.625`
- `Core Curriculum`: top-1 accuracy `0.432`
- `Assessment`: top-1 accuracy `0.2962`

Interpretation:

- the broader historical slice confirms the current matcher is still far from the `95%` long-term goal
- `Assessment` is currently the clearest large failure family and likely deserves focused analysis

## Why the next session should review the instructional materials

The worker training materials likely encode policy and label logic that is not fully visible in the raw matcher inputs:

- what counts as a valid curriculum record
- what should be excluded
- how workers distinguish `Core Curriculum` vs `Supplemental`
- what sources are acceptable
- how district-created, platform, assessment, or tool-like products are handled
- how ambiguous or blocked cases are escalated
- how record status and review workflow operate in batches

This matters because some apparent matcher failures may actually be policy mismatch or collection-rule mismatch, not just retrieval/scoring errors.

## Training materials inventory already identified

Folder:

- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/03-research/training`

High-value text-first artifacts:

- `CEMD Core Curriculum Data Collection - Instructional Manual.md`
- `2024-2025 AY CEMD Supplemental Curriculum Data Collection - Instructional Manual.md`
- `FAQ - Core Curriculum Research & Data Collection.md`
- `FAQ - Supplemental Curriculum Research & Data Collection.md`
- `CEMD Data Elements 2025.md`
- `.srt` transcript files for the training videos

Important note:

- the markdown manuals and FAQs are already the best starting point
- use the `.srt` transcripts only when the written docs leave important ambiguity unresolved
- the `.docx` and `.mp4` files do not need to be the first pass

## Recommended next-session deliverables

1. A structured review of the training materials focused on collection policy, inclusion/exclusion logic, and edge-case handling.
2. A compact “policy layer” artifact in the repo, ideally something like:
   - `docs/policy/data-collection-rules.md`
   - optionally a machine-friendly companion such as `docs/policy/data-collection-rules.json`
3. A short analysis report mapping training-policy rules to matcher failure risk, benchmark design, and QA/repair implications.
4. Clear recommendations on whether benchmark slices, field normalization, or QA categories should change based on the instructional materials.

## Open questions for the next session

- What exact inclusion and exclusion rules are workers applying that the matcher currently does not model?
- Are `Assessment` records governed by materially different collection instructions than core or supplemental records?
- Do the training docs define common “do not collect” or “do not match” categories that should become explicit matcher filters or QA reasons?
- Are there source-priority or evidence-quality rules that should inform future QA or evidence auditing?
- Do the manuals define distinctions between district-created materials, platforms, programs, and curricula that explain current false positives?

## Important constraints for the next session

- Do not prioritize URL evidence or LLM integration yet unless the training-policy review directly changes the QA or benchmark strategy.
- Keep benchmark-driven evaluation central.
- Treat the representative 1,000-row historical benchmark as the best current population-shaped benchmark.
- Preserve the current QA and repair improvements unless the policy review reveals a clear contradiction.

## Validation already completed this session

- `./.venv/bin/python -m unittest discover -s tests`
- QA and repair benchmark reruns completed successfully
- representative historical benchmark run completed successfully
- historical sample compatibility and representativeness analysis completed successfully

## Agent-Optimized Kickoff Prompt

Use this kickoff prompt for the next AI coding agent session:

Resume work on the curriculum matcher from this handoff:

`/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/docs/handoffs/2026-04-04-v006-instructional-materials-review-handoff.md`

Current priority is to review the worker instructional and training materials and extract collection-policy insights that should influence benchmark design, matcher evaluation, and QA.

Please:
1. review the handoff and the current benchmark/report artifacts,
2. review the training materials in `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/03-research/training`,
3. start with the markdown manuals and FAQs first, then use `.srt` transcripts only if the written docs leave important ambiguity unresolved,
4. extract the operational collection rules into a concise structured policy artifact,
5. produce a short analysis report describing how those collection rules should affect:
   - benchmark design,
   - matcher error interpretation,
   - QA triage or challenge reasons,
   - and any likely failure-family segmentation,
6. call out any policy distinctions that help explain why `Assessment` is currently much weaker than `Supplemental` or `Core Curriculum`.

Important context:
- The current representative historical benchmark is:
  - `benchmarks/gold/historical_07122025_representative_1000.csv`
  - `benchmarks/outputs/historical_07122025_representative_1000_fast_summary.json`
  - `benchmarks/outputs/historical_07122025_representative_1000_fast_records.csv`
- Current representative benchmark metrics are approximately:
  - top-1 accuracy `0.46`
  - top-3 recall `0.561`
  - prediction rate `0.932`
- The strongest current failure family appears to be:
  - `Assessment` top-1 accuracy `0.2962`
- Historical sample representativeness analysis is here:
  - `docs/analysis/2026-04-04-historical-sample-representativeness-report.md`
- Current QA and repair reports are here:
  - `docs/analysis/2026-04-04-human-qa-tuning-report.md`
  - `docs/analysis/2026-04-04-repair-strategy-report.md`

Constraints:
- Do not start by changing matcher logic.
- Prioritize policy extraction and interpretation first.
- Keep the review text-first and artifact-driven.
- Avoid spending time on the videos unless the markdown manuals or FAQs are insufficient.
- Focus on extracting rules and implications, not rewriting the training materials wholesale.

Expected outputs for the next session:
- a structured policy/rules artifact derived from the training materials
- a short analysis report linking collection policy to benchmark and matcher behavior
- a recommendation on whether benchmark slices or QA categories should change
- a recommendation on whether `Assessment` should become a first-class analysis slice or a separate matcher treatment path
