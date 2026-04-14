# Roadmap, QA, And URL Pilot Handoff

- Date: 2026-04-04
- Version: v005
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`
- Session focus: move the project from baseline/error analysis into milestone tracking, canonical QA benchmarking, QA threshold tuning, and a bounded live URL evidence pilot

## What changed this session

- Added roadmap and checkpoint tracking documents under `docs/roadmap/`.
- Added a repair-strategy report to support Milestone 1 review.
- Upgraded the human QA benchmark from a smaller synthetic-only set to a richer canonical silver benchmark with case metadata.
- Tuned the QA challenge logic against the canonical benchmark to improve recall while keeping precision strong.
- Added and ran a bounded live URL pilot against `adopted_curriculum_source_url`.
- Confirmed that the current project still has no live LLM reviewer calls wired into execution; the LLM seam exists but is not active.

## Important code and workflow additions

### Roadmap / milestone organization

New docs:

- `docs/roadmap/2026-04-04-roadmap-and-checkpoints.md`
- `docs/roadmap/2026-04-04-milestone-status.md`
- `docs/roadmap/weekly-checkpoint-template.md`

Purpose:

- capture milestone status in one place
- define weekly checkpoint rhythm
- make “completed / measured / in progress / deferred” explicit

### Repair analysis support

New script:

- `scripts/report_repair_strategy_metrics.py`

Generated report:

- `docs/analysis/2026-04-04-repair-strategy-report.md`

Current repair findings from the gold benchmark:

- repair attempted rate: `0.51`
- fallback usage rate: `0.37`
- repaired rows: `37`
- repair rescued no-prediction rows: `24`
- repair improved top-1 correctness: `7`
- repair hurt top-1 correctness: `3`

Interpretation:

- fallback repair is clearly increasing useful coverage
- `title_plus_publisher` is doing most of the rescuing
- `title_plus_publisher` is also responsible for observed “hurt” rows, so it remains the main candidate for future heuristic gating

### Canonical human QA benchmark

Updated builder:

- `scripts/build_human_qa_benchmark.py`

New canonical benchmark artifacts:

- `benchmarks/qa/starter_gold_human_qa_canonical.csv`
- `benchmarks/qa/starter_gold_human_qa_canonical_summary.json`

Canonical benchmark composition:

- row count: `261`
- correct human matches: `100`
- incorrect human matches: `161`
- label strengths:
  - gold positive: `100`
  - silver strong negative: `53`
  - silver negative: `108`
- benchmark slices:
  - `fallback_selected`: `105`
  - `has_adopted_url`: `156`

Interpretation:

- this is now the better working benchmark for Milestone 2
- it is materially stronger than the earlier synthetic-only QA set

### QA challenge tuning

Relevant code:

- `src/curriculum_matcher/qa.py`
- `tests/test_qa.py`
- `scripts/report_human_qa_tuning.py`

Key QA logic changes:

- support-score challenge threshold moved to `0.45`
- score-gap challenge margin moved to `0.05`
- low-support rows can now be challenged even without an alternate AI top candidate
- high-priority routing now uses:
  - support score `< 0.40`, or
  - score gap `>= 0.12`

Updated canonical QA output:

- `benchmarks/outputs/starter-gold-human-qa-canonical-summary.json`
- `docs/analysis/2026-04-04-human-qa-canonical-report.md`
- `docs/analysis/2026-04-04-human-qa-tuning-report.md`

Current tuned QA metrics:

- row count: `261`
- challenge rate: `0.5287`
- precision: `0.7391`
- recall: `0.6335`
- incorrect row rate: `0.6169`

Priority bucket interpretation from the tuning report:

- `high` priority precision: `0.7763`
- `medium` priority precision: `0.6935`
- `low` priority precision: `0.0000`

Current reviewer workflow recommendation:

- `high`: must-review queue
- `medium`: secondary review queue when reviewer capacity allows
- `low`: informational only unless another signal conflicts

Important caution:

- the new QA operating point is much better than before, but challenge rate is now `0.5287`, which may or may not be operationally acceptable for human reviewers
- the next session should evaluate reviewer-capacity fit, not just precision/recall

### Live URL evidence pilot

New script:

- `scripts/run_live_url_pilot.py`

Related helper hardening:

- `src/curriculum_matcher/url_evidence.py`

Generated pilot artifacts:

- `benchmarks/outputs/starter-gold-live-url-pilot.csv`
- `benchmarks/outputs/starter-gold-live-url-pilot-summary.json`
- `docs/analysis/2026-04-04-live-url-pilot-report.md`

Pilot configuration:

- bounded sample size: `20` rows
- attempted balanced sample across URL types:
  - district or org page
  - invalid
  - pdf
  - web page

Live pilot results:

- live fetch success rate: `0.25`
- fetch status counts:
  - http_error: `8`
  - success: `5`
  - invalid_url: `5`
  - network_error: `2`
- expected product mentions: `0.0`
- expected series mentions: `0.0`
- expected publisher mentions: `0.0`

Interpretation:

- the current URL field mix appears noisy and often low-signal
- successful fetches were mostly district or Google Drive style pages, not clearly product-specific vendor pages
- the current lightweight extraction is not yet surfacing evidence useful for matching or QA
- URL evidence should not be integrated into scoring yet

## Current benchmark state

### Gold benchmark with repair-aware matching

Artifacts:

- `benchmarks/outputs/starter-gold-fast-summary.json`
- `benchmarks/outputs/starter-gold-fast-records.csv`

Current metrics:

- record count: `100`
- top-1 accuracy: `0.45`
- top-3 recall: `0.57`
- prediction rate: `0.98`
- mean top-1 score: `0.6104`
- mean correct top-1 score: `0.663`

Confidence bands:

- high: `13` rows, top-1 accuracy `0.7692`
- medium: `72` rows, top-1 accuracy `0.4444`
- low: `11` rows, top-1 accuracy `0.1818`
- no prediction: `2` rows
- very low: `2` rows

Repair metrics:

- repair attempted rate: `0.51`
- fallback usage rate: `0.37`
- selected strategy counts:
  - primary: `63`
  - title_plus_publisher: `29`
  - publisher_as_title: `8`

## Tests / validation run

Verified repeatedly during this session:

- `./.venv/bin/python -m unittest discover -s tests`
- `./.venv/bin/python -m py_compile ...` for the new scripts and QA module

Result at end of session:

- `25` tests passed

## Most important current findings

1. Repair-aware matching remains valuable and is rescuing many rows, but the fallback layer still needs trust calibration because some repairs hurt top-1 correctness.
2. The canonical QA benchmark is now good enough to use as the Milestone 2 working benchmark.
3. QA tuning materially improved the operational value of the QA layer:
   - precision stayed strong
   - recall improved substantially
   - reviewer priority buckets now have a meaningful interpretation
4. The current challenge rate may still be too high for production reviewer capacity, so the next step is workload validation rather than another blind threshold sweep.
5. The first live URL pilot suggests the current URL field mix is too noisy and low-signal to justify immediate scoring integration.
6. The codebase still does not make live LLM calls; everything currently running is retrieval/rules/Python only.

## Recommended next step

Do not move to LLM integration yet.

The best next session should focus on Milestone 2 reviewer-capacity and workflow validation, plus a Milestone 1 decision on repair gating.

Recommended sequence:

1. Validate reviewer-capacity fit for the tuned QA operating point:
   - decide whether challenge rate `0.5287` is operationally acceptable
   - if not, test stricter triage policies that preserve `high` bucket precision
2. Freeze the first reviewer workflow policy:
   - `high` = must-review
   - `medium` = capacity-permitting review
   - `low` = informational only
3. Revisit repair gating:
   - specifically evaluate whether `title_plus_publisher` should be heuristic-gated because it both rescues and harms rows
4. Keep URL evidence exploratory for now:
   - only continue if the next pass improves extraction or narrows the sample to likely high-signal URLs

## Useful files for the next session

Roadmap / status:

- `docs/roadmap/2026-04-04-roadmap-and-checkpoints.md`
- `docs/roadmap/2026-04-04-milestone-status.md`
- `docs/roadmap/weekly-checkpoint-template.md`

Milestone 1 / matching:

- `docs/analysis/2026-04-04-repair-strategy-report.md`
- `benchmarks/outputs/starter-gold-fast-summary.json`
- `benchmarks/outputs/starter-gold-fast-records.csv`

Milestone 2 / QA:

- `benchmarks/qa/starter_gold_human_qa_canonical.csv`
- `benchmarks/qa/starter_gold_human_qa_canonical_summary.json`
- `benchmarks/outputs/starter-gold-human-qa-canonical-summary.json`
- `docs/analysis/2026-04-04-human-qa-canonical-report.md`
- `docs/analysis/2026-04-04-human-qa-tuning-report.md`

Milestone 3 / URL evidence:

- `benchmarks/outputs/starter-gold-url-audit-summary.json`
- `benchmarks/outputs/starter-gold-live-url-pilot-summary.json`
- `docs/analysis/2026-04-04-live-url-pilot-report.md`
