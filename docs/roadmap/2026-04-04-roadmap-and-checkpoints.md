# Curriculum Matcher Roadmap And Checkpoints

This document is the working roadmap for the current matcher program. It is meant to keep milestone status, current evidence, and next checkpoints in one place so we can track what is completed, what is measured, and what remains open.

## Status Key

- `not started`
- `in progress`
- `measured`
- `accepted`
- `deferred`

## Milestone Summary

| Milestone | Goal | Current Status | Primary Evidence | Next Checkpoint |
| --- | --- | --- | --- | --- |
| Milestone 0 | Lock the baseline | `accepted` | Gold benchmark, QA benchmark, URL audit, error-analysis docs | Use these artifacts as the reference point for all future reviews |
| Milestone 1 | Make repair-aware matching reliable | `in progress` | Repair-aware benchmark summary and repair strategy report | Validate whether the `title_plus_publisher` gate should remain the default repair policy |
| Milestone 2 | Make human QA operational | `measured` | Canonical QA benchmark and QA summary | Validate the reduced reviewer queue against actual reviewer capacity |
| Milestone 3 | Make source URL evidence useful | `in progress` | URL audit summary and first live URL pilot | Decide whether stronger extraction or tighter URL targeting is worth pursuing |
| Milestone 4 | Add bounded LLM review | `not started` | LLM seam exists only | Start only after Milestones 2 and 3 produce stable signals |

## Milestone 0: Baseline Locked

Status: `accepted`

Completed artifacts:

- `benchmarks/outputs/starter-gold-fast-summary.json`
- `benchmarks/outputs/starter-gold-human-qa-summary.json`
- `benchmarks/outputs/starter-gold-url-audit-summary.json`
- `docs/analysis/2026-04-03-starter-gold-fast-error-analysis.md`
- `docs/analysis/2026-04-03-starter-gold-field-corruption-report.md`

Locked baseline metrics:

- Gold benchmark:
  - top-1 accuracy `0.45`
  - top-3 recall `0.57`
  - prediction rate `0.98`
- Repair usage:
  - repair attempted rate `0.51`
  - fallback usage rate `0.37`
- Human QA baseline:
  - challenge precision `0.4318`
  - challenge recall `0.2969`
- URL evidence baseline:
  - `adopted_curriculum_source_url` coverage `0.66`
  - `source_document_link` coverage `0.00`

Checkpoint rule:

- All future milestone reviews should compare against these values first.

## Milestone 1: Repair-Aware Matching

Status: `in progress`

Completed:

- repair-aware fallback matching is implemented
- evaluation outputs include repair metrics and selected strategy
- corruption benchmark generation and evaluation exist
- repair strategy report exists

Current checkpoint questions:

- Which fallback strategy is helping most often by product family?
- Where does `title_plus_publisher` overfit?
- Are fallback-selected matches less trustworthy than primary-selected matches?
- Should fallback selection remain score-based or become heuristic-gated?

Current measured state:

- gold benchmark:
  - top-1 accuracy `0.47`
  - top-3 recall `0.58`
- repair usage:
  - repair attempted rate `0.51`
  - fallback usage rate `0.27`
- repair outcomes:
  - repair rescued no-prediction rows `24`
  - repair improved top-1 correctness `6`
  - repair hurt top-1 correctness `0`

Working recommendation:

- keep repair selection score-based in general
- keep a heuristic gate specifically on `title_plus_publisher` so it needs a larger margin before replacing an existing primary match

Exit criteria:

- one report that clearly splits performance by `match_selected_strategy`
- one recommendation for whether repair selection should stay score-based

## Milestone 2: Human QA Scoring

Status: `measured`

Completed:

- synthetic QA benchmark builder exists
- richer canonical silver QA benchmark now exists
- QA evaluator exists
- QA output contract exists with support score, challenge flag, reasons, priority, and competing AI match

Current measured state:

- synthetic QA benchmark:
  - precision `0.4318`
  - recall `0.2969`
- canonical QA benchmark:
  - precision `0.8209`
  - recall `0.3416`
  - challenge rate `0.2567`
  - priority precision:
    - `high` `0.9310`
    - `medium` `0.7368`
    - `low` `0.0000`

Working reviewer workflow:

- `high`: must-review queue
- `medium`: overflow review or sampled audit queue
- `low`: pass-through with periodic calibration sampling

Still to complete:

- validate that the reduced challenge rate is operationally acceptable for human reviewers
- confirm whether medium-priority rows should be sampled at a fixed rate or reviewed only during spare capacity

Exit criteria:

- one canonical QA benchmark file
- one summary report with precision, recall, challenge rate, and reason breakdown
- one reviewer workflow recommendation with `high`, `medium`, and `low` queue meanings

## Milestone 3: Source URL Evidence

Status: `in progress`

Completed:

- source evidence audit CLI exists
- current gold-set audit summary exists
- URL classification and lightweight extraction are implemented

Current evidence:

- `adopted_curriculum_source_url` is populated on `66%` of the gold set
- `source_document_link` is empty in the current gold sample
- bounded live URL pilot completed on `20` rows
- live fetch success rate in the bounded pilot was `0.25`
- current lightweight extraction produced `0.00` expected product / series / publisher mentions in that pilot

Still to complete:

- stronger extraction on useful link types or tighter sampling toward higher-signal URLs
- fetch success analysis by link type beyond the first bounded pilot
- evidence usefulness analysis on hard rows
- decision on whether URL evidence should influence matching, QA, or only audits

Exit criteria:

- one live URL pilot report
- one explicit usage decision for URL evidence

## Milestone 4: Bounded LLM Review

Status: `not started`

Completed:

- LLM reviewer seam exists

Blocked by:

- stronger QA signal from Milestone 2
- better evidence utility signal from Milestone 3

Exit criteria:

- trigger policy
- structured prompt/response contract
- reviewed-case benchmark with quality and latency summary

## Immediate Next Work

1. Repair Evaluation Deepening
   - status: `in progress`
   - deliverable: confirm the `title_plus_publisher` gate on the next review slice and decide whether any other repair strategies need similar controls
2. Better QA Benchmark
   - status: `measured`
   - deliverable: reviewer-capacity validation against the current canonical operating point
3. Live URL Evidence Pilot
   - status: `in progress`
   - deliverable: bounded fetch report plus decision on whether to improve extraction or narrow URL targeting
4. LLM Trigger Design
   - status: `deferred`
   - deliverable: trigger policy after Milestones 2 and 3 stabilize

## Weekly Checkpoint Template

At each weekly review, record:

- benchmark delta vs Milestone 0 baseline
- rows improved
- rows degraded
- top unresolved failure families
- QA precision/recall trend
- URL evidence utility trend
- new failure modes introduced

Every weekly checkpoint should end with:

- one short status summary
- one benchmark artifact snapshot
- one go / no-go decision for the next milestone step
- one unresolved-risk list
