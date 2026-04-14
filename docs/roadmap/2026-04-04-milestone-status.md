# Milestone Status Snapshot

Date: 2026-04-04

## Overall Program Status

- Milestone 0: `accepted`
- Milestone 1: `in progress`
- Milestone 2: `in progress`
- Milestone 3: `in progress`
- Milestone 4: `not started`

## Completed

- benchmark and evaluation scaffolding
- gold and corruption benchmark generation
- repair-aware fallback matching
- repair-aware benchmark outputs
- synthetic human-QA benchmark and first QA scorer
- URL/source evidence audit tooling
- LLM integration seam without live calls

## Measured

- repair-aware matching benchmark:
  - top-1 accuracy `0.47`
  - top-3 recall `0.58`
  - prediction rate `0.98`
  - repair attempted rate `0.51`
  - fallback usage rate `0.27`
  - repair improved top-1 correctness `6`
  - repair hurt top-1 correctness `0`
- synthetic QA benchmark:
  - challenge precision `0.4318`
  - challenge recall `0.2969`
  - challenge rate `0.2683`
- canonical QA benchmark:
  - challenge precision `0.8209`
  - challenge recall `0.3416`
  - challenge rate `0.2567`
  - review priority precision:
    - `high` `0.9310`
    - `medium` `0.7368`
    - `low` `0.0000`
- source evidence audit:
  - `adopted_curriculum_source_url` coverage `0.66`
  - `source_document_link` coverage `0.00`
- live URL pilot:
  - sampled rows `20`
  - fetch success rate `0.25`
  - expected product / series / publisher mentions `0.00`

## In Progress

- Milestone 1:
  - validate the new `title_plus_publisher` gate on additional review slices
  - decide whether `publisher_as_title` needs a stronger control or separate de-prioritization
- Milestone 2:
  - reviewer workflow is now defined around `high` must-review and `medium` overflow audit
  - remaining work is reviewer-capacity validation rather than additional broad recall tuning
- Milestone 3:
  - move from bounded live pilot to better extraction or more targeted URL selection

## To Be Completed

- validate the first reviewer triage policy against actual reviewer capacity
- complete a stronger live URL evidence pilot with better extraction on useful link types
- decide whether URL evidence belongs in matching, QA, both, or audit-only
- define LLM trigger policy and structured reviewer outputs after Milestones 2 and 3 stabilize

## Current Risks

- `publisher_as_title` remains a low-trust fallback despite the `title_plus_publisher` gate improvement
- canonical QA metrics are improved, but medium-priority handling still needs a concrete reviewer-capacity policy
- current live URL pilot suggests many links are low-signal district pages, broken references, or inaccessible documents
- LLM review remains blocked until the non-LLM signals are better grounded
