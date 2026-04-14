# Overnight Autonomous Continuation Workflow

- Date: 2026-04-05
- Workspace: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`
- Time zone: `America/Phoenix`

## Purpose

Use baton-pass automation so work continues without manual "please proceed" prompts.

The workflow uses:

- one overnight continuation automation
- one morning summary automation

## Overnight Continuation Contract

Each overnight run must:

1. check local Phoenix time and self-skip outside `10:00 PM` to `6:00 AM`
2. read the newest handoff in `docs/handoffs/`
3. read the live tracker in `docs/roadmap/2026-04-05-next-phase-review-task-list.md`
4. resume exactly one active experiment or the next pending experiment
5. preserve the current rerank baseline unless a measured experiment explicitly replaces it
6. stop at one of these boundaries:
   - experiment decision reached
   - blocker discovered
   - context risk getting high
   - validation completed and next task should be handed off

Before ending, the run must always leave:

- tracker update
- review note if an experiment was measured
- fresh handoff doc
- exact next-session prompt inside the handoff

## Morning Summary Contract

The morning summary run must:

1. read the newest handoff and live tracker
2. produce a short executive summary only
3. report:
   - what changed overnight
   - latest accepted/rejected experiment decision
   - current baseline
   - blockers, if any
   - next planned task
4. avoid implementation work

## Stop Rules

The overnight automation must stop and hand off instead of pushing through when:

- multiple repo truths conflict and would change the baseline decision
- a benchmark is still running and its result is needed before the next experiment
- a change would require broad refactoring
- a change would alter more than one experimental variable
- the next step depends on product intent not discoverable from the repo

## Handoff Naming Convention

Continue versioned handoffs in `docs/handoffs/` with:

- date
- version
- experiment milestone
- decision state

Each handoff should include:

- verified current baseline
- completed work
- in-flight work
- artifact paths
- validation status
- next-session prompt

## Automation Prompt Text

### Overnight Continuation

Resume work on the curriculum matcher from the newest handoff in `docs/handoffs/`.

First:
- check the current local Phoenix time and exit immediately unless it is between `10:00 PM` and `6:00 AM`
- read the newest handoff in `docs/handoffs/`
- read the live tracker in `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

Then:
- continue exactly one active experiment or the next pending experiment
- keep the workflow benchmark-driven and one-variable-at-a-time
- preserve the Tkinter operator surface in `src/curriculum_matcher/app.py`
- avoid broad refactors
- preserve the accepted matcher and rerank baselines unless a measured experiment explicitly replaces them

Stop only when:
- an experiment decision is reached
- a blocker is discovered
- context risk is getting high
- validation is complete and the next task should be handed off

Before ending, always leave:
- tracker update
- review note if the experiment was measured
- fresh handoff doc in `docs/handoffs/`
- exact next-session prompt inside the handoff

If a benchmark is still running when the run ends, treat that as the current milestone and write the handoff around the in-flight benchmark rather than starting a new experiment.

### Morning Summary

Read the newest handoff in `docs/handoffs/` and the live tracker in `docs/roadmap/2026-04-05-next-phase-review-task-list.md`.

Produce a short executive summary only. Include:
- what changed overnight
- latest accepted/rejected experiment decision
- current matcher baseline
- current rerank baseline
- blockers, if any
- exact next planned task

Do not continue implementation work.

## Default Schedule

- overnight continuation cadence: hourly
- overnight allowed window: `10:00 PM` to `6:00 AM` Phoenix time
- morning summary: `7:30 AM` Phoenix time

## Success Criteria

- no manual continuation prompts are needed
- each morning has one latest handoff and one concise summary
- no experiment advances without explicit decision artifacts
- baseline drift is avoided because each run re-anchors from the newest handoff and tracker
