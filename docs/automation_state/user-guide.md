# Manual-First Automation User Guide

This guide explains how to use the curriculum automations without babysitting them all day or repeatedly typing "please proceed".

## Quick Start

If you want the simplest operating flow:

1. Read the latest handoff:
   - `docs/handoffs/2026-04-06-v034-e1-cross-encoder-rejected-handoff.md`
2. If you want a status snapshot only, run `Curriculum Morning Canonical`.
3. If you want one roadmap item advanced, run `Curriculum Worker Canonical`.
4. If you want a bounded chunk of up to `3` compatible roadmap items advanced, run `Curriculum Worker Batch`.
5. When the run finishes, read the newest handoff and the tracker update before launching another run.

Current roadmap note:

- the current roadmap has no active experiment after `E1` and `E2` were rejected
- the next worker or batch run should start by defining the next experiment set, not by silently inventing a new one

## What Exists Now

Three canonical automations are available and paused by default:

- `Curriculum Morning Canonical` (`curriculum-morning-4`)
- `Curriculum Worker Canonical` (`curriculum-overnight-4`)
- `Curriculum Worker Batch` (`curriculum-overnight-batch`)

All older curriculum automations are legacy copies and should stay paused.

## What Each Automation Does

### Morning automation

Use `Curriculum Morning Canonical` when you want a status snapshot.

It will:

- read the newest handoff
- read the live task tracker
- give you a short executive summary
- not continue implementation work

### Worker automation

Use `Curriculum Worker Canonical` when you want the baton-passing workflow to do actual work.

It will:

- read the newest handoff
- read the live task tracker
- claim the baton lock if no other worker owns it
- complete exactly one roadmap unit of work
- leave a tracker update, review note if needed, and fresh handoff
- stop after that one task unit is resolved

This is the key behavior change: one run should advance one task, then stop cleanly.

### Batch worker automation

Use `Curriculum Worker Batch` when you want the system to work through a small bounded set of compatible task-list items in one session without constant manual approvals.

It will:

- read the newest handoff
- read the live task tracker
- claim the baton lock if no other worker owns it
- complete up to `3` roadmap units of work in one run
- keep experiment-changing work serial
- use subagents or parallel agents only for supporting sidecar tasks, not for competing experiment changes
- stop early if it hits a blocker, a decision boundary, or an incompatible next task
- leave tracker updates, review notes if needed, and a fresh handoff

## How To Start An Automation

### Start a morning summary run

Use this when you want to know where the project stands before deciding what to run next.

Steps:

1. Open the automation UI.
2. Find `Curriculum Morning Canonical`.
3. Manually launch it.
4. Read the summary inbox item it produces.

Do not unpause it for daily recurrence unless you intentionally want scheduled summaries again.

### Start a worker run

Use this when you want the system to advance the roadmap without needing a manual "please proceed" after every small checkpoint.

Steps:

1. Open the automation UI.
2. Find `Curriculum Worker Canonical`.
3. Manually launch it.
4. Let it complete one roadmap unit of work.
5. Review the resulting inbox item, tracker update, and handoff.
6. If you want another task advanced, launch it again.

You do not need to stay in the loop during the run. The worker is expected to make one bounded decision-complete pass and stop on its own.

### Start a batch worker run

Use this when you want a longer autonomous session that can consume a small chunk of the roadmap without you manually re-launching after every item.

Steps:

1. Open the automation UI.
2. Find `Curriculum Worker Batch`.
3. Manually launch it.
4. Let it work through up to `3` compatible roadmap units.
5. Review the resulting inbox item, tracker updates, and fresh handoff.

Use this mode when the next few tasks are likely to be compatible in sequence. If the roadmap is at a major decision boundary, use the one-task worker instead.

## Which Automation Should I Pick?

Use `Curriculum Morning Canonical` when:

- you want a concise status update
- you are deciding whether to launch a worker
- you do not want code or benchmark work to continue

Use `Curriculum Worker Canonical` when:

- you want one clean task boundary per run
- the next task is important or risky enough that you want a review point immediately after it
- the roadmap is at a decision boundary

Use `Curriculum Worker Batch` when:

- you want to reduce manual relaunches
- the next few task-list items are sequentially compatible
- you want the run to use subagents or parallel sidecar work where helpful
- you still want strict protection against parallel competing experiment changes

## How To Use It By Task List Item

The live tracker is:

- `docs/roadmap/2026-04-05-next-phase-review-task-list.md`

The worker automation is designed to consume this tracker one task at a time.

The batch worker automation is designed to consume this tracker in a bounded chunk of up to `3` sequentially compatible roadmap units.

Use this operating pattern:

1. Start one worker run.
2. Let it complete one task item or blocker decision.
3. Read the fresh handoff and tracker update.
4. Decide whether you want another task advanced now.
5. If yes, start the worker again.

This gives you control by task-list item rather than by constant chat approvals.

If you want a bounded chunk instead of a single item, use this pattern:

1. Start one batch worker run.
2. Let it process up to `3` compatible roadmap units.
3. Review the fresh handoff and tracker updates.
4. If you want another bounded chunk, start the batch worker again.

For the current project state, the first item in that chunk should be:

- define the next experiment set after the rejection of `E1` and `E2`

## What Counts As One Task Unit

One worker run may complete exactly one of these:

- one task moved to `accepted`, `rejected`, `blocked`, or `deferred`
- one blocker investigation with an explicit stop decision
- one benchmarked experiment with artifacts and a written decision

It must not:

- finish one experiment and immediately start another
- keep running just because more time is available
- consume multiple roadmap items in one launch

Batch worker mode may consume up to `3` roadmap units, but it must still stop when:

- a blocker is found
- a roadmap decision boundary appears
- the next task would compete with the active experiment change
- the current task sequence is no longer cleanly serial

## How To Stop An Automation

### Normal stop

You usually do not need to manually stop the canonical worker. It is supposed to stop by itself after one task unit.

### If you want to stop before the next run

Keep the canonical automations paused. Launch them manually only when wanted. That is the main stop control.

### If a run appears stuck

Check:

- `docs/automation_state/current_run.json`

Look for:

- `status`
- `automation_id`
- `heartbeat_at`
- `current_task_id`
- `current_handoff`

Interpretation:

- if `status` is `idle`, nothing owns the baton
- if `status` is `running` with a recent heartbeat, let it finish
- if `status` is `running` but stale, a later worker run may reclaim ownership

Stale thresholds:

- morning summary: 30 minutes
- worker run: 90 minutes

## How Duplicate Protection Works

The ownership file is:

- `docs/automation_state/current_run.json`

Rules:

- a morning summary will not run if a worker already owns the baton with a fresh lock
- a worker run will not start if another worker or batch worker has a fresh lock
- a batch worker run will not start if another worker or batch worker has a fresh lock
- if a lock is stale, the next worker may take over

This is what prevents accidental overlapping runs when the UI does not clearly show that something is already in progress.

## Recommended Daily Pattern

If you want low supervision:

1. Run `Curriculum Morning Canonical` to get oriented.
2. If the next task looks safe to delegate, run `Curriculum Worker Canonical`.
3. Let it finish one task item.
4. Review the inbox item and handoff.
5. If you want more progress, run the worker again.

This gives you a task-by-task baton passing loop without needing to type "please proceed" throughout the day.

If you want higher autonomy with bounded risk:

1. Run `Curriculum Morning Canonical` if you want a status check first.
2. Run `Curriculum Worker Batch`.
3. Let it work through up to `3` compatible roadmap units.
4. Review the inbox item and latest handoff.
5. Re-run it only if you want another bounded chunk.

## How Batch Mode Uses Subagents

`Curriculum Worker Batch` is the automation to use when you want the best fit for subagents or parallel help.

It is allowed to use subagents for bounded support work such as:

- comparing artifacts
- slice analysis
- drafting review notes
- running focused test or validation passes
- implementing disjoint code changes that support the same active task

It must not:

- run two competing experiment changes in parallel
- open two different experiment decisions at once
- use parallel agents to bypass the one-variable-at-a-time roadmap rule

The safe mental model is:

- serial for experiment decisions
- parallel for analysis, validation, and non-competing sidecar work

## If You Want Scheduled Runs Again Later

The canonical automations still have schedule fields because the app requires recurring schedule formats.

If you temporarily want scheduled execution:

1. unpause the canonical automation you want
2. set the desired weekly or hourly schedule in the UI
3. let it run during that period
4. pause it again when you want to return to manual-first control

Do not unpause the legacy automation copies.
