# Automation State

This directory holds the baton-ownership state for manual-first curriculum automations.

Operator guide:

- `user-guide.md` explains how to start, stop, and use the canonical automations by task-list item.

## Canonical Automations

- Morning summary automation: `curriculum-morning-4`
- Worker automation: `curriculum-overnight-4`
- Bounded batch worker automation: `curriculum-overnight-batch`

All other curriculum automations are legacy copies and should remain paused.

## Ownership File

`current_run.json` is the single source of truth for whether an automation currently owns the baton.

It records:

- `automation_role`: `morning`, `worker`, or `worker_batch`
- `automation_id`: automation folder id such as `curriculum-overnight-4`
- `repo_path`: authoritative live repo path
- `run_started_at`: ISO-8601 timestamp
- `heartbeat_at`: ISO-8601 timestamp
- `current_handoff`: latest handoff path when known
- `current_task_id`: roadmap task id when known
- `status`: `idle`, `running`, `completed`, or `abandoned`
- `taken_over_from`: prior owner id if a stale run was reclaimed
- `notes`: short human-readable status note

## Duplicate Protection Rules

- Morning runs must not proceed if a worker run is marked `running` with a heartbeat less than 30 minutes old.
- Worker runs and batch worker runs must not proceed if another worker or batch worker run is marked `running` with a heartbeat less than 90 minutes old.
- If the lock is stale, the new run may take over and must record that takeover in `taken_over_from` and `notes`.

## Worker Stop Rule

One worker run may complete exactly one roadmap unit of work:

- one task moved to `accepted`, `rejected`, `blocked`, or `deferred`
- one blocker investigation that ends with an explicit stop decision
- one benchmarked experiment with artifacts and a written decision

Worker runs must not chain into a second roadmap task just because time remains.

Batch worker runs may process up to `3` roadmap units, but they must still keep experiment-changing work serial and must stop when a blocker, decision boundary, or incompatible next task appears.
