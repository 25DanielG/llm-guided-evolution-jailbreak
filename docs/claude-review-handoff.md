# Independent review handoff

Codex owns all implementation, deployment and fixes. Review an isolated source
export created by `src.review_snapshot`, at the commit in `REVIEW-SNAPSHOT.json`.
Do not edit the active checkout or run Slurm submission/staging commands.

Read `docs/overnight-operations.md`, `docs/deployment-validation-2026-11-11.md`,
`docs/shared-model-serving.md` and the selected run's `status.json` plus
`scheduler-observation.json`. The run record schema is version 1; job state is
separate from application state; its contract is `docs/run-status.schema.json`.
Follow `console_log` only within the provided
sanitized artifacts. GPU smoke evidence is each model's `result.json` and logs.

Focus on checkpoint pair consistency, restored generation/RNG/archive state,
failure and signal handling, cleanup, bounded polling, and whether logs support
the reported inference/FP8/reasoning claims. A stale generation alone is not a
confirmed hang; unavailable accounting is not proof of a failed job.

Return a short findings list with severity, source/line or log evidence and a
proposed fix for Codex. Do not invent successful inference or silently recommend
restarting from seed. No credentials/tokens should be provided or requested;
use your own authenticated tools. Keep the review bounded; no continuous agent
or unlimited subscription usage is part of this operating procedure.
