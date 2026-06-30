# BitLesson Knowledge Base

This file is project-specific. Keep entries precise and reusable for future rounds.

## Entry Template (Strict)

Use this exact field order for every entry:

```markdown
## Lesson: <unique-id>
Lesson ID: <BL-YYYYMMDD-short-name>
Scope: <component/subsystem/files>
Problem Description: <specific failure mode with trigger conditions>
Root Cause: <direct technical cause>
Solution: <exact fix that resolved the problem>
Constraints: <limits, assumptions, non-goals>
Validation Evidence: <tests/commands/logs/PR evidence>
Source Rounds: <round numbers where problem appeared and was solved>
```

## Entries

## Lesson: pgrep-env-vs-argv
Lesson ID: BL-2026-06-30-pgrep-env-vs-argv
Scope: toolkits/embodied_tuner/runner.py — per-trial subprocess cleanup
Problem Description: When the per-trial cleanup uses a tag set ONLY as an environment variable (e.g. `RLINF_TUNER_TRIAL_ID=<id>`) and identifies orphan workers via `pgrep -f <tag>`, the cleanup silently misses every orphan because `pgrep -f` matches argv (the command-line arguments) and NOT the process environment. Ray workers spawned by the trial inherit the env var but their argv typically does not contain it, so `pgrep -f` returns no matches even when several orphan Ray workers are still running.
Root Cause: `pgrep -f`'s `-f` flag matches against the full command line (argv), not the inherited environment block exposed at `/proc/<pid>/environ`.
Solution: Scan `/proc/<pid>/environ` directly for the tag pattern, union those PIDs with whatever `pgrep -f` returns, and kill the union. This catches env-tagged orphans on Linux without requiring the train script to forward the tag into argv. Skip gracefully on non-Linux hosts (no `/proc`).
Constraints: POSIX Linux only for the `/proc`-based scan. The pgrep fallback path is still useful when callers DO inject the tag into argv. Treat cleanup as best-effort; never raise from the cleanup path.
Validation Evidence: `toolkits/embodied_tuner/runner.py::_pids_with_env_match`; the orphan-related tests in `tests/test_runner.py` continue to pass via dependency injection of `pgrep_runner`/`kill_runner`.
Source Rounds: 1 (initial scoped-cleanup design), 3 (Codex review fix to scan `/proc/<pid>/environ`)

