# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-trial subprocess runner for the embodied auto-tuner.

The runner launches a :class:`~toolkits.embodied_tuner.override_wrapper.LaunchSpec`
in its own POSIX process group (``os.setsid``), enforces a per-trial
timeout, and on failure/timeout escalates SIGTERM → SIGKILL, invokes a
configurable ``ray stop --force`` hook, and uses a scoped
``pgrep -f "RLINF_TUNER_TRIAL_ID=<id>"`` check to detect orphan workers
spawned by Ray. Orphan PIDs are killed before the next trial starts.

Profiler env exports
====================

By default, the runner adds these env vars to the launch spec so the
``rlinf_timeline`` autopatch (see ``profiler/rlinf_timeline/autopatch.py``)
and the nvitop/NVML samplers turn on without the user having to ``source``
``profiler/enable2.sh``:

- ``RLINF_TIMELINE=1``
- ``RLINF_TIMELINE_WORKER_TIMER=1``
- ``RLINF_TIMELINE_ACTOR_TRAINING=1``
- ``RLINF_TIMELINE_DIR=auto``
- ``RLINF_NVITOP=1``
- ``RLINF_NVML=1``

``disable_profiler=True`` (CLI ``--no-profiler``) skips all timeline
flags; ``disable_memory_telemetry=True`` (CLI ``--no-collect-memory``)
skips ``RLINF_NVITOP`` / ``RLINF_NVML`` only.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from toolkits.embodied_tuner.override_wrapper import LaunchSpec


# Profiler env vars added to every trial by default. Mirrors the
# ``profiler/enable2.sh`` exports plus the two NVITOP/NVML flags that
# the stock script leaves commented out.
_TIMELINE_ENV: dict[str, str] = {
    "RLINF_TIMELINE": "1",
    "RLINF_TIMELINE_WORKER_TIMER": "1",
    "RLINF_TIMELINE_ACTOR_TRAINING": "1",
    "RLINF_TIMELINE_DIR": "auto",
}
_MEMORY_ENV: dict[str, str] = {
    "RLINF_NVITOP": "1",
    "RLINF_NVML": "1",
}


class TrialRunnerError(RuntimeError):
    """Raised for runner-internal errors (e.g. failed to create log dir)."""


@dataclass(frozen=True)
class TrialOutcome:
    """Result of a single ``TrialRunner.launch`` call.

    Attributes:
        log_dir: Absolute path of the trial's log directory.
        returncode: Subprocess exit code (``None`` only if the runner
            failed before ``Popen`` could start).
        timed_out: ``True`` when the subprocess exceeded the timeout and
            the runner had to escalate signals.
        wall_clock_seconds: Wall-clock seconds from launch to reap.
        cleanup_outcome: One of ``"ok"``, ``"sigkill_required"``,
            ``"ray_stop_failed"``, ``"orphans_killed"``, ``"orphans_remain"``.
            The value reflects the worst-case outcome of the cleanup
            pipeline (later stages may override earlier ``"ok"``).
        stdout_path: Path to captured stdout log (``run_embodiment.log``
            inside ``log_dir``).
        spec: The :class:`LaunchSpec` actually launched (after env merging).
    """

    log_dir: Path
    returncode: int | None
    timed_out: bool
    wall_clock_seconds: float
    cleanup_outcome: str
    stdout_path: Path
    spec: LaunchSpec


@dataclass(frozen=True)
class TrialRunner:
    """Launch a :class:`LaunchSpec` and reap its subprocess safely.

    Attributes:
        timeout_seconds: Default per-trial wall-clock budget (overridable
            in :meth:`launch`). Matches the plan's AC-5 default of
            ``2700`` seconds.
        sigterm_grace_seconds: After SIGTERM the runner waits this long
            before escalating to SIGKILL.
        disable_profiler: When ``True``, the ``RLINF_TIMELINE*`` env vars
            are NOT exported. The parser will classify the resulting
            trial ``(OK, METRICS_PARTIAL)`` because ``timeline/*.jsonl``
            will be missing.
        disable_memory_telemetry: When ``True``, the ``RLINF_NVITOP`` /
            ``RLINF_NVML`` env vars are NOT exported. Timeline still on.
        ray_stop_hook: Callable invoked after subprocess reap.
            Defaults to ``_default_ray_stop_hook`` which runs
            ``ray stop --force`` if the ``ray`` CLI is on PATH.
        pgrep_runner: Callable that returns the list of PIDs whose
            command line contains the supplied pattern. Defaults to
            ``_default_pgrep_runner`` which shells out to ``pgrep -f``.
            Injectable for tests so the suite never depends on a real
            ``pgrep`` binary.
        kill_runner: Callable that sends ``signal.SIGKILL`` to a list of
            PIDs. Injectable for tests.
    """

    timeout_seconds: float = 2700.0
    sigterm_grace_seconds: float = 30.0
    disable_profiler: bool = False
    disable_memory_telemetry: bool = False
    ray_stop_hook: Callable[[], bool] | None = None
    pgrep_runner: Callable[[str], list[int]] | None = None
    kill_runner: Callable[[list[int], int], None] | None = None
    extra_env: Mapping[str, str] = field(default_factory=dict)

    # ----- Public API ------------------------------------------------------

    def launch(self, spec: LaunchSpec, *, timeout: float | None = None) -> TrialOutcome:
        """Spawn ``spec``, wait for completion or timeout, then clean up.

        Args:
            spec: Output of :meth:`OverrideWrapper.build_invocation`.
            timeout: Optional override of ``self.timeout_seconds`` for
                this call.

        Returns:
            A :class:`TrialOutcome` describing what happened.

        Raises:
            TrialRunnerError: when the runner cannot create the log
                directory or the subprocess cannot be spawned at all.
        """
        deadline = float(timeout if timeout is not None else self.timeout_seconds)
        if deadline <= 0:
            raise TrialRunnerError(f"timeout must be positive, got {deadline}")

        log_dir = self._prepare_log_dir(spec.log_dir)
        stdout_path = log_dir / "run_embodiment.log"
        effective_spec = self._with_profiler_env(spec)

        wall_start = time.monotonic()
        try:
            with stdout_path.open("w") as fh:
                # ``start_new_session=True`` puts the child in its own
                # process group (POSIX), so we can signal the whole tree
                # with ``os.killpg``.
                proc = subprocess.Popen(  # noqa: S603 — argv is trusted
                    list(effective_spec.argv),
                    env=effective_spec.env,
                    cwd=str(_repo_path_from_env(effective_spec.env)),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as exc:
            raise TrialRunnerError(f"failed to spawn subprocess: {exc}") from exc

        returncode, timed_out, cleanup_outcome = self._reap(
            proc=proc,
            timeout=deadline,
            trial_id=effective_spec.env["RLINF_TUNER_TRIAL_ID"],
        )
        wall_seconds = time.monotonic() - wall_start

        return TrialOutcome(
            log_dir=log_dir,
            returncode=returncode,
            timed_out=timed_out,
            wall_clock_seconds=wall_seconds,
            cleanup_outcome=cleanup_outcome,
            stdout_path=stdout_path,
            spec=effective_spec,
        )

    # ----- Implementation -------------------------------------------------

    def _prepare_log_dir(self, log_dir: Path) -> Path:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TrialRunnerError(f"cannot create log_dir {log_dir}: {exc}") from exc
        return log_dir

    def _with_profiler_env(self, spec: LaunchSpec) -> LaunchSpec:
        """Return a new :class:`LaunchSpec` with profiler env added."""
        env = dict(spec.env)
        if not self.disable_profiler:
            for key, value in _TIMELINE_ENV.items():
                env.setdefault(key, value)
        if not self.disable_memory_telemetry:
            for key, value in _MEMORY_ENV.items():
                env.setdefault(key, value)
        for key, value in self.extra_env.items():
            env[key] = value
        return replace(spec, env=env)

    def _reap(
        self,
        *,
        proc: subprocess.Popen,
        timeout: float,
        trial_id: str,
    ) -> tuple[int | None, bool, str]:
        """Wait for ``proc``; on timeout escalate signals; then clean up.

        Returns:
            Tuple ``(returncode, timed_out, cleanup_outcome)``.
        """
        cleanup_outcome = "ok"
        timed_out = False
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._signal_pgroup(proc.pid, signal.SIGTERM)
            try:
                returncode = proc.wait(timeout=self.sigterm_grace_seconds)
            except subprocess.TimeoutExpired:
                self._signal_pgroup(proc.pid, signal.SIGKILL)
                cleanup_outcome = "sigkill_required"
                try:
                    returncode = proc.wait(timeout=self.sigterm_grace_seconds)
                except subprocess.TimeoutExpired:
                    # The process did not die even after SIGKILL — likely
                    # stuck in a kernel D-state. Surface this; caller
                    # decides whether to fail the campaign.
                    cleanup_outcome = "orphans_remain"
                    returncode = None

        # ray stop --force
        ray_ok = self._invoke_ray_stop()
        if not ray_ok and cleanup_outcome == "ok":
            cleanup_outcome = "ray_stop_failed"

        # Scoped orphan sweep.
        orphans = self._find_orphans(trial_id)
        if orphans:
            self._kill_orphans(orphans)
            # Re-check to confirm.
            still = self._find_orphans(trial_id)
            if still:
                cleanup_outcome = "orphans_remain"
            elif cleanup_outcome == "ok":
                cleanup_outcome = "orphans_killed"

        return returncode, timed_out, cleanup_outcome

    @staticmethod
    def _signal_pgroup(pid: int, sig: int) -> None:
        """Send ``sig`` to the process group of ``pid``; swallow ``ProcessLookupError``."""
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError):
            # The pgroup already exited; nothing to do.
            return

    def _invoke_ray_stop(self) -> bool:
        hook = self.ray_stop_hook or _default_ray_stop_hook
        try:
            return bool(hook())
        except Exception:  # noqa: BLE001 — ray cleanup is best-effort
            return False

    def _find_orphans(self, trial_id: str) -> list[int]:
        runner = self.pgrep_runner or _default_pgrep_runner
        pattern = f"RLINF_TUNER_TRIAL_ID={trial_id}"
        try:
            return list(runner(pattern))
        except Exception:  # noqa: BLE001 — pgrep failure shouldn't tank cleanup
            return []

    def _kill_orphans(self, pids: list[int]) -> None:
        killer = self.kill_runner or _default_kill_runner
        try:
            killer(pids, signal.SIGKILL)
        except Exception:  # noqa: BLE001 — kill is best-effort
            return


# ---------------------------------------------------------------------------
# Default hook implementations
# ---------------------------------------------------------------------------


def _default_ray_stop_hook() -> bool:
    """Run ``ray stop --force`` if available; treat absence as success.

    A return value of ``True`` means "no leftover ray workers expected".
    When ``ray`` is not on PATH, the auto-tuner has no managed Ray cluster
    in the first place — return ``True`` so the cleanup outcome stays ``ok``.
    """
    if shutil.which("ray") is None:
        return True
    try:
        result = subprocess.run(  # noqa: S603, S607 — known argv
            ["ray", "stop", "--force"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _default_pgrep_runner(pattern: str) -> list[int]:
    """Return PIDs matching ``pattern`` via ``pgrep -f``; ``[]`` if absent."""
    if shutil.which("pgrep") is None:
        return []
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["pgrep", "-f", pattern],
            capture_output=True,
            timeout=10.0,
            check=False,
            text=True,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode not in (0, 1):  # 0=found, 1=none
        return []
    return [int(line) for line in result.stdout.split() if line.strip().isdigit()]


def _default_kill_runner(pids: list[int], sig: int) -> None:
    """Send ``sig`` to each PID; swallow lookup errors."""
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            continue


def _repo_path_from_env(env: Mapping[str, str]) -> Path:
    """Return ``REPO_PATH`` from the launch env, falling back to ``cwd``."""
    value = env.get("REPO_PATH")
    if value:
        return Path(value)
    return Path.cwd()
