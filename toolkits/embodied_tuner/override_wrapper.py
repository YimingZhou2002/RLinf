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

"""Hydra-override wrapper around ``examples/embodiment/run_embodiment.sh``.

The stock entry script only injects ``runner.logger.log_path=${LOG_DIR}``
into the Hydra invocation. The auto-tuner needs to forward arbitrary
Hydra overrides (knob deltas, ``runner.max_epochs=3``, etc.) without
modifying the stock script (which is forbidden by the plan's Path
Boundaries).

``OverrideWrapper.build_invocation(...)`` returns a fully formed
:class:`LaunchSpec` (argv, env, log_dir) the trial runner can pass to
``subprocess.Popen``. Hydra override precedence — later overrides win —
is preserved by appending user overrides AFTER the
``runner.logger.log_path`` injection that the stock script does. This
matches the stock script's CMD layout in ``run_embodiment.sh:58`` and
means user-supplied ``runner.logger.log_path=<X>`` would override the
wrapper-injected value, which is the correct Hydra semantic.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


# Env-var defaults mirroring ``examples/embodiment/run_embodiment.sh``.
# Only the ones the stock script unconditionally exports or defaults are
# replicated. ``ROBOTWIN_PATH`` etc. remain user-controlled via the
# caller's environment.
_STOCK_DEFAULTS: dict[str, str] = {
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "OMNIGIBSON_NO_OMNI_LOGS": "1",
    "OMNIGIBSON_DEBUG": "0",
    "OMNIGIBSON_HEADLESS": "1",
    "LIBERO_TYPE": "standard",
    "ROBOT_PLATFORM": "LIBERO",
}


class OverrideWrapperError(RuntimeError):
    """Raised when an invocation cannot be built (e.g. baseline missing)."""


@dataclass(frozen=True)
class LaunchSpec:
    """A fully formed RLinf trial invocation, ready to spawn.

    Attributes:
        argv: ``subprocess``-style argv list. ``argv[0]`` is the Python
            interpreter; ``argv[1]`` is the path to
            ``train_embodied_agent.py``.
        env: Environment-variable mapping to pass to ``subprocess``.
            Contains the stock-script defaults plus ``RLINF_TUNER_TRIAL_ID``
            for scoped cleanup (consumed by the trial runner in AC-5).
        log_dir: Absolute path of the per-trial log directory the wrapper
            instructs Hydra to use. The runner is responsible for
            creating this directory before launch.
        config_name: Hydra ``--config-name`` value (echoed for telemetry).
        baseline_overrides: The Hydra override tokens that come before
            user overrides (currently only ``runner.logger.log_path=...``).
            Exposed for debugging and ledger persistence.
        user_overrides: The Hydra override tokens supplied by the caller,
            preserved as a tuple in stable iteration order.
    """

    argv: tuple[str, ...]
    env: dict[str, str]
    log_dir: Path
    config_name: str
    baseline_overrides: tuple[str, ...]
    user_overrides: tuple[str, ...] = field(default_factory=tuple)

    def overrides_in_order(self) -> tuple[str, ...]:
        """Return the full Hydra override list in resolution order."""
        return self.baseline_overrides + self.user_overrides


@dataclass(frozen=True)
class OverrideWrapper:
    """Build :class:`LaunchSpec` objects for the auto-tuner trial runner.

    Attributes:
        repo_path: RLinf repository root (the directory that contains
            ``examples/embodiment/run_embodiment.sh`` and ``rlinf/``).
            Defaults to two levels above this file.
        config_dir: Hydra ``--config-path`` directory. Defaults to the
            embodied config directory under ``examples/embodiment/config``.
        python_executable: Interpreter to launch with. Defaults to
            :data:`sys.executable` so the wrapper inherits the active venv.
    """

    repo_path: Path
    config_dir: Path
    python_executable: str = field(default_factory=lambda: sys.executable)

    @classmethod
    def for_repo(cls, repo_path: Path | str) -> OverrideWrapper:
        """Construct a wrapper using the canonical RLinf layout under ``repo_path``."""
        root = Path(repo_path).resolve()
        return cls(
            repo_path=root,
            config_dir=root / "examples" / "embodiment" / "config",
        )

    @property
    def train_script(self) -> Path:
        """Absolute path to ``train_embodied_agent.py``."""
        return self.repo_path / "examples" / "embodiment" / "train_embodied_agent.py"

    def build_invocation(
        self,
        config_name: str,
        overrides: Sequence[str | tuple[str, object]] | Mapping[str, object] = (),
        *,
        log_dir: Path | str,
        trial_id: str,
        extra_env: Mapping[str, str] | None = None,
    ) -> LaunchSpec:
        """Construct a :class:`LaunchSpec` for the given trial.

        Args:
            config_name: The Hydra ``--config-name`` (e.g. ``maniskill_ppo_openvla``).
            overrides: Hydra overrides forwarded after the stock-script
                injection. Accepts a sequence of ``"key=value"`` strings
                or ``(key, value)`` tuples, or a mapping ``{key: value}``.
                The wrapper formats each entry verbatim and preserves order;
                callers are responsible for using Hydra-legal syntax
                (e.g. ``"actor.micro_batch_size=64"`` or
                ``"cluster.component_placement={actor:0-7,env:0-3,rollout:4-7}"``).
            log_dir: Absolute path used for ``runner.logger.log_path``.
                The runner is responsible for creating this directory.
            trial_id: Stable identifier exported as ``RLINF_TUNER_TRIAL_ID``
                so the trial runner can ``pgrep -f`` against it for
                orphan-worker cleanup.
            extra_env: Optional extra environment variables merged on top
                of the stock-script defaults and the caller's current env.

        Returns:
            A frozen :class:`LaunchSpec`.

        Raises:
            OverrideWrapperError: when the configured ``train_script`` or
                ``config_dir`` does not exist.
        """
        if not self.train_script.is_file():
            raise OverrideWrapperError(
                f"train script not found: {self.train_script}; "
                "is the repo_path correct?"
            )
        if not self.config_dir.is_dir():
            raise OverrideWrapperError(
                f"Hydra config dir not found: {self.config_dir}"
            )
        if not str(trial_id):
            raise OverrideWrapperError("trial_id must be a non-empty string")

        log_dir_path = Path(log_dir)
        baseline_overrides = (f"runner.logger.log_path={log_dir_path}",)
        user_overrides = _normalize_overrides(overrides)

        argv: tuple[str, ...] = (
            self.python_executable,
            str(self.train_script),
            "--config-path",
            f"{self.config_dir}/",
            "--config-name",
            config_name,
            *baseline_overrides,
            *user_overrides,
        )

        env = _build_env(self.repo_path, trial_id, extra_env)

        return LaunchSpec(
            argv=argv,
            env=env,
            log_dir=log_dir_path,
            config_name=config_name,
            baseline_overrides=baseline_overrides,
            user_overrides=user_overrides,
        )


def _normalize_overrides(
    overrides: Sequence[str | tuple[str, object]] | Mapping[str, object],
) -> tuple[str, ...]:
    """Convert ``overrides`` into a tuple of Hydra ``key=value`` strings."""
    if isinstance(overrides, Mapping):
        items = list(overrides.items())
    else:
        items = []
        for entry in overrides:
            if isinstance(entry, str):
                if "=" not in entry:
                    raise OverrideWrapperError(
                        f"override {entry!r} must be 'key=value' or (key, value)"
                    )
                items.append(tuple(entry.split("=", 1)))
            elif isinstance(entry, tuple) and len(entry) == 2:
                items.append((entry[0], entry[1]))
            else:
                raise OverrideWrapperError(
                    f"override {entry!r}: expected str or (key, value) tuple"
                )

    formatted: list[str] = []
    for key, value in items:
        if not isinstance(key, str) or not key:
            raise OverrideWrapperError(f"override key must be a non-empty string, got {key!r}")
        formatted.append(f"{key}={_format_hydra_value(value)}")
    return tuple(formatted)


def _format_hydra_value(value: object) -> str:
    """Format a Python value as a Hydra override RHS.

    Booleans use lowercase ``true``/``false`` per Hydra convention.
    Mappings are formatted as inline ``{k:v, ...}`` blocks.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        body = ", ".join(f"{k}: {_format_hydra_value(v)}" for k, v in value.items())
        return "{" + body + "}"
    return str(value)


def _build_env(
    repo_path: Path,
    trial_id: str,
    extra_env: Mapping[str, str] | None,
) -> dict[str, str]:
    """Construct the subprocess env mirroring the stock script's exports."""
    env = dict(os.environ)
    env.setdefault("REPO_PATH", str(repo_path))
    env.setdefault("EMBODIED_PATH", str(repo_path / "examples" / "embodiment"))
    env.setdefault(
        "SRC_FILE",
        str(repo_path / "examples" / "embodiment" / "train_embodied_agent.py"),
    )
    for key, default in _STOCK_DEFAULTS.items():
        env.setdefault(key, default)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_path}:{existing_pp}" if existing_pp else str(repo_path)
    )
    # Scoped cleanup tag — AC-5 reads RLINF_TUNER_TRIAL_ID via pgrep -f.
    env["RLINF_TUNER_TRIAL_ID"] = str(trial_id)
    if extra_env:
        env.update(extra_env)
    return env
