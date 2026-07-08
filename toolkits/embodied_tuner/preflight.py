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

"""Preflight validator for the embodied auto-tuner.

Preflight composes the baseline Hydra config + a knob delta, runs the
knob schema, runs a local placement-legality check
(:func:`toolkits.embodied_tuner.placement_enum.is_legal_placement`), and
runs a targeted subset of the divisibility assertions found in
``rlinf/config.py`` (``validate_cfg`` / ``validate_embodied_cfg``) —
without touching the GPU or starting Ray.

The targeted divisibility checks mirror these RLinf assertions (line
numbers in ``rlinf/config.py``):

- ``total_num_envs % env_world_size == 0`` (line 962, embodied train)
- ``total_num_envs % env_world_size % pipeline_stage_num == 0`` (line 965)
- ``max_steps_per_rollout_epoch % num_action_chunks == 0`` (line 980)
- ``global_batch_size % (micro_batch_size * actor_world_size) == 0``
  (lines 1363-1368, FSDP branch)
- ``(total_num_envs // env_world_size) % rollout_world_size == 0`` and
  ``(total_num_envs // env_world_size) % actor_world_size == 0`` —
  routing-layer assertion in ``rlinf/scheduler/worker/routing.py:139``
  (``CommMapper.get_dst_ranks``). Not mirrored by
  ``validate_embodied_cfg``, so preflight is the only pre-launch gate for
  this. See ``toolkits/embodied_tuner/wiki/07-constraints.md`` §2.6.

The reason for re-implementing instead of calling RLinf's validators
directly: ``validate_embodied_cfg`` instantiates
``HybridComponentPlacement(cfg, Cluster())`` at ``rlinf/config.py:922``,
and ``Cluster.__init__`` calls ``ray.init`` (``rlinf/scheduler/cluster/cluster.py:332``).
Preflight is contractually GPU-free and must not start Ray. We compute
``actor_world_size`` / ``env_world_size`` directly from the
``cluster.component_placement`` GPU range strings, which is exactly what
``HybridComponentPlacement`` would do, just without spinning up a
distributed runtime.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from toolkits.embodied_tuner.placement_enum import (
    PlacementParseError,
    is_legal_placement,
    parse_range_spec,
)
from toolkits.embodied_tuner.schema import KnobSchema, KnobSchemaError


class PreflightError(RuntimeError):
    """Raised for infrastructure-level preflight failures (compose, IO)."""


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of :func:`compose_and_validate`.

    Attributes:
        ok: ``True`` when every check passed.
        errors: Tuple of structured error messages. Empty when ``ok``.
        resolved_cfg: The fully Hydra-resolved ``DictConfig`` (or ``None``
            when Hydra composition itself failed).
        resolved_config_sha: SHA-256 of the resolved YAML, with sorted
            keys, suitable for ledger reproducibility (or ``None`` when
            composition failed).
        placement_kind: Categorical placement classification on success
            (``"collocated"``, ``"disaggregated"``, ``"hybrid"``, ``"all"``).
    """

    ok: bool
    errors: tuple[str, ...] = ()
    resolved_cfg: DictConfig | None = None
    resolved_config_sha: str | None = None
    placement_kind: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compose_and_validate(
    baseline_path: Path | str,
    delta: Mapping[str, object] | None = None,
    *,
    hydra_overrides: Sequence[str] = (),
    schema: KnobSchema | None = None,
    num_gpus: int = 8,
) -> ValidationResult:
    """Compose ``baseline + delta + hydra_overrides`` and validate the result.

    Args:
        baseline_path: Path to the baseline Hydra config (must be a
            ``.yaml`` file inside a Hydra config directory; e.g.
            ``examples/embodiment/config/maniskill_ppo_openvla.yaml``).
        delta: Optional knob delta from the LLM critic. Keys are
            Hydra dotted paths; values follow :class:`KnobSchema`.
        hydra_overrides: Extra Hydra override strings (``"key=value"``)
            to apply in addition to ``delta``. These are applied AFTER
            ``delta`` so they win on collisions.
        schema: Knob schema for validating ``delta``. Defaults to
            :class:`KnobSchema()` (canonical domains).
        num_gpus: Total GPU count of the target single-node host.

    Returns:
        A :class:`ValidationResult` carrying ``ok``, errors, the resolved
        config, and a SHA-256 of the resolved YAML.

    Raises:
        PreflightError: when the baseline path is missing or Hydra
            composition itself fails (these are infrastructure errors,
            not validation failures the critic should retry).
    """
    baseline = Path(baseline_path)
    if not baseline.is_file():
        raise PreflightError(f"baseline config not found: {baseline}")

    schema = schema or KnobSchema()
    delta = dict(delta or {})
    errors: list[str] = []

    # 1. Schema check on the delta (cheap; fail-fast).
    try:
        schema.validate(delta)
    except KnobSchemaError as exc:
        errors.append(f"schema: {exc}")

    # 2. Compose the resolved Hydra config.
    try:
        resolved = _compose_cfg(baseline, delta, hydra_overrides)
    except Exception as exc:  # noqa: BLE001 — surface Hydra error verbatim
        message = str(exc) or type(exc).__name__
        return ValidationResult(
            ok=False,
            errors=tuple(errors + [f"hydra-compose: {message}"]),
        )

    sha = _sha256_of_resolved(resolved)

    # 3. Placement legality (local check; no Cluster()).
    placement_kind: str | None = None
    placement = OmegaConf.select(resolved, "cluster.component_placement")
    if placement is None:
        errors.append("placement: cluster.component_placement is missing from resolved config")
    else:
        try:
            placement_map = _placement_to_str_map(placement)
        except PreflightError as exc:
            errors.append(f"placement: {exc}")
            placement_map = None
        if placement_map is not None:
            ok, reason = is_legal_placement(placement_map, num_gpus=num_gpus)
            if ok:
                placement_kind = reason
            else:
                errors.append(f"placement: {reason}")

    # 4. Targeted divisibility checks (mirrors validate_embodied_cfg /
    #    validate_cfg without calling them, to avoid Cluster()/ray.init).
    if placement is not None and not any(e.startswith("placement:") for e in errors):
        try:
            errors.extend(_check_divisibility(resolved, num_gpus))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"divisibility: unexpected error: {exc}")

    return ValidationResult(
        ok=not errors,
        errors=tuple(errors),
        resolved_cfg=resolved,
        resolved_config_sha=sha,
        placement_kind=placement_kind,
    )


# ---------------------------------------------------------------------------
# Hydra composition
# ---------------------------------------------------------------------------


def _compose_cfg(
    baseline_path: Path,
    delta: Mapping[str, object],
    hydra_overrides: Sequence[str],
) -> DictConfig:
    """Compose the baseline + delta + extra overrides via Hydra.

    Hydra's ``compose`` API requires the config directory to be
    registered. ``initialize_config_dir`` is the right entry point for
    in-process composition; ``version_base=None`` keeps the call working
    across Hydra 1.1+ and 1.4 dev builds.

    Embodied baselines (e.g. ``maniskill_ppo_openvla.yaml``) declare
    ``hydra.searchpath: [file://${oc.env:EMBODIED_PATH}/config/]`` so the
    ``env``/``model``/``training_backend`` sub-configs in
    ``examples/embodiment/config/`` resolve. We set ``EMBODIED_PATH`` and
    ``REPO_PATH`` from the baseline path itself so preflight works even
    when the user has not exported them.
    """
    import os

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = baseline_path.parent
    config_name = baseline_path.stem

    # examples/embodiment/config/foo.yaml -> EMBODIED_PATH=examples/embodiment
    # examples/embodiment/config -> .parent twice to the repo root.
    embodied_path = config_dir.parent
    repo_path = embodied_path.parent.parent
    os.environ.setdefault("EMBODIED_PATH", str(embodied_path))
    os.environ.setdefault("REPO_PATH", str(repo_path))

    override_tokens = _format_delta_as_overrides(delta) + list(hydra_overrides)

    # Hydra's GlobalHydra is process-wide; guard against re-entry that
    # would otherwise raise "GlobalHydra is already initialized".
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = compose(config_name=config_name, overrides=override_tokens)
    return cfg


def _format_delta_as_overrides(delta: Mapping[str, object]) -> list[str]:
    """Convert a knob delta to Hydra override strings."""
    out: list[str] = []
    for key, value in delta.items():
        if isinstance(value, bool):
            out.append(f"{key}={'true' if value else 'false'}")
        elif isinstance(value, Mapping):
            # For inline mappings (e.g. placement dict), use Hydra's
            # ``{k: v, ...}`` syntax and prefix with ``++`` to allow
            # struct-mode assignment of arbitrary nested keys.
            body = ", ".join(f"{k}: {_render_scalar(v)}" for k, v in value.items())
            out.append(f"++{key}={{{body}}}")
        else:
            out.append(f"{key}={value}")
    return out


def _render_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return f"{value}"


# ---------------------------------------------------------------------------
# Placement map extraction
# ---------------------------------------------------------------------------


def _placement_to_str_map(placement: DictConfig | object) -> dict[str, str]:
    """Convert ``cluster.component_placement`` to a ``{name: range}`` map.

    RLinf accepts both the dict-of-strings form
    (``actor: 0-7\\nenv: 0-3\\nrollout: 4-7``) and a list-of-dicts form
    (rarely seen for embodied configs). We support the dict form here and
    raise :class:`PreflightError` for unrecognised structures.
    """
    if isinstance(placement, DictConfig):
        out: dict[str, str] = {}
        for name in ("actor", "env", "rollout"):
            if name not in placement:
                raise PreflightError(
                    f"placement is missing component {name!r}; got keys {list(placement.keys())}"
                )
            value = placement[name]
            if isinstance(value, DictConfig):
                if "placement" not in value:
                    raise PreflightError(
                        f"placement[{name!r}] is a dict without a 'placement' field"
                    )
                value = value["placement"]
            out[name] = str(value)
        return out
    raise PreflightError(
        f"cluster.component_placement has unsupported type {type(placement).__name__}"
    )


# ---------------------------------------------------------------------------
# Divisibility checks
# ---------------------------------------------------------------------------


def _check_divisibility(cfg: DictConfig, num_gpus: int) -> list[str]:
    """Apply the targeted divisibility checks (see module docstring)."""
    errors: list[str] = []

    placement = _placement_to_str_map(cfg.cluster.component_placement)
    env_gpus = parse_range_spec(placement["env"], num_gpus=num_gpus)
    actor_gpus = parse_range_spec(placement["actor"], num_gpus=num_gpus)
    rollout_gpus = parse_range_spec(placement["rollout"], num_gpus=num_gpus)
    env_world_size = len(env_gpus)
    actor_world_size = len(actor_gpus)
    rollout_world_size = len(rollout_gpus)

    # env.train.total_num_envs % env_world_size == 0  (rlinf/config.py:962)
    total_num_envs = _get(cfg, "env.train.total_num_envs")
    if total_num_envs is not None and env_world_size > 0:
        if total_num_envs % env_world_size != 0:
            errors.append(
                f"divisibility: env.train.total_num_envs={total_num_envs} is not divisible "
                f"by env_world_size={env_world_size} (see rlinf/config.py:962)"
            )
        else:
            stage_num = _get(cfg, "rollout.pipeline_stage_num") or 1
            per_rank = total_num_envs // env_world_size
            # rlinf/config.py:965 — total_num_envs % env_world_size % pipeline_stage_num == 0
            if stage_num and per_rank % stage_num != 0:
                errors.append(
                    f"divisibility: per-rank env count {per_rank} is not divisible "
                    f"by rollout.pipeline_stage_num={stage_num} (see rlinf/config.py:965)"
                )

            # Routing-layer assertion (rlinf/scheduler/worker/routing.py:139,
            # CommMapper.get_dst_ranks): per-env-rank batch must be divisible
            # by every downstream world size the env worker sends to.
            # See toolkits/embodied_tuner/wiki/07-constraints.md §2.6.
            if rollout_world_size > 0 and per_rank % rollout_world_size != 0:
                errors.append(
                    f"divisibility: per-rank env count {per_rank} "
                    f"(total_num_envs={total_num_envs} // env_world_size={env_world_size}) "
                    f"is not divisible by rollout_world_size={rollout_world_size} "
                    f"— will crash rlinf/scheduler/worker/routing.py:139 "
                    f"(env→rollout send). See wiki/07-constraints.md §2.6."
                )
            if actor_world_size > 0 and per_rank % actor_world_size != 0:
                errors.append(
                    f"divisibility: per-rank env count {per_rank} "
                    f"(total_num_envs={total_num_envs} // env_world_size={env_world_size}) "
                    f"is not divisible by actor_world_size={actor_world_size} "
                    f"— will crash rlinf/scheduler/worker/routing.py:139 "
                    f"(rollout→actor send). See wiki/07-constraints.md §2.6."
                )

    # max_steps_per_rollout_epoch % num_action_chunks == 0  (rlinf/config.py:980)
    max_steps = _get(cfg, "env.train.max_steps_per_rollout_epoch")
    num_action_chunks = _first_existing(
        cfg,
        ("actor.model.num_action_chunks", "actor.num_action_chunks"),
    )
    if max_steps is not None and num_action_chunks:
        if max_steps % num_action_chunks != 0:
            errors.append(
                f"divisibility: env.train.max_steps_per_rollout_epoch={max_steps} is not "
                f"divisible by num_action_chunks={num_action_chunks} (see rlinf/config.py:980)"
            )

    # actor.global_batch_size % (actor.micro_batch_size * actor_world_size) == 0
    # (rlinf/config.py:1363-1368, FSDP branch)
    gbs = _get(cfg, "actor.global_batch_size")
    mbs = _get(cfg, "actor.micro_batch_size")
    if gbs is not None and mbs is not None and actor_world_size > 0:
        denom = mbs * actor_world_size
        if denom == 0 or gbs % denom != 0:
            errors.append(
                f"divisibility: actor.global_batch_size={gbs} is not divisible by "
                f"(actor.micro_batch_size={mbs} * actor_world_size={actor_world_size}) "
                f"(see rlinf/config.py:1363-1368)"
            )

    return errors


def _get(cfg: DictConfig, dotted: str) -> object | None:
    """Return ``cfg[dotted]`` or ``None`` if the key is absent."""
    return OmegaConf.select(cfg, dotted)


def _first_existing(cfg: DictConfig, keys: Sequence[str]) -> object | None:
    """Return the first non-``None`` value of ``cfg[key]`` across ``keys``."""
    for key in keys:
        value = OmegaConf.select(cfg, key)
        if value is not None:
            return value
    return None


# ---------------------------------------------------------------------------
# SHA-256 of resolved YAML
# ---------------------------------------------------------------------------


def _sha256_of_resolved(cfg: DictConfig) -> str:
    """Return the SHA-256 of the resolved YAML, key-sorted for reproducibility.

    Interpolations such as ``oc.env:OMNIGIBSON_DATA_PATH`` are NOT resolved
    here: doing so would fail when those env vars are absent at preflight
    time (the tuner runs without the full embodied env setup), and the
    hash should depend only on the composed config text, not on the host
    environment.
    """
    text = OmegaConf.to_yaml(cfg, sort_keys=True, resolve=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
