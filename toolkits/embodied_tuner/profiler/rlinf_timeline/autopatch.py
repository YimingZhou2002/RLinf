"""Import-time monkey patching controlled entirely by environment variables."""

from __future__ import annotations

import functools
import importlib
import importlib.abc
import importlib.machinery
import inspect
import os
import sys
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

from .nvitop_sampler import is_nvitop_enabled, start_local_process_nvitop_sampler
from .nvml_sampler import is_nvml_enabled, start_local_process_sampler
from .writer import append_event, is_enabled


@dataclass(frozen=True)
class PatchSpec:
    module: str
    qualname: str
    component: str
    tag: str


_INSTALLED = False
_PATCHED: set[tuple[str, str]] = set()
# Specs that failed to patch because their target module/class was still
# partially initialized at import-hook time (a latent circular import in the
# target package, e.g. rlinf.envs.maniskill). The class typically finishes
# defining later in the same process; _retry_deferred() re-attempts these on
# subsequent imports until they succeed.
_DEFERRED_SPECS: list[PatchSpec] = []
_WORKER_TIMER_PATCHED = False
_ACTOR_TRAINING_PATCHED = False
_WORKER_NVML_PATCHED = False
_RUNNER_NVML_PATCHED: set[str] = set()
_BEHAVIOR_NVML_PATCHED = False
_RESOURCE_WORKER_INIT_PATCHED: set[tuple[str, str]] = set()
_RESOURCE_WORKER_MODULES = {
    "rlinf.workers.actor.fsdp_actor_worker",
    "rlinf.workers.env.async_env_worker",
    "rlinf.workers.env.env_worker",
    "rlinf.workers.reward.reward_worker",
    "rlinf.workers.rollout.hf.async_huggingface_worker",
    "rlinf.workers.rollout.hf.huggingface_worker",
}
_DEFAULT_WORKER_TIMER_EXCLUDE_TAGS = {
    "interact",
    "run_interact_once",
    "generate_one_epoch",
    "recv_rollout_results",
    "run_training",
}


def _resource_sampling_enabled() -> bool:
    return is_nvml_enabled() or is_nvitop_enabled()


def _start_local_resource_samplers(
    *,
    component: str,
    rank: int,
    global_step_getter: Callable[[], Any] | None = None,
    extra: dict[str, Any] | None = None,
    log_dir: str | None = None,
) -> None:
    nvml_dir = None
    nvitop_dir = None
    env_nvml_dir = os.environ.get("RLINF_NVML_DIR")
    env_nvitop_dir = os.environ.get("RLINF_NVITOP_DIR")
    if env_nvml_dir and env_nvml_dir.lower() != "auto":
        nvml_dir = env_nvml_dir
    elif log_dir:
        nvml_dir = os.path.join(log_dir, "nvml")
    if env_nvitop_dir and env_nvitop_dir.lower() != "auto":
        nvitop_dir = env_nvitop_dir
    elif log_dir:
        nvitop_dir = os.path.join(log_dir, "nvitop")

    start_local_process_sampler(
        component=component,
        rank=rank,
        global_step_getter=global_step_getter,
        extra=extra,
        nvml_dir=nvml_dir,
    )
    start_local_process_nvitop_sampler(
        component=component,
        rank=rank,
        global_step_getter=global_step_getter,
        extra=extra,
        nvitop_dir=nvitop_dir,
    )


def _ensure_resource_sampling_for_object(obj: Any) -> None:
    if not _resource_sampling_enabled():
        return
    log_dir = _log_dir_from_cfg(getattr(obj, "cfg", None))
    _start_local_resource_samplers(
        component=_component_from_context((obj,), "worker"),
        rank=_rank_from_context((obj,)),
        global_step_getter=lambda: _step_from_context((obj,)),
        extra={
            "class_name": type(obj).__qualname__,
            "module": type(obj).__module__,
            "worker_name": getattr(obj, "_worker_name", None),
        },
        log_dir=log_dir,
    )


def _debug(msg: str) -> None:
    if os.environ.get("RLINF_TIMELINE_DEBUG", "").lower() in {"1", "true", "yes"}:
        print(f"[rlinf_timeline] {msg}", file=sys.stderr)


def _csv_set(value: str | None) -> set[str]:
    if value is None:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _worker_timer_exclude_tags() -> set[str]:
    value = os.environ.get("RLINF_TIMELINE_WORKER_TIMER_EXCLUDE_TAGS")
    if value is None:
        return set(_DEFAULT_WORKER_TIMER_EXCLUDE_TAGS)
    return _csv_set(value)


def _split_spec(raw: str) -> PatchSpec | None:
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        return None

    parts = [p.strip() for p in raw.split(":")]
    if len(parts) < 2:
        _debug(f"skip invalid patch spec: {raw!r}")
        return None

    module, qualname = parts[0], parts[1]
    component = parts[2] if len(parts) >= 3 and parts[2] else qualname.split(".")[0]
    tag = parts[3] if len(parts) >= 4 and parts[3] else qualname
    return PatchSpec(module=module, qualname=qualname, component=component, tag=tag)


def _read_patch_specs() -> list[PatchSpec]:
    raw_items: list[str] = []

    env_specs = os.environ.get("RLINF_TIMELINE_PATCHES", "")
    if env_specs:
        raw_items.extend(env_specs.replace("\n", ";").split(";"))

    spec_file = os.environ.get("RLINF_TIMELINE_PATCH_FILE")
    if spec_file:
        try:
            with open(spec_file, encoding="utf-8") as f:
                raw_items.extend(f.readlines())
        except OSError as exc:
            _debug(f"cannot read patch file {spec_file!r}: {exc}")

    specs = []
    for raw in raw_items:
        spec = _split_spec(raw)
        if spec is not None:
            specs.append(spec)
    return specs


def _rank_from_context(args: tuple[Any, ...]) -> int:
    if args:
        obj = args[0]
        for attr in ("_rank", "rank", "global_rank", "local_rank"):
            value = getattr(obj, attr, None)
            if isinstance(value, int):
                return value
        worker = getattr(obj, "_rlinf_worker", None)
        get_parent_rank = getattr(worker, "get_parent_rank", None)
        if callable(get_parent_rank):
            try:
                return int(get_parent_rank())
            except Exception:
                pass

    for name in ("RANK", "LOCAL_RANK", "WORKER_RANK"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return 0


def _step_from_context(args: tuple[Any, ...]) -> int | None:
    if not args:
        return None
    obj = args[0]
    for attr in (
        "global_steps",
        "global_step",
        "_global_steps",
        "_global_step",
        "version",
        "_version",
    ):
        value = _safe_int(getattr(obj, attr, None))
        if value is not None:
            return value
    return None


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                return getter(key)
            except Exception:
                return default
        except Exception:
            return default
    return getattr(obj, key, default)


def _timeline_dir_from_context(args: tuple[Any, ...]) -> str | None:
    env_dir = os.environ.get("RLINF_TIMELINE_DIR")
    if env_dir and env_dir.lower() != "auto":
        return env_dir

    if not args:
        return None
    cfg = getattr(args[0], "cfg", None)
    runner = _cfg_get(cfg, "runner")
    timeline_dir = _cfg_get(runner, "timeline_dir")
    if timeline_dir:
        return str(timeline_dir)
    logger = _cfg_get(runner, "logger")
    log_path = _cfg_get(logger, "log_path")
    if log_path:
        return os.path.join(str(log_path), "timeline")
    return None


def _log_dir_from_cfg(cfg: Any) -> str | None:
    runner = _cfg_get(cfg, "runner")
    logger = _cfg_get(runner, "logger")
    log_path = _cfg_get(logger, "log_path")
    if log_path:
        return str(log_path)
    return None


def _set_log_dir_env_from_cfg(cfg: Any) -> None:
    log_dir = _log_dir_from_cfg(cfg)
    if log_dir:
        os.environ["RLINF_LOG_DIR"] = os.path.abspath(log_dir)


def _runner_cfg_from_init_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "cfg" in kwargs:
        return kwargs["cfg"]
    if len(args) >= 2:
        return args[1]
    return None


def _component_from_context(args: tuple[Any, ...], fallback: str) -> str:
    if not args:
        return fallback
    obj = args[0]
    name = obj.__class__.__name__.lower()
    module = obj.__class__.__module__.lower()
    if "env" in name or ".env." in module:
        return "env"
    if "rollout" in name or ".rollout." in module:
        return "rollout"
    if "actor" in name or ".actor." in module:
        return "actor"
    if "reward" in name or ".reward." in module:
        return "reward"
    if "runner" in name or ".runner" in module:
        return "runner"
    return fallback


def _safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _next_call_index(obj: Any, key: str) -> int:
    counters = getattr(obj, "_rlinf_timeline_counters", None)
    if counters is None:
        counters = {}
        try:
            setattr(obj, "_rlinf_timeline_counters", counters)
        except Exception:
            return 0
    value = int(counters.get(key, 0))
    counters[key] = value + 1
    return value


def _bind_call_args(func: Callable[..., Any], self_obj: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        bound = inspect.signature(func).bind_partial(self_obj, *args, **kwargs)
    except Exception:
        return {}
    return dict(bound.arguments)


def _loop_metadata_from_sequence(obj: Any, timer_tag: str, call_index: int, metadata: dict[str, Any]) -> None:
    stage_count = (
        _safe_int(getattr(obj, "num_pipeline_stages", None))
        or _safe_int(getattr(obj, "stage_num", None))
        or 1
    )
    train_chunks = _safe_int(getattr(obj, "n_train_chunk_steps", None))
    eval_chunks = _safe_int(getattr(obj, "n_eval_chunk_steps", None))

    if timer_tag == "generate_one_epoch":
        metadata.setdefault("rollout_epoch", call_index)
        metadata.setdefault("phase", "train")
        return

    if timer_tag == "predict":
        mode = str(metadata.get("mode") or "train")
        if mode == "eval" and eval_chunks:
            total = max(eval_chunks * stage_count, 1)
            pos = call_index % total
            metadata.setdefault("eval_rollout_epoch", call_index // total)
            metadata.setdefault("chunk_step", pos // stage_count)
            metadata.setdefault("stage_id", pos % stage_count)
            metadata.setdefault("phase", "eval")
            return

        if train_chunks:
            train_total = train_chunks * stage_count
            total = train_total + stage_count
            pos = call_index % max(total, 1)
            metadata.setdefault("rollout_epoch", call_index // max(total, 1))
            if pos < train_total:
                metadata.setdefault("chunk_step", pos // stage_count)
                metadata.setdefault("stage_id", pos % stage_count)
                metadata.setdefault("phase", "train")
            else:
                metadata.setdefault("chunk_step", train_chunks)
                metadata.setdefault("stage_id", pos - train_total)
                metadata.setdefault("phase", "bootstrap")
        return

    if timer_tag in {
        "env_interact_step",
        "recv_rollout_results",
        "compute_bootstrap_rewards",
        "get_reward_model_output",
        "recv_reward_results",
    } and train_chunks:
        train_total = train_chunks * stage_count
        if timer_tag == "env_interact_step":
            total = train_total
            pos = call_index % max(total, 1)
            metadata.setdefault("rollout_epoch", call_index // max(total, 1))
            metadata.setdefault("chunk_step", pos // stage_count)
            metadata.setdefault("stage_id", pos % stage_count)
            metadata.setdefault("phase", "train")
        else:
            total = train_total + stage_count
            pos = call_index % max(total, 1)
            metadata.setdefault("rollout_epoch", call_index // max(total, 1))
            if pos < train_total:
                metadata.setdefault("chunk_step", pos // stage_count)
                metadata.setdefault("stage_id", pos % stage_count)
                metadata.setdefault("phase", "train")
            else:
                metadata.setdefault("chunk_step", train_chunks)
                metadata.setdefault("stage_id", pos - train_total)
                metadata.setdefault("phase", "bootstrap")


def _call_metadata(
    *,
    obj: Any,
    func: Callable[..., Any],
    timer_tag: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    call_index: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"call_index": call_index}
    bound = _bind_call_args(func, obj, args, kwargs)
    for key in (
        "stage_id",
        "chunk_step",
        "chunk_step_idx",
        "rollout_epoch",
        "eval_rollout_epoch",
        "mode",
        "last_run",
        "cooperative_yield",
    ):
        if key in bound:
            value = bound[key]
            if key in {"stage_id", "chunk_step", "chunk_step_idx", "rollout_epoch", "eval_rollout_epoch"}:
                value = _safe_int(value)
            if value is not None:
                metadata[key] = value

    if "chunk_step_idx" in metadata and "chunk_step" not in metadata:
        metadata["chunk_step"] = metadata["chunk_step_idx"]

    for attr, key in (
        ("version", "global_step"),
        ("rollout_epoch", "configured_rollout_epochs"),
        ("n_train_chunk_steps", "configured_train_chunk_steps"),
        ("n_eval_chunk_steps", "configured_eval_chunk_steps"),
        ("num_pipeline_stages", "configured_pipeline_stages"),
        ("stage_num", "configured_pipeline_stages"),
    ):
        if key in metadata:
            continue
        value = _safe_int(getattr(obj, attr, None))
        if value is not None:
            metadata[key] = value

    _loop_metadata_from_sequence(obj, timer_tag, call_index, metadata)
    return metadata


def _actor_training_metadata(obj: Any, state: dict[str, Any], phase: str) -> dict[str, Any]:
    idx = int(state.get("microbatch_index", -1))
    grad_accum = _safe_int(getattr(obj, "gradient_accumulation", None)) or 1
    global_batch_index = idx // grad_accum if idx >= 0 else None
    raw_count = int(state.get(f"{phase}_count", 0))
    metadata: dict[str, Any] = {
        "phase": phase,
        "call_index": max(raw_count - 1, 0),
        "microbatch_index": idx if idx >= 0 else None,
        "grad_accum_index": idx % grad_accum if idx >= 0 else None,
        "global_batch_index": global_batch_index,
        "configured_gradient_accumulation": grad_accum,
    }
    for attr, key in (
        ("version", "global_step"),
        ("optimizer_steps", "optimizer_steps"),
        ("_rank", "actor_rank"),
        ("_world_size", "actor_world_size"),
    ):
        value = _safe_int(getattr(obj, attr, None))
        if value is not None:
            metadata[key] = value
    return {k: v for k, v in metadata.items() if v is not None}


def _append_actor_training_event(
    obj: Any,
    state: dict[str, Any],
    *,
    phase: str,
    tag: str,
    t0: float,
    t1: float,
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = _actor_training_metadata(obj, state, phase)
    if extra:
        metadata.update(extra)
    append_event(
        component="actor",
        rank=_rank_from_context((obj,)),
        tag=tag,
        t0=t0,
        t1=t1,
        global_step=_step_from_context((obj,)),
        extra=metadata,
        timeline_dir=_timeline_dir_from_context((obj,)),
    )


def _wrap_callable(func: Callable[..., Any], spec: PatchSpec) -> Callable[..., Any]:
    if getattr(func, "_rlinf_timeline_wrapped", False):
        return func

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if _DEFERRED_SPECS:
                _retry_deferred()
            if args:
                _ensure_resource_sampling_for_object(args[0])
            call_index = _next_call_index(args[0], spec.tag) if args else 0
            t0 = time.time()
            error = None
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                error = type(exc).__name__
                raise
            finally:
                extra = {"module": spec.module, "qualname": spec.qualname}
                if args:
                    extra.update(
                        _call_metadata(
                            obj=args[0],
                            func=func,
                            timer_tag=spec.tag,
                            args=args[1:],
                            kwargs=kwargs,
                            call_index=call_index,
                        )
                    )
                if error:
                    extra["exception"] = error
                append_event(
                    component=spec.component,
                    rank=_rank_from_context(args),
                    tag=spec.tag,
                    t0=t0,
                    t1=time.time(),
                    global_step=_step_from_context(args),
                    extra=extra,
                    timeline_dir=_timeline_dir_from_context(args),
                )

        async_wrapper._rlinf_timeline_wrapped = True  # type: ignore[attr-defined]
        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _DEFERRED_SPECS:
            _retry_deferred()
        if args:
            _ensure_resource_sampling_for_object(args[0])
        call_index = _next_call_index(args[0], spec.tag) if args else 0
        t0 = time.time()
        error = None
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            extra = {"module": spec.module, "qualname": spec.qualname}
            if args:
                extra.update(
                    _call_metadata(
                        obj=args[0],
                        func=func,
                        timer_tag=spec.tag,
                        args=args[1:],
                        kwargs=kwargs,
                        call_index=call_index,
                    )
                )
            if error:
                extra["exception"] = error
            append_event(
                component=spec.component,
                rank=_rank_from_context(args),
                tag=spec.tag,
                t0=t0,
                t1=time.time(),
                global_step=_step_from_context(args),
                extra=extra,
                timeline_dir=_timeline_dir_from_context(args),
            )

    wrapper._rlinf_timeline_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def _patch_one(module: ModuleType, spec: PatchSpec) -> None:
    key = (spec.module, spec.qualname)
    if key in _PATCHED:
        return

    parts = spec.qualname.split(".")
    parent: Any = module
    for name in parts[:-1]:
        parent = getattr(parent, name)

    attr_name = parts[-1]
    target = getattr(parent, attr_name)
    if not callable(target):
        _debug(f"target is not callable: {spec.module}:{spec.qualname}")
        return

    setattr(parent, attr_name, _wrap_callable(target, spec))
    _PATCHED.add(key)
    _debug(f"patched {spec.module}:{spec.qualname} as {spec.component}/{spec.tag}")


def _retry_deferred() -> None:
    """Re-attempt specs that previously failed because their module was partial.

    Safe to call from any import hook: it only touches modules already present
    and fully loaded in sys.modules, and never changes import control flow.
    """
    if not _DEFERRED_SPECS:
        return
    remaining: list[PatchSpec] = []
    for spec in _DEFERRED_SPECS:
        module = sys.modules.get(spec.module)
        if module is None:
            remaining.append(spec)
            continue
        try:
            _patch_one(module, spec)
            _debug(
                f"deferred-retry patched {spec.module}:{spec.qualname} "
                f"as {spec.component}/{spec.tag}"
            )
        except Exception:
            # Still not ready (class not yet defined); keep for a later retry.
            remaining.append(spec)
    _DEFERRED_SPECS[:] = remaining


def _patch_module(module: ModuleType, specs: list[PatchSpec]) -> None:
    for spec in specs:
        try:
            _patch_one(module, spec)
        except Exception as exc:
            _debug(f"failed to patch {spec.module}:{spec.qualname}: {exc}")
            # Target module/class likely not fully defined yet (circular import);
            # retry opportunistically on subsequent imports.
            if (spec.module, spec.qualname) not in _PATCHED:
                _DEFERRED_SPECS.append(spec)
    _retry_deferred()



class _TimelineLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, specs: list[PatchSpec]) -> None:
        self.loader = loader
        self.specs = specs

    def create_module(self, spec):
        create_module = getattr(self.loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.loader.exec_module(module)  # type: ignore[attr-defined]
        if _resource_sampling_enabled():
            module_name = getattr(module, "__name__", "")
            if module_name == _WorkerNVMLFinder.MODULE:
                _patch_worker_nvml_module(module)
            elif module_name == _BehaviorNVMLFinder.MODULE:
                _patch_behavior_nvml_module(module)
            elif module_name.startswith(_RunnerNVMLFinder.MODULE_PREFIX):
                _patch_runner_nvml_module(module)
        _patch_module(module, self.specs)
        if (
            module.__name__ == _ActorTrainingFinder.MODULE
            and os.environ.get("RLINF_TIMELINE_ACTOR_TRAINING", "").lower()
            in {"1", "true", "yes", "on", "y"}
        ):
            _patch_actor_training_module(module)


class _TimelineFinder(importlib.abc.MetaPathFinder):
    def __init__(self, by_module: dict[str, list[PatchSpec]]) -> None:
        self.by_module = by_module

    def find_spec(self, fullname: str, path: Any, target: Any = None):
        specs = self.by_module.get(fullname)
        if not specs:
            return None

        found = importlib.machinery.PathFinder.find_spec(fullname, path)
        if found is None or found.loader is None:
            return found

        if isinstance(found.loader, _TimelineLoader):
            return found

        found.loader = _TimelineLoader(found.loader, specs)
        return found


class _WorkerTimerLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader) -> None:
        self.loader = loader

    def create_module(self, spec):
        create_module = getattr(self.loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.loader.exec_module(module)  # type: ignore[attr-defined]
        if _resource_sampling_enabled():
            _patch_worker_nvml_module(module)
        _patch_worker_timer_module(module)


class _WorkerTimerFinder(importlib.abc.MetaPathFinder):
    MODULE = "rlinf.scheduler.worker.worker"

    def find_spec(self, fullname: str, path: Any, target: Any = None):
        if fullname != self.MODULE:
            return None
        found = importlib.machinery.PathFinder.find_spec(fullname, path)
        if found is None or found.loader is None:
            return found
        if isinstance(found.loader, _WorkerTimerLoader):
            return found
        found.loader = _WorkerTimerLoader(found.loader)
        return found


class _ActorTrainingLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader) -> None:
        self.loader = loader

    def create_module(self, spec):
        create_module = getattr(self.loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.loader.exec_module(module)  # type: ignore[attr-defined]
        _patch_actor_training_module(module)
        if _resource_sampling_enabled() and module.__name__.startswith(_RunnerNVMLFinder.MODULE_PREFIX):
            _patch_runner_nvml_module(module)


class _ActorTrainingFinder(importlib.abc.MetaPathFinder):
    MODULE = "rlinf.workers.actor.fsdp_actor_worker"

    def find_spec(self, fullname: str, path: Any, target: Any = None):
        if fullname != self.MODULE:
            return None
        found = importlib.machinery.PathFinder.find_spec(fullname, path)
        if found is None or found.loader is None:
            return found
        if isinstance(found.loader, _ActorTrainingLoader):
            return found
        found.loader = _ActorTrainingLoader(found.loader)
        return found


class _WorkerNVMLLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader) -> None:
        self.loader = loader

    def create_module(self, spec):
        create_module = getattr(self.loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.loader.exec_module(module)  # type: ignore[attr-defined]
        _patch_worker_nvml_module(module)
        if (
            os.environ.get("RLINF_TIMELINE_WORKER_TIMER", "").lower()
            in {"1", "true", "yes", "on", "y"}
        ):
            _patch_worker_timer_module(module)


class _WorkerNVMLFinder(importlib.abc.MetaPathFinder):
    MODULE = "rlinf.scheduler.worker.worker"

    def find_spec(self, fullname: str, path: Any, target: Any = None):
        if fullname != self.MODULE:
            return None
        found = importlib.machinery.PathFinder.find_spec(fullname, path)
        if found is None or found.loader is None:
            return found
        if isinstance(found.loader, _WorkerNVMLLoader):
            return found
        found.loader = _WorkerNVMLLoader(found.loader)
        return found


class _RunnerNVMLLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader) -> None:
        self.loader = loader

    def create_module(self, spec):
        create_module = getattr(self.loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.loader.exec_module(module)  # type: ignore[attr-defined]
        _patch_runner_nvml_module(module)
        specs = _read_patch_specs()
        if specs:
            module_specs = [spec for spec in specs if spec.module == module.__name__]
            if module_specs:
                _patch_module(module, module_specs)


class _RunnerNVMLFinder(importlib.abc.MetaPathFinder):
    MODULE_PREFIX = "rlinf.runners."

    def find_spec(self, fullname: str, path: Any, target: Any = None):
        if not fullname.startswith(self.MODULE_PREFIX):
            return None
        found = importlib.machinery.PathFinder.find_spec(fullname, path)
        if found is None or found.loader is None:
            return found
        if isinstance(found.loader, _RunnerNVMLLoader):
            return found
        found.loader = _RunnerNVMLLoader(found.loader)
        return found


class _BehaviorNVMLLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader) -> None:
        self.loader = loader

    def create_module(self, spec):
        create_module = getattr(self.loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.loader.exec_module(module)  # type: ignore[attr-defined]
        _patch_behavior_nvml_module(module)


class _BehaviorNVMLFinder(importlib.abc.MetaPathFinder):
    MODULE = "rlinf.envs.behavior.behavior_env"

    def find_spec(self, fullname: str, path: Any, target: Any = None):
        if fullname != self.MODULE:
            return None
        found = importlib.machinery.PathFinder.find_spec(fullname, path)
        if found is None or found.loader is None:
            return found
        if isinstance(found.loader, _BehaviorNVMLLoader):
            return found
        found.loader = _BehaviorNVMLLoader(found.loader)
        return found


class _ResourceWorkerInitLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader) -> None:
        self.loader = loader

    def create_module(self, spec):
        create_module = getattr(self.loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.loader.exec_module(module)  # type: ignore[attr-defined]
        _patch_resource_worker_init_module(module)


class _ResourceWorkerInitFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any, target: Any = None):
        if fullname not in _RESOURCE_WORKER_MODULES:
            return None
        found = importlib.machinery.PathFinder.find_spec(fullname, path)
        if found is None or found.loader is None:
            return found
        if isinstance(found.loader, _ResourceWorkerInitLoader):
            return found
        found.loader = _ResourceWorkerInitLoader(found.loader)
        return found


class _ScopedPatch:
    def __init__(self, owner: Any, name: str, replacement: Any) -> None:
        self.owner = owner
        self.name = name
        self.replacement = replacement
        self.original = None

    def __enter__(self):
        self.original = getattr(self.owner, self.name)
        setattr(self.owner, self.name, self.replacement)
        return self.original

    def __exit__(self, exc_type, exc, tb) -> None:
        setattr(self.owner, self.name, self.original)


class _DeferredRetryLoader(importlib.abc.Loader):
    """Wraps any module's loader to retry deferred patches after it finishes loading.

    Only active while _DEFERRED_SPECS is non-empty; the finder short-circuits
    otherwise so there is zero overhead on the normal import path.
    """

    def __init__(self, loader: importlib.abc.Loader) -> None:
        self.loader = loader

    def create_module(self, spec):
        create_module = getattr(self.loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.loader.exec_module(module)  # type: ignore[attr-defined]
        _retry_deferred()


class _DeferredRetryFinder(importlib.abc.MetaPathFinder):
    """Appended last in sys.meta_path; retries deferred specs on every later import.

    Specialized finders (registered at the front of sys.meta_path) claim their
    own modules and are never seen here, so no double-wrapping occurs.
    """

    def find_spec(self, fullname: str, path: Any, target: Any = None):
        if not _DEFERRED_SPECS:
            return None
        found = importlib.machinery.PathFinder.find_spec(fullname, path)
        if found is None or found.loader is None:
            return found
        if isinstance(found.loader, (_DeferredRetryLoader, _TimelineLoader)):
            return found
        found.loader = _DeferredRetryLoader(found.loader)
        return found


def _patch_actor_training_module(module: ModuleType) -> None:
    global _ACTOR_TRAINING_PATCHED
    if _ACTOR_TRAINING_PATCHED:
        return

    actor_cls = getattr(module, "EmbodiedFSDPActor", None)
    if actor_cls is None:
        _debug("EmbodiedFSDPActor not found for actor training patch")
        return

    original_run_training = getattr(actor_cls, "run_training", None)
    if original_run_training is None or getattr(
        original_run_training, "_rlinf_timeline_actor_training_wrapped", False
    ):
        _ACTOR_TRAINING_PATCHED = True
        return

    @functools.wraps(original_run_training)
    def run_training_wrapper(self, *args, **kwargs):
        _ensure_resource_sampling_for_object(self)
        state: dict[str, Any] = {
            "microbatch_index": -1,
            "forward_count": 0,
            "policy_loss_count": 0,
            "backward_count": 0,
            "optimizer_step_count": 0,
        }

        model = getattr(self, "model", None)
        model_cls = type(model) if model is not None else None
        original_model_call = getattr(model_cls, "__call__", None) if model_cls else None

        def model_call_wrapper(model_self, *m_args, **m_kwargs):
            if model_self is not model:
                return original_model_call(model_self, *m_args, **m_kwargs)
            state["microbatch_index"] = int(state.get("forward_count", 0))
            state["forward_count"] = int(state["forward_count"]) + 1
            t0 = time.time()
            error = None
            try:
                return original_model_call(model_self, *m_args, **m_kwargs)
            except Exception as exc:
                error = type(exc).__name__
                raise
            finally:
                extra = {
                    "qualname": f"{model_cls.__module__}.{model_cls.__qualname__}.__call__",
                    "model_class": model_cls.__qualname__,
                    "compute_logprobs": m_kwargs.get("compute_logprobs", None),
                    "compute_entropy": m_kwargs.get("compute_entropy", None),
                    "compute_values": m_kwargs.get("compute_values", None),
                }
                if error:
                    extra["exception"] = error
                _append_actor_training_event(
                    self,
                    state,
                    phase="forward",
                    tag="actor_forward",
                    t0=t0,
                    t1=time.time(),
                    extra=extra,
                )

        original_policy_loss = getattr(module, "policy_loss", None)

        @functools.wraps(original_policy_loss)
        def policy_loss_wrapper(*pl_args, **pl_kwargs):
            state["policy_loss_count"] = int(state.get("policy_loss_count", 0)) + 1
            t0 = time.time()
            error = None
            try:
                return original_policy_loss(*pl_args, **pl_kwargs)
            except Exception as exc:
                error = type(exc).__name__
                raise
            finally:
                extra = {
                    "qualname": "rlinf.workers.actor.fsdp_actor_worker.policy_loss",
                    "loss_type": pl_kwargs.get("loss_type", None),
                    "logprob_type": pl_kwargs.get("logprob_type", None),
                    "reward_type": pl_kwargs.get("reward_type", None),
                    "critic_warmup": pl_kwargs.get("critic_warmup", None),
                }
                if error:
                    extra["exception"] = error
                _append_actor_training_event(
                    self,
                    state,
                    phase="policy_loss",
                    tag="actor_policy_loss",
                    t0=t0,
                    t1=time.time(),
                    extra=extra,
                )

        try:
            import torch

            original_backward = torch.Tensor.backward
        except Exception:
            torch = None
            original_backward = None

        def backward_wrapper(tensor_self, *bw_args, **bw_kwargs):
            state["backward_count"] = int(state.get("backward_count", 0)) + 1
            t0 = time.time()
            error = None
            try:
                return original_backward(tensor_self, *bw_args, **bw_kwargs)
            except Exception as exc:
                error = type(exc).__name__
                raise
            finally:
                extra = {"qualname": "torch.Tensor.backward"}
                if error:
                    extra["exception"] = error
                _append_actor_training_event(
                    self,
                    state,
                    phase="backward",
                    tag="actor_backward",
                    t0=t0,
                    t1=time.time(),
                    extra=extra,
                )

        original_optimizer_step = getattr(self, "optimizer_step")

        @functools.wraps(original_optimizer_step)
        def optimizer_step_wrapper(*opt_args, **opt_kwargs):
            state["optimizer_step_count"] = int(state.get("optimizer_step_count", 0)) + 1
            t0 = time.time()
            error = None
            try:
                return original_optimizer_step(*opt_args, **opt_kwargs)
            except Exception as exc:
                error = type(exc).__name__
                raise
            finally:
                extra = {"qualname": f"{type(self).__qualname__}.optimizer_step"}
                if error:
                    extra["exception"] = error
                _append_actor_training_event(
                    self,
                    state,
                    phase="optimizer_step",
                    tag="actor_optimizer_step",
                    t0=t0,
                    t1=time.time(),
                    extra=extra,
                )

        patches = []
        if model_cls is not None and original_model_call is not None:
            patches.append(_ScopedPatch(model_cls, "__call__", model_call_wrapper))
        if original_policy_loss is not None:
            patches.append(_ScopedPatch(module, "policy_loss", policy_loss_wrapper))
        if torch is not None and original_backward is not None:
            patches.append(_ScopedPatch(torch.Tensor, "backward", backward_wrapper))
        patches.append(_ScopedPatch(self, "optimizer_step", optimizer_step_wrapper))

        exits = []
        try:
            for patch in patches:
                patch.__enter__()
                exits.append(patch)
            return original_run_training(self, *args, **kwargs)
        finally:
            for patch in reversed(exits):
                patch.__exit__(None, None, None)

    run_training_wrapper._rlinf_timeline_actor_training_wrapped = True  # type: ignore[attr-defined]
    setattr(actor_cls, "run_training", run_training_wrapper)
    _ACTOR_TRAINING_PATCHED = True
    _debug("patched EmbodiedFSDPActor.run_training for actor training timeline")


def _patch_worker_nvml_module(module: ModuleType) -> None:
    global _WORKER_NVML_PATCHED
    if _WORKER_NVML_PATCHED:
        return

    worker_cls = getattr(module, "Worker", None)
    if worker_cls is None:
        _debug("Worker class not found for NVML patch")
        return

    original_init = getattr(worker_cls, "__init__", None)
    if original_init is None or getattr(original_init, "_rlinf_nvml_wrapped", False):
        _WORKER_NVML_PATCHED = True
        return

    @functools.wraps(original_init)
    def init_wrapper(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)
        # Publish RLINF_LOG_DIR from the worker's cfg so PatchSpec-wrapped
        # methods whose `self` is not a Worker (e.g. ManiskillOffloadEnv, whose
        # .cfg is env.train without runner.logger.log_path) still resolve the
        # correct run timeline dir via default_timeline_dir().
        _set_log_dir_env_from_cfg(getattr(self, "cfg", None))
        rank = _rank_from_context((self,))
        if rank >= 0:
            _start_local_resource_samplers(
                component=_component_from_context((self,), "worker"),
                rank=rank,
                global_step_getter=lambda: _step_from_context((self,)),
                extra={
                    "class_name": type(self).__qualname__,
                    "module": type(self).__module__,
                    "worker_name": getattr(self, "_worker_name", None),
                },
            )
        return result

    init_wrapper._rlinf_nvml_wrapped = True  # type: ignore[attr-defined]
    setattr(worker_cls, "__init__", init_wrapper)
    _WORKER_NVML_PATCHED = True
    _debug("patched Worker.__init__ for resource sampling")


def _patch_runner_nvml_module(module: ModuleType) -> None:
    module_name = getattr(module, "__name__", "")
    if module_name in _RUNNER_NVML_PATCHED:
        return

    patched_any = False
    for attr_name in dir(module):
        candidate = getattr(module, attr_name, None)
        if not inspect.isclass(candidate) or not attr_name.endswith("Runner"):
            continue
        original_init = getattr(candidate, "__init__", None)
        if original_init is None or getattr(original_init, "_rlinf_nvml_wrapped", False):
            continue

        @functools.wraps(original_init)
        def init_wrapper(self, *args, __original_init=original_init, **kwargs):
            cfg = _runner_cfg_from_init_args((self, *args), kwargs)
            if cfg is not None:
                _set_log_dir_env_from_cfg(cfg)
            result = __original_init(self, *args, **kwargs)
            cfg = getattr(self, "cfg", cfg)
            if cfg is not None:
                _set_log_dir_env_from_cfg(cfg)
            log_dir = _log_dir_from_cfg(cfg)
            _start_local_resource_samplers(
                component="runner",
                rank=_rank_from_context((self,)),
                global_step_getter=lambda: _step_from_context((self,)),
                extra={
                    "class_name": type(self).__qualname__,
                    "module": type(self).__module__,
                },
                log_dir=log_dir,
            )
            return result

        init_wrapper._rlinf_nvml_wrapped = True  # type: ignore[attr-defined]
        setattr(candidate, "__init__", init_wrapper)
        patched_any = True

    if patched_any:
        _RUNNER_NVML_PATCHED.add(module_name)
        _debug(f"patched runner module {module_name} for resource sampling")


def _patch_behavior_nvml_module(module: ModuleType) -> None:
    global _BEHAVIOR_NVML_PATCHED
    if _BEHAVIOR_NVML_PATCHED:
        return

    original_worker = getattr(module, "_behavior_env_worker", None)
    if original_worker is None or getattr(original_worker, "_rlinf_nvml_wrapped", False):
        _BEHAVIOR_NVML_PATCHED = True
        return

    @functools.wraps(original_worker)
    def worker_wrapper(cfg, conn, num_envs, *args, **kwargs):
        _start_local_resource_samplers(
            component="behavior_subproc",
            rank=_rank_from_context(tuple()),
            extra={
                "num_envs": num_envs,
                "module": getattr(module, "__name__", None),
            },
        )
        return original_worker(cfg, conn, num_envs, *args, **kwargs)

    worker_wrapper._rlinf_nvml_wrapped = True  # type: ignore[attr-defined]
    setattr(module, "_behavior_env_worker", worker_wrapper)
    _BEHAVIOR_NVML_PATCHED = True
    _debug("patched behavior subprocess worker for resource sampling")


def _patch_resource_worker_init_module(module: ModuleType) -> None:
    module_name = getattr(module, "__name__", "")
    patched_any = False
    for attr_name in dir(module):
        candidate = getattr(module, attr_name, None)
        if not inspect.isclass(candidate):
            continue
        class_name = getattr(candidate, "__name__", "")
        if not (
            class_name.endswith("Worker")
            or class_name.endswith("Actor")
            or class_name.endswith("Critic")
        ):
            continue
        key = (module_name, class_name)
        if key in _RESOURCE_WORKER_INIT_PATCHED:
            continue
        original_init = getattr(candidate, "__init__", None)
        if original_init is None or getattr(original_init, "_rlinf_resource_wrapped", False):
            _RESOURCE_WORKER_INIT_PATCHED.add(key)
            continue

        @functools.wraps(original_init)
        def init_wrapper(self, *args, __original_init=original_init, **kwargs):
            result = __original_init(self, *args, **kwargs)
            # Set RLINF_LOG_DIR right at worker construction (self.cfg is now set)
            # so init-time PatchSpec events whose `self` isn't this Worker (e.g.
            # ManiskillOffloadEnv.onload/offload fired during _init_env, before
            # any timer event) still resolve the run timeline dir.
            _set_log_dir_env_from_cfg(getattr(self, "cfg", None))
            _ensure_resource_sampling_for_object(self)
            return result

        init_wrapper._rlinf_resource_wrapped = True  # type: ignore[attr-defined]
        setattr(candidate, "__init__", init_wrapper)
        _RESOURCE_WORKER_INIT_PATCHED.add(key)
        patched_any = True

    if patched_any:
        _debug(f"patched resource worker init module {module_name}")


def _patch_worker_timer_module(module: ModuleType) -> None:
    global _WORKER_TIMER_PATCHED
    if _WORKER_TIMER_PATCHED:
        return

    worker_cls = getattr(module, "Worker", None)
    if worker_cls is None:
        _debug("Worker class not found for timer patch")
        return

    original_timer = getattr(worker_cls, "timer")
    if getattr(original_timer, "_rlinf_timeline_timer_wrapped", False):
        _WORKER_TIMER_PATCHED = True
        return

    def timer(tag: str | None = None):
        def decorator(func):
            timer_tag = tag or func.__name__
            original_decorator = original_timer(timer_tag)
            timed_func = original_decorator(func)
            if timer_tag in _worker_timer_exclude_tags():
                return timed_func

            if inspect.iscoroutinefunction(timed_func):

                @functools.wraps(timed_func)
                async def async_wrapper(self, *args, **kwargs):
                    if _DEFERRED_SPECS:
                        _retry_deferred()
                    if "RLINF_LOG_DIR" not in os.environ:
                        _set_log_dir_env_from_cfg(getattr(self, "cfg", None))
                    _ensure_resource_sampling_for_object(self)
                    call_index = _next_call_index(self, timer_tag)
                    t0 = time.time()
                    error = None
                    try:
                        return await timed_func(self, *args, **kwargs)
                    except Exception as exc:
                        error = type(exc).__name__
                        raise
                    finally:
                        extra = {
                            "module": func.__module__,
                            "qualname": getattr(func, "__qualname__", func.__name__),
                            "worker_timer": timer_tag,
                        }
                        extra.update(
                            _call_metadata(
                                obj=self,
                                func=func,
                                timer_tag=timer_tag,
                                args=args,
                                kwargs=kwargs,
                                call_index=call_index,
                            )
                        )
                        if error:
                            extra["exception"] = error
                        append_event(
                            component=_component_from_context((self,), "worker"),
                            rank=_rank_from_context((self,)),
                            tag=timer_tag,
                            t0=t0,
                            t1=time.time(),
                            global_step=_step_from_context((self,)),
                            extra=extra,
                            timeline_dir=_timeline_dir_from_context((self,)),
                        )

                async_wrapper._rlinf_timeline_wrapped = True  # type: ignore[attr-defined]
                return async_wrapper

            @functools.wraps(timed_func)
            def wrapper(self, *args, **kwargs):
                if _DEFERRED_SPECS:
                    _retry_deferred()
                if "RLINF_LOG_DIR" not in os.environ:
                    _set_log_dir_env_from_cfg(getattr(self, "cfg", None))
                _ensure_resource_sampling_for_object(self)
                call_index = _next_call_index(self, timer_tag)
                t0 = time.time()
                error = None
                try:
                    return timed_func(self, *args, **kwargs)
                except Exception as exc:
                    error = type(exc).__name__
                    raise
                finally:
                    extra = {
                        "module": func.__module__,
                        "qualname": getattr(func, "__qualname__", func.__name__),
                        "worker_timer": timer_tag,
                    }
                    extra.update(
                        _call_metadata(
                            obj=self,
                            func=func,
                            timer_tag=timer_tag,
                            args=args,
                            kwargs=kwargs,
                            call_index=call_index,
                        )
                    )
                    if error:
                        extra["exception"] = error
                    append_event(
                        component=_component_from_context((self,), "worker"),
                        rank=_rank_from_context((self,)),
                        tag=timer_tag,
                        t0=t0,
                        t1=time.time(),
                        global_step=_step_from_context((self,)),
                        extra=extra,
                        timeline_dir=_timeline_dir_from_context((self,)),
                    )

            wrapper._rlinf_timeline_wrapped = True  # type: ignore[attr-defined]
            return wrapper

        return decorator

    timer._rlinf_timeline_timer_wrapped = True  # type: ignore[attr-defined]
    worker_cls.timer = staticmethod(timer)
    _WORKER_TIMER_PATCHED = True
    _debug("patched Worker.timer for timeline events")


def _install_worker_timer_from_env() -> None:
    if os.environ.get("RLINF_TIMELINE_WORKER_TIMER", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
        "y",
    }:
        return

    module_name = _WorkerTimerFinder.MODULE
    module = sys.modules.get(module_name)
    if module is not None:
        _patch_worker_timer_module(module)
    else:
        sys.meta_path.insert(0, _WorkerTimerFinder())
        _debug("installed Worker.timer import hook")


def _install_actor_training_from_env() -> None:
    if os.environ.get("RLINF_TIMELINE_ACTOR_TRAINING", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
        "y",
    }:
        return

    module_name = _ActorTrainingFinder.MODULE
    module = sys.modules.get(module_name)
    if module is not None:
        _patch_actor_training_module(module)
    else:
        sys.meta_path.insert(0, _ActorTrainingFinder())
        _debug("installed actor training import hook")


def _install_nvml_from_env() -> None:
    if not _resource_sampling_enabled():
        return

    worker_module = _WorkerNVMLFinder.MODULE
    module = sys.modules.get(worker_module)
    if module is not None:
        _patch_worker_nvml_module(module)
    else:
        sys.meta_path.insert(0, _WorkerNVMLFinder())
        _debug("installed Worker resource import hook")

    behavior_module = _BehaviorNVMLFinder.MODULE
    module = sys.modules.get(behavior_module)
    if module is not None:
        _patch_behavior_nvml_module(module)
    else:
        sys.meta_path.insert(0, _BehaviorNVMLFinder())
        _debug("installed behavior resource import hook")

    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith(_RunnerNVMLFinder.MODULE_PREFIX):
            continue
        _patch_runner_nvml_module(module)
    sys.meta_path.insert(0, _RunnerNVMLFinder())
    _debug("installed runner resource import hook")

    for module_name in _RESOURCE_WORKER_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            _patch_resource_worker_init_module(module)
    sys.meta_path.insert(0, _ResourceWorkerInitFinder())
    _debug("installed worker class resource import hook")


def install_from_env() -> None:
    global _INSTALLED
    if _INSTALLED or not (is_enabled() or _resource_sampling_enabled()):
        return

    _install_nvml_from_env()
    _install_worker_timer_from_env()
    _install_actor_training_from_env()

    specs = _read_patch_specs()
    if not specs:
        _debug("enabled, but no RLINF_TIMELINE_PATCHES or RLINF_TIMELINE_PATCH_FILE")
        _INSTALLED = True
        return

    by_module: dict[str, list[PatchSpec]] = {}
    for spec in specs:
        by_module.setdefault(spec.module, []).append(spec)

    for module_name, module_specs in by_module.items():
        module = sys.modules.get(module_name)
        if module is not None:
            _patch_module(module, module_specs)

    sys.meta_path.insert(0, _TimelineFinder(by_module))
    # Catch-all retry hook: fires on every later import so specs that failed at
    # import-hook time (circular-import targets like maniskill_offload_env) get
    # retried once their class finishes defining. No-op when nothing is deferred.
    # Inserted ahead of PathFinder so it actually sees normal file imports.
    try:
        _pf_idx = sys.meta_path.index(importlib.machinery.PathFinder)
    except ValueError:
        _pf_idx = len(sys.meta_path)
    sys.meta_path.insert(_pf_idx, _DeferredRetryFinder())
    _debug(f"installed import hook for {len(by_module)} modules")
    _INSTALLED = True


def patch_now(spec_text: str) -> None:
    """Patch already importable targets from a semicolon-separated spec string."""

    specs = []
    for raw in spec_text.replace("\n", ";").split(";"):
        spec = _split_spec(raw)
        if spec is not None:
            specs.append(spec)

    for spec in specs:
        module = importlib.import_module(spec.module)
        _patch_module(module, [spec])
