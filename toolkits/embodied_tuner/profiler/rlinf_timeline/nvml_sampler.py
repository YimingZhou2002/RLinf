"""Optional per-process NVML sampler for RLinf sidecar profiling."""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from typing import Any, Callable

try:  # Prefer Ray's vendored binding when available inside RLinf envs.
    import ray._private.thirdparty.pynvml as _pynvml  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - fallback depends on local env
    try:
        import pynvml as _pynvml  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        _pynvml = None


_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_LOCK = threading.Lock()
_SAMPLER: "_LocalProcessNVMLSampler | None" = None


def is_nvml_enabled() -> bool:
    return os.environ.get("RLINF_NVML", "").lower() in _TRUE_VALUES


def default_nvml_dir() -> str | None:
    env_dir = os.environ.get("RLINF_NVML_DIR")
    if env_dir and env_dir.lower() != "auto":
        return env_dir

    log_dir = os.environ.get("RLINF_LOG_DIR")
    if log_dir:
        return os.path.join(log_dir, "nvml")

    return None


def _sample_interval_seconds() -> float:
    raw = os.environ.get("RLINF_NVML_INTERVAL", "").strip()
    if raw:
        try:
            return max(float(raw), 0.01)
        except ValueError:
            pass

    raw_ms = os.environ.get("RLINF_NVML_INTERVAL_MS", "").strip()
    if raw_ms:
        try:
            return max(float(raw_ms) / 1000.0, 0.01)
        except ValueError:
            pass

    return 0.2


def _debug(msg: str) -> None:
    if os.environ.get("RLINF_TIMELINE_DEBUG", "").lower() in _TRUE_VALUES:
        print(f"[rlinf_timeline] {msg}", file=os.sys.stderr)


def _current_global_step(getter: Callable[[], Any] | None) -> int | None:
    if getter is None:
        return None
    try:
        value = getter()
    except Exception:
        return None
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _torch_memory_snapshot() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {}

    if not torch.cuda.is_available():
        return {}

    try:
        device_idx = torch.cuda.current_device()
    except Exception:
        device_idx = None

    try:
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        max_allocated = int(torch.cuda.max_memory_allocated())
        max_reserved = int(torch.cuda.max_memory_reserved())
    except Exception:
        return {}

    return {
        "torch_device_index": device_idx,
        "torch_allocated_bytes": allocated,
        "torch_reserved_bytes": reserved,
        "torch_max_allocated_bytes": max_allocated,
        "torch_max_reserved_bytes": max_reserved,
    }


class _LocalProcessNVMLSampler:
    def __init__(
        self,
        *,
        component: str,
        rank: int,
        global_step_getter: Callable[[], Any] | None = None,
        extra: dict[str, Any] | None = None,
        nvml_dir: str | None = None,
    ) -> None:
        self.component = component
        self.rank = rank
        self.global_step_getter = global_step_getter
        self.extra = dict(extra or {})
        self.pid = os.getpid()
        self.interval_s = _sample_interval_seconds()
        self.nvml_dir = nvml_dir or default_nvml_dir()
        self.path = os.path.join(
            self.nvml_dir, f"{self.component}_rank{self.rank}_pid{self.pid}.jsonl"
        )
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"rlinf-nvml-{self.component}-rank{self.rank}",
            daemon=True,
        )
        self._started = False
        self.backend = "pynvml" if _pynvml is not None else "none"

    def start(self) -> None:
        if self._started:
            return
        os.makedirs(self.nvml_dir, exist_ok=True)
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 4))

    def _run(self) -> None:
        if self.backend == "none":
            _debug("NVML sampler requested but pynvml is not available")
            return

        try:
            _pynvml.nvmlInit()
        except Exception as exc:  # pragma: no cover - depends on host NVML
            _debug(f"NVML init failed: {exc}")
            return

        try:
            while not self._stop_event.is_set():
                sample = self._build_sample()
                if sample is not None:
                    self._append_sample(sample)
                self._stop_event.wait(self.interval_s)
        finally:
            try:
                _pynvml.nvmlShutdown()
            except Exception:
                pass

    def _iter_device_samples_pynvml(self) -> list[dict[str, Any]]:
        device_count = int(_pynvml.nvmlDeviceGetCount())
        samples: list[dict[str, Any]] = []
        for device_index in range(device_count):
            handle = _pynvml.nvmlDeviceGetHandleByIndex(device_index)
            used_bytes = 0
            # Different NVML bindings expose both *_v2 and legacy process
            # getters. They can report the same PID, so keep the maximum
            # reported memory instead of summing duplicate entries.
            for getter_name in (
                "nvmlDeviceGetComputeRunningProcesses_v2",
                "nvmlDeviceGetComputeRunningProcesses",
                "nvmlDeviceGetGraphicsRunningProcesses_v2",
                "nvmlDeviceGetGraphicsRunningProcesses",
            ):
                getter = getattr(_pynvml, getter_name, None)
                if getter is None:
                    continue
                try:
                    processes = getter(handle)
                except Exception:
                    continue
                for proc in processes:
                    if int(getattr(proc, "pid", -1)) != self.pid:
                        continue
                    mem = int(getattr(proc, "usedGpuMemory", 0))
                    if mem > 0:
                        used_bytes = max(used_bytes, mem)
            if used_bytes <= 0:
                continue
            samples.append(
                {
                    "gpu_index": device_index,
                    "used_memory_bytes": used_bytes,
                    "used_memory_gib": used_bytes / (2**30),
                }
            )
        return samples

    def _build_sample(self) -> dict[str, Any] | None:
        device_samples = self._iter_device_samples_pynvml()
        if not device_samples:
            return None

        total_used_bytes = sum(s["used_memory_bytes"] for s in device_samples)
        sample: dict[str, Any] = {
            "ts": time.time(),
            "pid": self.pid,
            "component": self.component,
            "rank": self.rank,
            "global_step": _current_global_step(self.global_step_getter),
            "sample_interval_s": self.interval_s,
            "backend": self.backend,
            "devices": device_samples,
            "nvml_total_used_bytes": total_used_bytes,
            "nvml_total_used_gib": total_used_bytes / (2**30),
        }
        if self.extra:
            sample.update(self.extra)

        torch_snapshot = _torch_memory_snapshot()
        if torch_snapshot:
            sample.update(torch_snapshot)

        return sample

    def _append_sample(self, sample: dict[str, Any]) -> None:
        line = json.dumps(sample, ensure_ascii=False) + "\n"
        with _LOCK:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)


def start_local_process_sampler(
    *,
    component: str,
    rank: int,
    global_step_getter: Callable[[], Any] | None = None,
    extra: dict[str, Any] | None = None,
    nvml_dir: str | None = None,
) -> None:
    global _SAMPLER
    if not is_nvml_enabled():
        return
    if _SAMPLER is not None:
        return
    resolved_dir = nvml_dir or default_nvml_dir()
    if resolved_dir is None:
        _debug(
            "NVML sampler skipped because neither RLINF_NVML_DIR nor RLINF_LOG_DIR is set"
        )
        return

    sampler = _LocalProcessNVMLSampler(
        component=component,
        rank=rank,
        global_step_getter=global_step_getter,
        extra=extra,
        nvml_dir=resolved_dir,
    )
    _SAMPLER = sampler
    sampler.start()
    _debug(
        "started NVML sampler "
        f"component={component} rank={rank} pid={os.getpid()} "
        f"backend={sampler.backend} path={sampler.path}"
    )
    atexit.register(sampler.stop)
