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

"""Latency + peak-VRAM measurement for a single atomic op.

``timed_vram`` runs ``fn`` a few warm-up times, then times ``repeats`` CUDA-
synchronized invocations and records the peak allocated / reserved memory across
them. It measures whatever ``fn`` does on the *current* CUDA device, so it needs
no knowledge of the worker or the op.

For CPU-heavy ops (e.g. the env simulator step) it can *optionally* sample host
RAM, process CPU utilization and GPU SM utilization across the timed region via a
lightweight background thread (``sample_host`` / ``sample_gpu``). These are opt-in
so the fast GPU ops (rollout/actor) pay no sampler overhead by default.
"""

from __future__ import annotations

import threading
from time import perf_counter, sleep
from typing import Any, Callable, Iterable

import torch


class _ResourceSampler:
    """Poll host RAM / process CPU% / GPU util on a daemon thread.

    Only samples while ``start()``..``stop()`` is active (i.e. the timed region),
    at ``interval_s`` cadence. Any backend that fails to import is silently
    disabled, so this never adds a hard dependency.
    """

    def __init__(
        self,
        *,
        sample_host: bool,
        sample_gpu: bool,
        device_index: int | None,
        interval_s: float = 0.005,
    ):
        self._interval = interval_s
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._proc = None
        self._rss_samples: list[int] = []
        self._cpu_samples: list[float] = []
        self._rss_start: int | None = None
        if sample_host:
            try:
                import psutil

                self._proc = psutil.Process()
                # Prime cpu_percent so the first real sample is a delta, not 0.
                self._proc.cpu_percent(None)
            except Exception:
                self._proc = None

        self._gpu_handle = None
        self._gpu_samples: list[float] = []
        if sample_gpu and device_index is not None:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._pynvml = pynvml
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception:
                self._gpu_handle = None

    @property
    def active(self) -> bool:
        return self._proc is not None or self._gpu_handle is not None

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._proc is not None:
                try:
                    self._rss_samples.append(self._proc.memory_info().rss)
                    self._cpu_samples.append(self._proc.cpu_percent(None))
                except Exception:
                    pass
            if self._gpu_handle is not None:
                try:
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    self._gpu_samples.append(float(util.gpu))
                except Exception:
                    pass
            sleep(self._interval)

    def start(self) -> None:
        if not self.active:
            return
        if self._proc is not None:
            try:
                self._rss_start = self._proc.memory_info().rss
            except Exception:
                self._rss_start = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._thread = None

        out: dict[str, Any] = {}
        if self._rss_samples:
            peak = max(self._rss_samples)
            out["host_rss_peak_MB"] = round(peak / 1e6, 2)
            if self._rss_start is not None:
                out["host_rss_delta_MB"] = round((peak - self._rss_start) / 1e6, 2)
        if self._cpu_samples:
            out["cpu_percent_peak"] = round(max(self._cpu_samples), 1)
            out["cpu_percent_mean"] = round(
                sum(self._cpu_samples) / len(self._cpu_samples), 1
            )
        if self._gpu_samples:
            out["gpu_util_peak"] = round(max(self._gpu_samples), 1)
            out["gpu_util_mean"] = round(
                sum(self._gpu_samples) / len(self._gpu_samples), 1
            )
        return out


def timed_once(
    fn: Callable[[], Any],
    *,
    sample_host: bool = False,
    sample_gpu: bool = False,
    device_index: int | None = None,
) -> dict[str, Any]:
    """Time a single, non-repeatable op (e.g. weight onload/offload, env build).

    Unlike ``timed_vram`` this runs ``fn`` **exactly once** -- no warmup, no
    repeats -- because onload/offload and env construction mutate state and
    cannot be replayed in place. Returns ``{"ms": ...}`` plus, when CUDA is
    available, ``peak_alloc_MB``/``peak_reserved_MB`` measured across the call,
    and (when requested) the same ``host_rss_*`` / ``cpu_percent_*`` /
    ``gpu_util_*`` samples ``timed_vram`` collects. Exceptions raised by ``fn``
    propagate (the sampler is still stopped).
    """
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    sampler = _ResourceSampler(
        sample_host=sample_host,
        sample_gpu=sample_gpu,
        device_index=device_index,
    )
    sampler.start()

    t0 = perf_counter()
    try:
        fn()
        if cuda:
            torch.cuda.synchronize()
        ms = (perf_counter() - t0) * 1000.0
    finally:
        resource_stats = sampler.stop()

    record: dict[str, Any] = {"ms": round(ms, 4)}
    if cuda:
        record["peak_alloc_MB"] = round(torch.cuda.max_memory_allocated() / 1e6, 2)
        record["peak_reserved_MB"] = round(torch.cuda.max_memory_reserved() / 1e6, 2)
    record.update(resource_stats)
    return record


def progress_iter(items: Iterable, *, desc: str, enabled: bool = True):
    """Wrap ``items`` in a tqdm progress bar when possible.

    Returns a tqdm instance (so callers may ``set_postfix``) when tqdm is
    importable and ``enabled`` is True; otherwise returns a plain list so the
    loop still works without the dependency. tqdm output goes to the worker's
    stderr, i.e. its per-worker log stream.
    """
    materialized = list(items)
    if not enabled:
        return materialized
    try:
        from tqdm.auto import tqdm
    except Exception:
        return materialized
    return tqdm(materialized, desc=desc, leave=True, dynamic_ncols=True)



def _statistics(times_ms: list[float]) -> dict[str, float]:
    n = len(times_ms)
    mean = sum(times_ms) / n
    var = sum((t - mean) ** 2 for t in times_ms) / n if n > 1 else 0.0
    return {
        "ms_mean": round(mean, 4),
        "ms_std": round(var**0.5, 4),
        "ms_min": round(min(times_ms), 4),
        "ms_max": round(max(times_ms), 4),
    }


def timed_vram(
    fn: Callable[[], Any],
    *,
    warmup: int = 2,
    repeats: int = 5,
    sample_host: bool = False,
    sample_gpu: bool = False,
    device_index: int | None = None,
) -> dict[str, Any]:
    """Time ``fn`` and record peak VRAM on the current CUDA device.

    Returns a dict with ``ms_mean/std/min/max``, ``peak_alloc_MB``,
    ``peak_reserved_MB``, ``warmup`` and ``repeats``. Exceptions raised by ``fn``
    (e.g. CUDA OOM) propagate to the caller.

    When ``sample_host``/``sample_gpu`` are set, a background thread samples host
    RAM / process CPU% / GPU util across the timed region and adds
    ``host_rss_peak_MB``, ``host_rss_delta_MB``, ``cpu_percent_peak/mean`` and
    ``gpu_util_peak/mean`` (whichever backends are importable). ``device_index``
    is the CUDA device ordinal to read GPU util for.
    """
    cuda = torch.cuda.is_available()

    for _ in range(max(0, warmup)):
        fn()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    sampler = _ResourceSampler(
        sample_host=sample_host,
        sample_gpu=sample_gpu,
        device_index=device_index,
    )
    sampler.start()

    times_ms: list[float] = []
    try:
        for _ in range(max(1, repeats)):
            if cuda:
                torch.cuda.synchronize()
            t0 = perf_counter()
            fn()
            if cuda:
                torch.cuda.synchronize()
            times_ms.append((perf_counter() - t0) * 1000.0)
    finally:
        resource_stats = sampler.stop()

    record: dict[str, Any] = _statistics(times_ms)
    record["warmup"] = warmup
    record["repeats"] = repeats
    if cuda:
        record["peak_alloc_MB"] = round(torch.cuda.max_memory_allocated() / 1e6, 2)
        record["peak_reserved_MB"] = round(torch.cuda.max_memory_reserved() / 1e6, 2)
    record.update(resource_stats)
    return record
