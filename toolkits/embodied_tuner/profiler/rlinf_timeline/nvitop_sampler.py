"""Optional per-process nvitop sampler for RLinf sidecar profiling."""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
import time
from typing import Any, Callable

try:
    import nvitop as _nvitop  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - depends on runtime env
    _nvitop = None


_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_LOCK = threading.Lock()
_SAMPLER: "_LocalProcessNvitopSampler | None" = None
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def is_nvitop_enabled() -> bool:
    return os.environ.get("RLINF_NVITOP", "").lower() in _TRUE_VALUES


def default_nvitop_dir() -> str | None:
    env_dir = os.environ.get("RLINF_NVITOP_DIR")
    if env_dir and env_dir.lower() != "auto":
        return env_dir

    log_dir = os.environ.get("RLINF_LOG_DIR")
    if log_dir:
        return os.path.join(log_dir, "nvitop")

    return None


def _sample_interval_seconds() -> float:
    raw = os.environ.get("RLINF_NVITOP_INTERVAL", "").strip()
    if raw:
        try:
            return max(float(raw), 0.05)
        except ValueError:
            pass

    raw_ms = os.environ.get("RLINF_NVITOP_INTERVAL_MS", "").strip()
    if raw_ms:
        try:
            return max(float(raw_ms) / 1000.0, 0.05)
        except ValueError:
            pass

    raw_nvml = os.environ.get("RLINF_NVML_INTERVAL", "").strip()
    if raw_nvml:
        try:
            return max(float(raw_nvml), 0.05)
        except ValueError:
            pass

    return 0.5


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


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[0].rstrip(":")
                try:
                    values[key] = int(parts[1]) * 1024
                except ValueError:
                    continue
    except OSError:
        return {}
    return values


def _read_system_cpu_ticks() -> tuple[int, int] | None:
    try:
        with open("/proc/stat", encoding="utf-8") as f:
            line = f.readline()
    except OSError:
        return None
    parts = line.split()
    if not parts or parts[0] != "cpu":
        return None
    try:
        ticks = [int(value) for value in parts[1:]]
    except ValueError:
        return None
    total = sum(ticks)
    idle = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)
    return total, idle


def _read_process_cpu_ticks(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return None
    rparen = content.rfind(")")
    if rparen < 0:
        return None
    fields = content[rparen + 2 :].split()
    try:
        utime = int(fields[11])
        stime = int(fields[12])
        cutime = int(fields[13])
        cstime = int(fields[14])
    except (IndexError, ValueError):
        return None
    return utime + stime + cutime + cstime


def _read_process_rss_bytes(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("VmRSS:"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except (OSError, ValueError):
        pass

    try:
        with open(f"/proc/{pid}/statm", encoding="utf-8") as f:
            parts = f.read().split()
        return int(parts[1]) * _PAGE_SIZE
    except (OSError, IndexError, ValueError):
        return None


def _read_process_threads(pid: int) -> int | None:
    try:
        return len(os.listdir(f"/proc/{pid}/task"))
    except OSError:
        return None


def _maybe_call(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        attr = getattr(obj, name, None)
        if attr is None:
            continue
        try:
            return attr() if callable(attr) else attr
        except Exception:
            continue
    return None


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    number = float(match.group(0))
    unit = text[match.end() :].strip().lower()
    if unit.startswith("kib") or unit.startswith("kb"):
        number *= 2**10
    elif unit.startswith("mib") or unit.startswith("mb"):
        number *= 2**20
    elif unit.startswith("gib") or unit.startswith("gb"):
        number *= 2**30
    elif unit.startswith("tib") or unit.startswith("tb"):
        number *= 2**40
    return int(number)


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    if match is None:
        return None
    return float(match.group(0))


class _LocalProcessNvitopSampler:
    def __init__(
        self,
        *,
        component: str,
        rank: int,
        global_step_getter: Callable[[], Any] | None = None,
        extra: dict[str, Any] | None = None,
        nvitop_dir: str | None = None,
    ) -> None:
        self.component = component
        self.rank = rank
        self.global_step_getter = global_step_getter
        self.extra = dict(extra or {})
        self.pid = os.getpid()
        self.interval_s = _sample_interval_seconds()
        self.nvitop_dir = nvitop_dir or default_nvitop_dir()
        self.path = os.path.join(
            self.nvitop_dir,
            f"{self.component}_rank{self.rank}_pid{self.pid}.jsonl",
        )
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"rlinf-nvitop-{self.component}-rank{self.rank}",
            daemon=True,
        )
        self._started = False
        self._last_system_cpu: tuple[int, int] | None = None
        self._last_process_cpu: int | None = None

    def start(self) -> None:
        if self._started:
            return
        os.makedirs(self.nvitop_dir, exist_ok=True)
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 4))

    def _run(self) -> None:
        if _nvitop is None:
            _debug("nvitop sampler requested but nvitop is not available")
            return

        while not self._stop_event.is_set():
            sample = self._build_sample()
            if sample is not None:
                self._append_sample(sample)
            self._stop_event.wait(self.interval_s)

    def _cpu_snapshot(self) -> dict[str, Any]:
        sample: dict[str, Any] = {}
        meminfo = _read_meminfo()
        if meminfo:
            total = meminfo.get("MemTotal")
            available = meminfo.get("MemAvailable")
            if total is not None:
                sample["system_memory_total_bytes"] = total
                sample["system_memory_total_gib"] = total / (2**30)
            if available is not None:
                sample["system_memory_available_bytes"] = available
                sample["system_memory_available_gib"] = available / (2**30)
            if total is not None and available is not None:
                used = total - available
                sample["system_memory_used_bytes"] = used
                sample["system_memory_used_gib"] = used / (2**30)
                sample["system_memory_used_percent"] = used / total * 100.0

        rss = _read_process_rss_bytes(self.pid)
        if rss is not None:
            sample["process_rss_bytes"] = rss
            sample["process_rss_gib"] = rss / (2**30)
        threads = _read_process_threads(self.pid)
        if threads is not None:
            sample["process_threads"] = threads

        system_cpu = _read_system_cpu_ticks()
        process_cpu = _read_process_cpu_ticks(self.pid)
        if system_cpu is not None and self._last_system_cpu is not None:
            total_delta = system_cpu[0] - self._last_system_cpu[0]
            idle_delta = system_cpu[1] - self._last_system_cpu[1]
            if total_delta > 0:
                sample["system_cpu_percent"] = (
                    (total_delta - idle_delta) / total_delta * 100.0
                )
        if (
            system_cpu is not None
            and process_cpu is not None
            and self._last_system_cpu is not None
            and self._last_process_cpu is not None
        ):
            total_delta = system_cpu[0] - self._last_system_cpu[0]
            process_delta = process_cpu - self._last_process_cpu
            if total_delta > 0:
                sample["process_cpu_percent"] = (
                    process_delta / total_delta * (os.cpu_count() or 1) * 100.0
                )

        self._last_system_cpu = system_cpu
        self._last_process_cpu = process_cpu
        return sample

    def _gpu_snapshot(self) -> list[dict[str, Any]]:
        device_cls = getattr(_nvitop, "Device", None)
        if device_cls is None:
            return []
        all_devices = getattr(device_cls, "all", None)
        if not callable(all_devices):
            return []

        try:
            devices = list(all_devices())
        except Exception:
            return []

        samples: list[dict[str, Any]] = []
        for device in devices:
            gpu_index = _maybe_call(device, ("index", "physical_index"))
            entry: dict[str, Any] = {
                "gpu_index": _to_int(gpu_index),
            }
            for key, names in (
                ("name", ("name",)),
                ("uuid", ("uuid",)),
            ):
                value = _maybe_call(device, names)
                if value is not None:
                    entry[key] = str(value)
            for key, names in (
                ("gpu_util_percent", ("gpu_utilization", "utilization_gpu")),
                ("memory_util_percent", ("memory_utilization", "utilization_memory")),
                ("temperature_c", ("temperature",)),
                ("power_usage_w", ("power_usage",)),
            ):
                value = _to_float(_maybe_call(device, names))
                if value is not None:
                    entry[key] = value
            for key, names in (
                ("memory_total_bytes", ("memory_total", "total_memory")),
                ("memory_used_bytes", ("memory_used", "used_memory")),
                ("memory_free_bytes", ("memory_free", "free_memory")),
            ):
                value = _to_int(_maybe_call(device, names))
                if value is not None:
                    entry[key] = value
                    entry[key.replace("_bytes", "_gib")] = value / (2**30)

            process_entries = []
            processes_fn = getattr(device, "processes", None)
            if callable(processes_fn):
                try:
                    processes = processes_fn()
                    if isinstance(processes, dict):
                        processes = processes.values()
                except Exception:
                    processes = []
                for proc in processes:
                    pid = _to_int(_maybe_call(proc, ("pid",)))
                    if pid != self.pid:
                        continue
                    proc_entry: dict[str, Any] = {"pid": self.pid}
                    for key, names in (
                        ("gpu_memory_bytes", ("gpu_memory", "gpu_memory_usage")),
                        ("gpu_sm_util_percent", ("gpu_sm_utilization",)),
                        ("gpu_memory_util_percent", ("gpu_memory_utilization",)),
                    ):
                        value = _maybe_call(proc, names)
                        parsed = _to_int(value) if key.endswith("_bytes") else _to_float(value)
                        if parsed is not None:
                            proc_entry[key] = parsed
                            if key.endswith("_bytes"):
                                proc_entry[key.replace("_bytes", "_gib")] = parsed / (2**30)
                    process_entries.append(proc_entry)
            if process_entries:
                entry["processes"] = process_entries
            samples.append({k: v for k, v in entry.items() if v is not None})
        return samples

    def _build_sample(self) -> dict[str, Any] | None:
        sample: dict[str, Any] = {
            "ts": time.time(),
            "pid": self.pid,
            "component": self.component,
            "rank": self.rank,
            "global_step": _current_global_step(self.global_step_getter),
            "sample_interval_s": self.interval_s,
            "backend": "nvitop",
        }
        if self.extra:
            sample.update(self.extra)
        sample.update(self._cpu_snapshot())
        gpus = self._gpu_snapshot()
        if gpus:
            sample["gpus"] = gpus
        return sample

    def _append_sample(self, sample: dict[str, Any]) -> None:
        line = json.dumps(sample, ensure_ascii=False) + "\n"
        with _LOCK:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)


def start_local_process_nvitop_sampler(
    *,
    component: str,
    rank: int,
    global_step_getter: Callable[[], Any] | None = None,
    extra: dict[str, Any] | None = None,
    nvitop_dir: str | None = None,
) -> None:
    global _SAMPLER
    if not is_nvitop_enabled():
        return
    if _SAMPLER is not None:
        return
    resolved_dir = nvitop_dir or default_nvitop_dir()
    if resolved_dir is None:
        _debug(
            "nvitop sampler skipped because neither RLINF_NVITOP_DIR nor RLINF_LOG_DIR is set"
        )
        return

    sampler = _LocalProcessNvitopSampler(
        component=component,
        rank=rank,
        global_step_getter=global_step_getter,
        extra=extra,
        nvitop_dir=resolved_dir,
    )
    _SAMPLER = sampler
    sampler.start()
    _debug(
        "started nvitop sampler "
        f"component={component} rank={rank} pid={os.getpid()} path={sampler.path}"
    )
    atexit.register(sampler.stop)
