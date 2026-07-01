"""JSONL timeline writer used by the sidecar RLinf instrumentation."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_LOCK = threading.Lock()

_TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def is_enabled() -> bool:
    return os.environ.get("RLINF_TIMELINE", "").lower() in _TRUE_VALUES


def default_timeline_dir() -> str:
    env_dir = os.environ.get("RLINF_TIMELINE_DIR")
    if env_dir and env_dir.lower() != "auto":
        return env_dir

    log_dir = os.environ.get("RLINF_LOG_DIR")
    if log_dir:
        return os.path.join(log_dir, "timeline")

    return os.path.abspath("timeline")


def append_event(
    *,
    component: str,
    rank: int,
    tag: str,
    t0: float,
    t1: float,
    global_step: int | None = None,
    extra: dict[str, Any] | None = None,
    timeline_dir: str | None = None,
) -> None:
    if not is_enabled():
        return

    base = timeline_dir or default_timeline_dir()
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"{component}_rank{rank}.jsonl")

    rec: dict[str, Any] = {
        "t0": t0,
        "t1": t1,
        "component": component,
        "rank": rank,
        "tag": tag,
        "global_step": global_step,
        "pid": os.getpid(),
    }
    if extra:
        rec.update(extra)

    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def append_timeline_event(
    cfg: Any,
    *,
    component: str,
    rank: int,
    tag: str,
    t0: float,
    t1: float,
    global_step: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Compatibility helper for explicit cfg-based call sites."""

    runner = getattr(cfg, "runner", None)
    if runner is None or not runner.get("timeline_trace", False):
        return

    timeline_dir = runner.get("timeline_dir")
    if not timeline_dir:
        logger = getattr(runner, "logger", None)
        log_path = getattr(logger, "log_path", None)
        if log_path:
            timeline_dir = os.path.join(log_path, "timeline")

    old_value = os.environ.get("RLINF_TIMELINE")
    os.environ["RLINF_TIMELINE"] = "1"
    try:
        append_event(
            component=component,
            rank=rank,
            tag=tag,
            t0=t0,
            t1=t1,
            global_step=global_step,
            extra=extra,
            timeline_dir=timeline_dir,
        )
    finally:
        if old_value is None:
            os.environ.pop("RLINF_TIMELINE", None)
        else:
            os.environ["RLINF_TIMELINE"] = old_value


class record:
    """Context manager for manual sidecar timings."""

    def __init__(
        self,
        component: str,
        tag: str,
        *,
        rank: int = 0,
        global_step: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.component = component
        self.tag = tag
        self.rank = rank
        self.global_step = global_step
        self.extra = extra
        self.t0 = 0.0

    def __enter__(self) -> "record":
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        extra = dict(self.extra or {})
        if exc_type is not None:
            extra["exception"] = getattr(exc_type, "__name__", str(exc_type))
        append_event(
            component=self.component,
            rank=self.rank,
            tag=self.tag,
            t0=self.t0,
            t1=time.time(),
            global_step=self.global_step,
            extra=extra,
        )
