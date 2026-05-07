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

"""
Fine-grained Timeline Profiler for distributed RL training.

This module provides tools to record detailed timing information across
multiple workers (env, rollout, actor) and visualize the execution timeline
as a pipeline bubble chart.
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class TimelineEvent:
    """A single event in the timeline."""

    name: str  # 事件名称
    start_time: float  # 开始时间戳 (相对时间, 秒)
    end_time: float  # 结束时间戳 (相对时间, 秒)
    worker_type: str  # "env" / "rollout" / "actor" / "runner"
    rank: int  # worker rank
    stage_id: int  # pipeline stage id
    metadata: dict = field(default_factory=dict)  # 额外元数据

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "worker_type": self.worker_type,
            "rank": self.rank,
            "stage_id": self.stage_id,
            "metadata": self.metadata,
        }

    def to_chrome_trace_event(self) -> dict:
        """Convert to Chrome Trace Event Format."""
        return {
            "name": self.name,
            "cat": self.worker_type,
            "ph": "X",  # Complete event
            "ts": self.start_time * 1e6,  # Convert to microseconds
            "dur": self.duration * 1e6,
            "pid": self.rank,
            "tid": f"{self.worker_type}_stage{self.stage_id}",
            "args": self.metadata,
        }


class TimelineProfiler:
    """
    Records fine-grained timing events for pipeline visualization.

    This profiler tracks the detailed interaction between env, rollout, and actor
    workers, capturing the multi-turn communication patterns.

    Usage:
        profiler = TimelineProfiler()

        # Start a profiling session
        profiler.start_session()

        # Record events
        with profiler.record("env_step", "env", rank=0, stage_id=0):
            # ... env interaction code ...

        with profiler.record("predict", "rollout", rank=0, stage_id=0):
            # ... model inference code ...

        # Export results
        profiler.save_trace("profile.json")
        profiler.save_bubble_chart("profile.png")
    """

    def __init__(self):
        self._events: list[TimelineEvent] = []
        self._start_time: Optional[float] = None
        self._session_active = False
        self._enabled = True

    def enable(self):
        """Enable profiling."""
        self._enabled = True

    def disable(self):
        """Disable profiling."""
        self._enabled = False

    def start_session(self):
        """Start a new profiling session, clearing previous events."""
        self._start_time = time.time()
        self._events.clear()
        self._session_active = True

    def end_session(self):
        """End the current profiling session."""
        self._session_active = False

    @property
    def events(self) -> list[TimelineEvent]:
        return self._events.copy()

    @property
    def session_duration(self) -> float:
        """Total duration of the profiling session."""
        if not self._events:
            return 0.0
        return max(e.end_time for e in self._events) - min(e.start_time for e in self._events)

    @contextmanager
    def record(
        self,
        name: str,
        worker_type: str,
        rank: int = 0,
        stage_id: int = 0,
        metadata: Optional[dict] = None,
    ):
        """
        Context manager to record a timing event.

        Args:
            name: Event name (e.g., "env_step", "predict", "train")
            worker_type: Type of worker ("env", "rollout", "actor", "runner")
            rank: Worker rank
            stage_id: Pipeline stage ID
            metadata: Additional metadata to attach to the event
        """
        if not self._enabled or not self._session_active:
            yield
            return

        if self._start_time is None:
            self._start_time = time.time()

        start = time.time() - self._start_time
        yield
        end = time.time() - self._start_time

        self._events.append(
            TimelineEvent(
                name=name,
                start_time=start,
                end_time=end,
                worker_type=worker_type,
                rank=rank,
                stage_id=stage_id,
                metadata=metadata or {},
            )
        )

    def add_event(
        self,
        name: str,
        start_time: float,
        end_time: float,
        worker_type: str,
        rank: int = 0,
        stage_id: int = 0,
        metadata: Optional[dict] = None,
    ):
        """
        Manually add an event with specified timing.

        Useful for recording events from external timing data.
        """
        self._events.append(
            TimelineEvent(
                name=name,
                start_time=start_time,
                end_time=end_time,
                worker_type=worker_type,
                rank=rank,
                stage_id=stage_id,
                metadata=metadata or {},
            )
        )

    def merge_from(self, other: "TimelineProfiler"):
        """Merge events from another profiler instance."""
        # Adjust timestamps if needed
        if self._start_time is None and other._start_time is not None:
            self._start_time = other._start_time
        self._events.extend(other._events)

    def export_trace(self) -> dict:
        """
        Export events in Chrome Trace Event Format.

        This format can be loaded in chrome://tracing for interactive viewing.
        """
        return {
            "traceEvents": [e.to_chrome_trace_event() for e in self._events],
            "displayTimeUnit": "ms",
            "systemTraceEvents": "",
            "otherData": {
                "version": "RLinf Timeline Profiler v1.0",
                "total_events": len(self._events),
                "session_duration_s": self.session_duration,
            },
        }

    def export_raw_data(self) -> dict:
        """
        Export raw event data as a dictionary.

        Useful for further analysis or custom visualization.
        """
        return {
            "events": [e.to_dict() for e in self._events],
            "summary": self.get_summary(),
        }

    def get_summary(self) -> dict:
        """
        Generate a summary of recorded events.

        Returns statistics grouped by worker_type and event name.
        """
        summary = {
            "total_events": len(self._events),
            "session_duration_s": self.session_duration,
            "by_worker_type": {},
            "by_event_name": {},
        }

        # Group by worker type
        for event in self._events:
            wt = event.worker_type
            if wt not in summary["by_worker_type"]:
                summary["by_worker_type"][wt] = {
                    "count": 0,
                    "total_duration_s": 0.0,
                    "events": [],
                }
            summary["by_worker_type"][wt]["count"] += 1
            summary["by_worker_type"][wt]["total_duration_s"] += event.duration
            summary["by_worker_type"][wt]["events"].append(event.name)

        # Group by event name
        for event in self._events:
            name = event.name
            if name not in summary["by_event_name"]:
                summary["by_event_name"][name] = {
                    "count": 0,
                    "total_duration_s": 0.0,
                    "min_duration_s": float("inf"),
                    "max_duration_s": 0.0,
                    "avg_duration_s": 0.0,
                    "durations": [],
                }
            d = summary["by_event_name"][name]
            d["count"] += 1
            d["total_duration_s"] += event.duration
            d["min_duration_s"] = min(d["min_duration_s"], event.duration)
            d["max_duration_s"] = max(d["max_duration_s"], event.duration)
            d["durations"].append(event.duration)

        # Calculate averages
        for name_data in summary["by_event_name"].values():
            if name_data["durations"]:
                name_data["avg_duration_s"] = np.mean(name_data["durations"])

        return summary

    def save_trace(self, filepath: str):
        """
        Save trace data in Chrome Trace Event Format (JSON).

        Args:
            filepath: Path to save the JSON file
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        trace_data = self.export_trace()
        with open(filepath, "w") as f:
            json.dump(trace_data, f, indent=2)

    def save_raw_data(self, filepath: str):
        """
        Save raw event data as JSON.

        Args:
            filepath: Path to save the JSON file
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        raw_data = self.export_raw_data()
        with open(filepath, "w") as f:
            json.dump(raw_data, f, indent=2)

    def save_bubble_chart(self, filepath: str, figsize: tuple = (24, 10), dpi: int = 150):
        """
        Save a pipeline bubble chart visualization.

        Args:
            filepath: Path to save the image (e.g., "profile.png")
            figsize: Figure size (width, height) in inches
            dpi: Resolution of the saved image
        """
        from rlinf.utils.timeline_visualizer import PipelineBubbleChart

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        chart = PipelineBubbleChart(self)
        chart.plot(save_path=filepath, figsize=figsize, dpi=dpi)

    def clear(self):
        """Clear all recorded events."""
        self._events.clear()
        self._start_time = None
        self._session_active = False


class GlobalProfiler:
    """
    Global profiler instance for easy access across workers.

    This provides a singleton-like access to a TimelineProfiler,
    making it easy to record events from different parts of the codebase.
    """

    _instance: Optional[TimelineProfiler] = None
    _enabled: bool = False

    @classmethod
    def get(cls) -> TimelineProfiler:
        """Get the global profiler instance."""
        if cls._instance is None:
            cls._instance = TimelineProfiler()
        return cls._instance

    @classmethod
    def enable(cls):
        """Enable the global profiler."""
        cls._enabled = True
        cls.get().enable()

    @classmethod
    def disable(cls):
        """Disable the global profiler."""
        cls._enabled = False
        if cls._instance is not None:
            cls._instance.disable()

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if profiling is enabled."""
        return cls._enabled

    @classmethod
    def start_session(cls):
        """Start a new profiling session."""
        cls.get().start_session()

    @classmethod
    def end_session(cls):
        """End the current profiling session."""
        cls.get().end_session()

    @classmethod
    @contextmanager
    def record(
        cls,
        name: str,
        worker_type: str,
        rank: int = 0,
        stage_id: int = 0,
        metadata: Optional[dict] = None,
    ):
        """Context manager to record a timing event using the global profiler."""
        with cls.get().record(name, worker_type, rank, stage_id, metadata):
            yield

    @classmethod
    def save_all(cls, output_dir: str, prefix: str = "profile"):
        """
        Save all profiling data (trace, raw data, and visualization).

        Args:
            output_dir: Directory to save files
            prefix: Prefix for output filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        profiler = cls.get()

        # Save Chrome trace format
        profiler.save_trace(os.path.join(output_dir, f"{prefix}_trace.json"))

        # Save raw data
        profiler.save_raw_data(os.path.join(output_dir, f"{prefix}_raw.json"))

        # Save visualization
        profiler.save_bubble_chart(os.path.join(output_dir, f"{prefix}_timeline.png"))

        # Save summary
        summary = profiler.get_summary()
        with open(os.path.join(output_dir, f"{prefix}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[Profiler] Saved profiling data to {output_dir}/")
        print(f"  - {prefix}_trace.json (Chrome Trace Format)")
        print(f"  - {prefix}_raw.json (Raw Event Data)")
        print(f"  - {prefix}_timeline.png (Pipeline Bubble Chart)")
        print(f"  - {prefix}_summary.json (Statistics Summary)")
