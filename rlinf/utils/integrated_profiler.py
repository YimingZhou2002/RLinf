
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
Integrated Timeline Profiler for RLinf embodied training.

This module provides a fine-grained profiling integration that records
detailed timing information for env-rollout-actor interactions.
"""

import os
import time
from contextlib import contextmanager
from typing import Optional

from rlinf.utils.timeline_profiler import TimelineProfiler, TimelineEvent


class IntegratedProfiler:
    """
    Integrated profiler for embodied training pipeline.

    This profiler is designed to be used across multiple workers (env, rollout, actor)
    and provides fine-grained timing for each interaction step.

    Usage in EmbodiedRunner:
        profiler = IntegratedProfiler(enabled=True)
        profiler.start_step()

        with profiler.record("sync_weights", "runner"):
            update_rollout_weights()

        with profiler.record("generate_rollouts", "runner"):
            env_handle = env.interact(...)
            rollout_handle = rollout.generate(...)

        profiler.end_step()
        profiler.save_all("./profile_output")
    """

    _instance: Optional["IntegratedProfiler"] = None
    _enabled: bool = False

    def __init__(self, enabled: bool = False, output_dir: str = "./profile_output"):
        self._profiler = TimelineProfiler()
        self._enabled = enabled
        self._output_dir = output_dir
        self._step_count = 0
        self._current_step_start: Optional[float] = None
        self._worker_type: str = "unknown"
        self._rank: int = 0

    @classmethod
    def get_instance(cls) -> "IntegratedProfiler":
        """Get the global profiler instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def enable(cls, output_dir: str = "./profile_output"):
        """Enable the global profiler."""
        instance = cls.get_instance()
        instance._enabled = True
        instance._output_dir = output_dir
        instance._profiler.enable()

    @classmethod
    def disable(cls):
        """Disable the global profiler."""
        instance = cls.get_instance()
        instance._enabled = False
        instance._profiler.disable()

    @classmethod
    def set_worker_info(cls, worker_type: str, rank: int = 0):
        """Set the current worker type and rank."""
        instance = cls.get_instance()
        instance._worker_type = worker_type
        instance._rank = rank

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_step(self):
        """Start profiling a new training step."""
        if not self._enabled:
            return
        self._profiler.start_session()
        self._current_step_start = time.time()
        self._step_count += 1

    def end_step(self):
        """End the current training step profiling."""
        if not self._enabled:
            return
        self._profiler.end_session()

    @contextmanager
    def record(
        self,
        name: str,
        worker_type: Optional[str] = None,
        stage_id: int = 0,
        metadata: Optional[dict] = None,
    ):
        """Context manager to record a timing event."""
        if not self._enabled:
            yield
            return

        wt = worker_type or self._worker_type
        with self._profiler.record(name, wt, self._rank, stage_id, metadata):
            yield

    def add_event(
        self,
        name: str,
        start_time: float,
        end_time: float,
        worker_type: Optional[str] = None,
        stage_id: int = 0,
    ):
        """Manually add an event."""
        if not self._enabled:
            return
        wt = worker_type or self._worker_type
        self._profiler.add_event(name, start_time, end_time, wt, self._rank, stage_id)

    def record_start(self, name: str, worker_type: Optional[str] = None, stage_id: int = 0) -> float:
        """Record the start of an event and return the timestamp."""
        if not self._enabled:
            return 0.0
        return time.time()

    def record_end(
        self,
        name: str,
        start_time: float,
        worker_type: Optional[str] = None,
        stage_id: int = 0,
    ):
        """Record the end of an event using the start timestamp."""
        if not self._enabled:
            return
        end_time = time.time()
        self.add_event(name, start_time, end_time, worker_type, stage_id)

    def save_all(self, prefix: str = "profile"):
        """Save all profiling data."""
        if not self._enabled:
            return
        os.makedirs(self._output_dir, exist_ok=True)
        self._profiler.save_trace(os.path.join(self._output_dir, f"{prefix}_trace.json"))
        self._profiler.save_raw_data(os.path.join(self._output_dir, f"{prefix}_raw.json"))
        self._profiler.save_bubble_chart(os.path.join(self._output_dir, f"{prefix}_timeline.png"))

        import json

        summary = self._profiler.get_summary()
        with open(os.path.join(self._output_dir, f"{prefix}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[IntegratedProfiler] Saved profiling data to {self._output_dir}/")

    def get_summary(self) -> dict:
        """Get the profiling summary."""
        return self._profiler.get_summary()

    def get_events(self) -> list:
        """Get all recorded events."""
        return self._profiler.events

    def clear(self):
        """Clear all recorded events."""
        self._profiler.clear()


# Global convenience functions
def enable_profiling(output_dir: str = "./profile_output"):
    """Enable global profiling."""
    IntegratedProfiler.enable(output_dir)


def disable_profiling():
    """Disable global profiling."""
    IntegratedProfiler.disable()


def set_worker_info(worker_type: str, rank: int = 0):
    """Set the current worker info."""
    IntegratedProfiler.set_worker_info(worker_type, rank)


def get_profiler() -> IntegratedProfiler:
    """Get the global profiler instance."""
    return IntegratedProfiler.get_instance()


@contextmanager
def profile_event(name: str, worker_type: Optional[str] = None, stage_id: int = 0):
    """Context manager for profiling an event using the global profiler."""
    profiler = get_profiler()
    with profiler.record(name, worker_type, stage_id):
        yield
