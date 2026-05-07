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
Fine-grained Timeline Profiler for RLinf distributed training.

This module provides tools to record detailed timing intervals for each
sub-step in the async pipeline (env-rollout-actor interactions).

The key insight is that RLinf uses async pipeline with multiple iterations:
- Rollout: generate_one_epoch contains n_train_chunk_steps × num_pipeline_stages predict calls
- Env: run_interact_once contains rollout_epoch × n_train_chunk_steps × stage_num env_interact_step calls

We record each sub-step as a time interval [start, end] and visualize as a pipeline bubble chart.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TimeInterval:
    """A single time interval representing a sub-step execution."""
    
    name: str  # 事件名称 (e.g., "predict_0", "env_step_0_stage_0")
    start_time: float  # 开始时间戳 (秒)
    end_time: float  # 结束时间戳 (秒)
    worker_type: str  # "env" / "rollout" / "actor" / "runner"
    rank: int  # worker rank
    iteration_idx: int  # 在循环中的迭代索引
    stage_id: int  # pipeline stage id
    metadata: dict = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start": self.start_time,
            "end": self.end_time,
            "duration": self.duration,
            "worker_type": self.worker_type,
            "rank": self.rank,
            "iteration_idx": self.iteration_idx,
            "stage_id": self.stage_id,
            "metadata": self.metadata,
        }


class FineGrainedProfiler:
    """
    Records fine-grained time intervals for each sub-step in the pipeline.
    
    This profiler is designed to be used within Worker methods to record
    each iteration of the inner loops (predict, env_interact_step, etc.)
    
    Usage in RolloutWorker:
        profiler = FineGrainedProfiler.get_instance()
        profiler.start_session("rollout", rank=0)
        
        for i in range(n_train_chunk_steps):
            for j in range(num_pipeline_stages):
                with profiler.record_iteration("predict", i, j):
                    actions, result = self.predict(env_obs)
        
        intervals = profiler.get_intervals()
        # Return intervals as part of the result
    
    Usage in EnvWorker:
        profiler = FineGrainedProfiler.get_instance()
        profiler.start_session("env", rank=0)
        
        for epoch in range(rollout_epoch):
            for step in range(n_train_chunk_steps):
                for stage in range(stage_num):
                    with profiler.record_iteration("env_step", step, stage, epoch):
                        env_output, env_info = self.env_interact_step(...)
        
        intervals = profiler.get_intervals()
    """
    
    _instance: Optional["FineGrainedProfiler"] = None
    _enabled: bool = False
    
    def __init__(self):
        self._intervals: list[TimeInterval] = []
        self._session_start_time: Optional[float] = None
        self._session_active = False
        self._worker_type: str = "unknown"
        self._rank: int = 0
        self._enabled = False
    
    @classmethod
    def get_instance(cls) -> "FineGrainedProfiler":
        """Get the global profiler instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def enable(cls):
        """Enable the global profiler."""
        cls._enabled = True
        cls.get_instance()._enabled = True
    
    @classmethod
    def disable(cls):
        """Disable the global profiler."""
        cls._enabled = False
        if cls._instance is not None:
            cls._instance._enabled = False
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if profiling is enabled."""
        return cls._enabled
    
    def start_session(self, worker_type: str, rank: int = 0):
        """Start a new profiling session for this worker."""
        self._session_start_time = time.time()
        self._intervals.clear()
        self._worker_type = worker_type
        self._rank = rank
        self._session_active = True
    
    def end_session(self) -> list[TimeInterval]:
        """End the current session and return recorded intervals."""
        self._session_active = False
        return self._intervals.copy()
    
    def record_iteration(
        self,
        name: str,
        iteration_idx: int = 0,
        stage_id: int = 0,
        epoch_idx: int = 0,
        metadata: Optional[dict] = None,
    ):
        """Context manager to record a single iteration's time interval."""
        import contextlib
        
        @contextlib.contextmanager
        def _record():
            if not self._enabled or not self._session_active:
                yield
                return
            
            start = time.time() - self._session_start_time
            yield
            end = time.time() - self._session_start_time
            
            # Create unique name with indices
            full_name = f"{name}_{epoch_idx}_{iteration_idx}_s{stage_id}"
            
            self._intervals.append(
                TimeInterval(
                    name=full_name,
                    start_time=start,
                    end_time=end,
                    worker_type=self._worker_type,
                    rank=self._rank,
                    iteration_idx=iteration_idx,
                    stage_id=stage_id,
                    metadata={
                        "epoch": epoch_idx,
                        "iteration": iteration_idx,
                        "stage": stage_id,
                        **(metadata or {}),
                    },
                )
            )
        
        return _record()
    
    def get_intervals(self) -> list[TimeInterval]:
        """Get all recorded intervals."""
        return self._intervals.copy()
    
    def get_intervals_as_dict(self) -> list[dict]:
        """Get intervals as list of dicts for serialization."""
        return [i.to_dict() for i in self._intervals]
    
    def clear(self):
        """Clear all recorded intervals."""
        self._intervals.clear()
        self._session_start_time = None
        self._session_active = False


class TimelineAggregator:
    """
    Aggregates timeline intervals from multiple workers and generates visualization.
    
    This class takes the intervals returned from each worker (via Handle results)
    and combines them into a unified timeline for visualization.
    
    The data format is like:
    {
        "rollout": [
            {"name": "predict_0_0_s0", "start": 0.1, "end": 0.5, "rank": 0},
            {"name": "predict_0_1_s1", "start": 0.5, "end": 0.9, "rank": 0},
            ...
        ],
        "env": [
            {"name": "env_step_0_0_s0", "start": 0.0, "end": 0.3, "rank": 0},
            {"name": "env_step_0_1_s1", "start": 0.3, "end": 0.6, "rank": 0},
            ...
        ],
        "actor": [...],
    }
    """
    
    def __init__(self):
        self._intervals_by_worker: dict[str, list[TimeInterval]] = {}
        self._step_start_time: Optional[float] = None
    
    def add_worker_intervals(
        self,
        worker_type: str,
        intervals: list[dict],
        base_time_offset: float = 0.0,
    ):
        """Add intervals from a worker, adjusting for time offset."""
        if not intervals:
            return
        
        for interval_dict in intervals:
            interval = TimeInterval(
                name=interval_dict["name"],
                start_time=interval_dict["start"] + base_time_offset,
                end_time=interval_dict["end"] + base_time_offset,
                worker_type=worker_type,
                rank=interval_dict.get("rank", 0),
                iteration_idx=interval_dict.get("iteration_idx", 0),
                stage_id=interval_dict.get("stage_id", 0),
                metadata=interval_dict.get("metadata", {}),
            )
            
            if worker_type not in self._intervals_by_worker:
                self._intervals_by_worker[worker_type] = []
            self._intervals_by_worker[worker_type].append(interval)
    
    def normalize_timeline(self):
        """Normalize all intervals to start from time 0."""
        all_intervals = []
        for intervals in self._intervals_by_worker.values():
            all_intervals.extend(intervals)
        
        if not all_intervals:
            return
        
        min_start = min(i.start_time for i in all_intervals)
        
        # Adjust all intervals
        for worker_type in self._intervals_by_worker:
            for interval in self._intervals_by_worker[worker_type]:
                interval.start_time -= min_start
                interval.end_time -= min_start
    
    def get_all_intervals(self) -> list[TimeInterval]:
        """Get all intervals sorted by start time."""
        all_intervals = []
        for intervals in self._intervals_by_worker.values():
            all_intervals.extend(intervals)
        return sorted(all_intervals, key=lambda x: x.start_time)
    
    def export_chrome_trace(self) -> dict:
        """Export in Chrome Trace Event Format for chrome://tracing."""
        all_intervals = self.get_all_intervals()
        
        trace_events = []
        for interval in all_intervals:
            trace_events.append({
                "name": interval.name,
                "cat": interval.worker_type,
                "ph": "X",  # Complete event
                "ts": interval.start_time * 1e6,  # Convert to microseconds
                "dur": interval.duration * 1e6,
                "pid": interval.rank,
                "tid": f"{interval.worker_type}_stage{interval.stage_id}",
                "args": interval.metadata,
            })
        
        return {
            "traceEvents": trace_events,
            "displayTimeUnit": "ms",
        }
    
    def save_chrome_trace(self, filepath: str):
        """Save Chrome trace format to file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        trace_data = self.export_chrome_trace()
        with open(filepath, "w") as f:
            json.dump(trace_data, f, indent=2)
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        all_intervals = self.get_all_intervals()
        
        summary = {
            "total_intervals": len(all_intervals),
            "by_worker_type": {},
            "by_stage": {},
        }
        
        for interval in all_intervals:
            wt = interval.worker_type
            stage = interval.stage_id
            
            if wt not in summary["by_worker_type"]:
                summary["by_worker_type"][wt] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "intervals": [],
                }
            summary["by_worker_type"][wt]["count"] += 1
            summary["by_worker_type"][wt]["total_duration"] += interval.duration
            summary["by_worker_type"][wt]["intervals"].append(interval.name)
            
            if stage not in summary["by_stage"]:
                summary["by_stage"][stage] = {"count": 0, "total_duration": 0.0}
            summary["by_stage"][stage]["count"] += 1
            summary["by_stage"][stage]["total_duration"] += interval.duration
        
        return summary
    
    def clear(self):
        """Clear all intervals."""
        self._intervals_by_worker.clear()


# Convenience functions for use in workers
def get_profiler() -> FineGrainedProfiler:
    """Get the global profiler instance."""
    return FineGrainedProfiler.get_instance()


def enable_profiling():
    """Enable fine-grained profiling."""
    FineGrainedProfiler.enable()


def disable_profiling():
    """Disable fine-grained profiling."""
    FineGrainedProfiler.disable()


def is_profiling_enabled() -> bool:
    """Check if profiling is enabled."""
    return FineGrainedProfiler.is_enabled()