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
Pipeline Bubble Chart Visualizer for RLinf async training.

This module provides visualization tools to create pipeline bubble charts
(similar to NVIDIA Nsight Systems or Chrome tracing) from fine-grained
profiling data.

The visualization shows:
- Each worker (env, rollout, actor) as a separate row/track
- Each sub-step (predict, env_step) as a colored bar showing duration
- Pipeline stages as parallel tracks
- Time alignment to show the async pipeline execution pattern

Usage:
    # After collecting profiling data from a training step
    from rlinf.utils.fine_grained_profiler import TimelineAggregator
    from rlinf.utils.pipeline_visualizer import PipelineVisualizer
    
    aggregator = TimelineAggregator()
    aggregator.add_worker_intervals("rollout", rollout_intervals)
    aggregator.add_worker_intervals("env", env_intervals)
    aggregator.normalize_timeline()
    
    visualizer = PipelineVisualizer()
    visualizer.plot(aggregator, save_path="pipeline_bubble_chart.png")
    
    # Or save as Chrome trace format for interactive viewing
    aggregator.save_chrome_trace("timeline.json")
    # Open in chrome://tracing or https://ui.perfetto.dev/
"""

import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np

from rlinf.utils.fine_grained_profiler import TimeInterval, TimelineAggregator


class PipelineVisualizer:
    """
    Visualizes pipeline execution as a bubble chart.
    
    The chart shows:
    - Horizontal axis: Time (seconds)
    - Vertical axis: Worker tracks (env/rollout/actor per rank and stage)
    - Bars: Duration of each sub-step (predict, env_step, etc.)
    - Colors: Different operation types
    """
    
    # Color scheme for different operation types
    COLORS = {
        "predict": "#3498db",  # Blue - model inference
        "env_step": "#e74c3c",  # Red - environment interaction
        "recv_rollout_results": "#2ecc71",  # Green - communication
        "send_env_batch": "#f39c12",  # Orange - communication
        "sync_weights": "#9b59b6",  # Purple - weight sync
        "actor_training": "#1abc9c",  # Teal - actor training
        "default": "#95a5a6",  # Gray - unknown
    }
    
    # Worker type display names
    WORKER_NAMES = {
        "env": "EnvWorker",
        "rollout": "RolloutWorker",
        "actor": "ActorWorker",
        "runner": "Runner",
    }
    
    def __init__(
        self,
        figsize: tuple[int, int] = (16, 10),
        dpi: int = 100,
        bar_height: float = 0.6,
    ):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches.
            dpi: Dots per inch for the figure.
            bar_height: Height of each bar in the chart.
        """
        self.figsize = figsize
        self.dpi = dpi
        self.bar_height = bar_height
    
    def _get_track_id(self, interval: TimeInterval, track_mapping: dict) -> int:
        """Get or create a track ID for an interval."""
        key = (interval.worker_type, interval.rank, interval.stage_id)
        if key not in track_mapping:
            track_mapping[key] = len(track_mapping)
        return track_mapping[key]
    
    def _get_color(self, name: str) -> str:
        """Get color for an operation based on its name."""
        # Extract operation type from name (e.g., "predict_0_0_s0" -> "predict")
        op_type = name.split("_")[0] if "_" in name else name
        return self.COLORS.get(op_type, self.COLORS["default"])
    
    def plot(
        self,
        aggregator: TimelineAggregator,
        save_path: Optional[str] = None,
        title: str = "Pipeline Execution Timeline",
        show_bubbles: bool = True,
        max_time: Optional[float] = None,
    ):
        """
        Plot the pipeline bubble chart.
        
        Args:
            aggregator: TimelineAggregator with collected intervals.
            save_path: Path to save the figure. If None, shows interactively.
            title: Title for the chart.
            show_bubbles: Whether to show duration bars.
            max_time: Maximum time to show (for zooming). None shows all.
        """
        intervals = aggregator.get_all_intervals()
        if not intervals:
            print("No intervals to visualize.")
            return
        
        # Create track mapping
        track_mapping = {}
        for interval in intervals:
            self._get_track_id(interval, track_mapping)
        
        num_tracks = len(track_mapping)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot each interval as a bar
        patches = []
        for interval in intervals:
            track_id = self._get_track_id(interval, track_mapping)
            y = track_id  # Track position
            
            # Bar from start to end
            width = interval.duration
            left = interval.start_time
            
            if max_time is not None and left > max_time:
                continue
            
            color = self._get_color(interval.name)
            
            # Create rectangle patch
            rect = mpatches.Rectangle(
                (left, y - self.bar_height / 2),
                width,
                self.bar_height,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
            )
            patches.append(rect)
            
            # Add text label for very long operations
            if width > 0.1:  # Only label if duration > 0.1s
                ax.text(
                    left + width / 2,
                    y,
                    f"{width:.2f}s",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white",
                )
        
        # Add all patches
        ax.add_collection(PatchCollection(patches, match_original=True))
        
        # Set axis limits
        max_end_time = max(i.end_time for i in intervals)
        if max_time is not None:
            max_end_time = min(max_end_time, max_time)
        
        ax.set_xlim(0, max_end_time)
        ax.set_ylim(-0.5, num_tracks + 0.5)
        
        # Create track labels
        track_labels = {}
        for key, track_id in track_mapping.items():
            worker_type, rank, stage_id = key
            worker_name = self.WORKER_NAMES.get(worker_type, worker_type)
            track_labels[track_id] = f"{worker_name}[{rank}]_stage{stage_id}"
        
        # Set y-axis labels
        ax.set_yticks(range(num_tracks))
        ax.set_yticklabels([track_labels[i] for i in range(num_tracks)])
        
        # Labels and title
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Worker Track")
        ax.set_title(title)
        
        # Add legend
        legend_patches = [
            mpatches.Patch(color=color, label=name)
            for name, color in self.COLORS.items()
            if name != "default"
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
        
        # Grid
        ax.grid(True, axis="x", alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved pipeline chart to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
    
    def plot_pipeline_efficiency(
        self,
        aggregator: TimelineAggregator,
        save_path: Optional[str] = None,
    ):
        """
        Plot pipeline efficiency analysis.
        
        Shows:
        - Total time per worker type
        - Overlap percentage (how much parallel execution)
        - Bubble ratio (idle time vs active time)
        """
        intervals = aggregator.get_all_intervals()
        if not intervals:
            print("No intervals to analyze.")
            return
        
        # Group by worker type
        by_worker = {}
        for interval in intervals:
            wt = interval.worker_type
            if wt not in by_worker:
                by_worker[wt] = []
            by_worker[wt].append(interval)
        
        # Calculate metrics
        total_time = max(i.end_time for i in intervals)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Time breakdown by worker type
        ax1 = axes[0]
        worker_times = {}
        for wt, wt_intervals in by_worker.items():
            # Sum of durations (not accounting for overlap within same worker)
            total_duration = sum(i.duration for i in wt_intervals)
            worker_times[wt] = total_duration
        
        bars = ax1.bar(worker_times.keys(), worker_times.values(), color=[
            self.COLORS.get(wt + "_total", "#3498db") for wt in worker_times.keys()
        ])
        ax1.set_ylabel("Total Duration (seconds)")
        ax1.set_title("Time by Worker Type")
        ax1.set_xlabel("Worker Type")
        
        # Add value labels
        for bar, val in zip(bars, worker_times.values()):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}s", ha="center", va="bottom", fontsize=9)
        
        # Right: Pipeline utilization
        ax2 = axes[1]
        
        # Calculate overlap: time when multiple workers are active simultaneously
        # Create time bins and count active workers
        time_bins = np.linspace(0, total_time, 1000)
        active_counts = []
        
        for t in time_bins:
            count = 0
            for interval in intervals:
                if interval.start_time <= t < interval.end_time:
                    count += 1
            active_counts.append(count)
        
        ax2.plot(time_bins, active_counts, color="#2ecc71", linewidth=2)
        ax2.fill_between(time_bins, active_counts, alpha=0.3, color="#2ecc71")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Active Workers")
        ax2.set_title("Pipeline Parallelism Over Time")
        ax2.set_ylim(0, max(active_counts) + 1)
        ax2.grid(True, alpha=0.3)
        
        # Add utilization percentage
        avg_active = np.mean(active_counts)
        max_possible = len(track_mapping) if hasattr(self, '_last_track_mapping') else max(active_counts)
        utilization = avg_active / max_possible * 100 if max_possible > 0 else 0
        ax2.text(0.5, 0.95, f"Avg Utilization: {utilization:.1f}%",
                transform=ax2.transAxes, ha="center", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved efficiency analysis to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)


def visualize_from_json(json_path: str, save_path: str):
    """
    Load Chrome trace JSON and create visualization.
    
    Args:
        json_path: Path to Chrome trace JSON file.
        save_path: Path to save the PNG figure.
    """
    with open(json_path, "r") as f:
        trace_data = json.load(f)
    
    aggregator = TimelineAggregator()
    
    for event in trace_data.get("traceEvents", []):
        if event.get("ph") != "X":  # Only complete events
            continue
        
        interval_dict = {
            "name": event["name"],
            "start": event["ts"] / 1e6,  # Convert from microseconds
            "end": (event["ts"] + event["dur"]) / 1e6,
            "rank": event.get("pid", 0),
            "stage_id": 0,  # Extract from tid if available
            "worker_type": event.get("cat", "unknown"),
            "metadata": event.get("args", {}),
        }
        
        # Extract stage from tid if present
        tid = event.get("tid", "")
        if "stage" in str(tid):
            try:
                stage_id = int(str(tid).split("stage")[-1])
                interval_dict["stage_id"] = stage_id
            except:
                pass
        
        aggregator.add_worker_intervals(
            interval_dict["worker_type"],
            [interval_dict],
        )
    
    visualizer = PipelineVisualizer()
    visualizer.plot(aggregator, save_path=save_path)


# Example usage script
if __name__ == "__main__":
    # Demo with synthetic data
    print("Pipeline Visualizer Demo")
    print("=" * 50)
    
    # Create synthetic intervals for demo
    aggregator = TimelineAggregator()
    
    # Simulate a pipeline with 2 stages, 1 env worker, 1 rollout worker
    # Env steps (stage 0 and 1 interleaved)
    for i in range(5):
        aggregator.add_worker_intervals("env", [
            {"name": f"env_step_{i}_s0", "start": i * 0.5, "end": i * 0.5 + 0.3, "rank": 0, "stage_id": 0},
        ])
        aggregator.add_worker_intervals("env", [
            {"name": f"env_step_{i}_s1", "start": i * 0.5 + 0.1, "end": i * 0.5 + 0.4, "rank": 0, "stage_id": 1},
        ])
    
    # Rollout predict steps (slightly delayed after env)
    for i in range(5):
        aggregator.add_worker_intervals("rollout", [
            {"name": f"predict_{i}_s0", "start": i * 0.5 + 0.05, "end": i * 0.5 + 0.25, "rank": 0, "stage_id": 0},
        ])
        aggregator.add_worker_intervals("rollout", [
            {"name": f"predict_{i}_s1", "start": i * 0.5 + 0.15, "end": i * 0.5 + 0.35, "rank": 0, "stage_id": 1},
        ])
    
    # Visualize
    visualizer = PipelineVisualizer(figsize=(12, 6))
    visualizer.plot(aggregator, save_path="demo_pipeline_chart.png", title="Demo Pipeline Execution")
    visualizer.plot_pipeline_efficiency(aggregator, save_path="demo_efficiency.png")
    
    # Save Chrome trace
    aggregator.save_chrome_trace("demo_timeline.json")
    print("\nDemo files created:")
    print("  - demo_pipeline_chart.png")
    print("  - demo_efficiency.png")
    print("  - demo_timeline.json (open in chrome://tracing)")