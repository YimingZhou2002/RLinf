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
Timeline visualization for pipeline bubble charts.

This module provides visualization tools to create pipeline bubble charts
showing the execution timeline of env, rollout, and actor workers.
"""

import os
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba

if TYPE_CHECKING:
    from rlinf.utils.timeline_profiler import TimelineProfiler, TimelineEvent


# Color scheme for different worker types
WORKER_COLORS = {
    "env": "#FF6B6B",  # Red - environment interaction
    "rollout": "#4ECDC4",  # Teal - model inference
    "actor": "#45B7D1",  # Blue - training
    "runner": "#96CEB4",  # Green - coordination
    "reward": "#FFEAA7",  # Yellow - reward computation
}

# Darker shades for borders
WORKER_BORDER_COLORS = {
    "env": "#CC5555",
    "rollout": "#3BA39A",
    "actor": "#3692A8",
    "runner": "#77A58A",
    "reward": "#CCBB88",
}


class PipelineBubbleChart:
    """
    Creates a pipeline bubble chart visualization from TimelineProfiler data.

    This visualization shows the execution timeline of different workers,
    highlighting the parallel execution and communication patterns between
    env, rollout, and actor components.

    Example:
        >>> from rlinf.utils.timeline_profiler import TimelineProfiler
        >>> profiler = TimelineProfiler()
        >>> # ... record events ...
        >>> chart = PipelineBubbleChart(profiler)
        >>> chart.plot(save_path="timeline.png")
    """

    def __init__(self, profiler: "TimelineProfiler"):
        """
        Initialize the bubble chart with a profiler instance.

        Args:
            profiler: TimelineProfiler instance containing recorded events
        """
        self.profiler = profiler
        self.events = profiler.events

    def _group_events_by_row(self) -> dict[tuple[str, int, int], list["TimelineEvent"]]:
        """
        Group events by (worker_type, rank, stage_id) for row display.

        Returns:
            Dictionary mapping row keys to lists of events
        """
        events_by_row: dict[tuple[str, int, int], list["TimelineEvent"]] = defaultdict(list)
        for event in self.events:
            key = (event.worker_type, event.rank, event.stage_id)
            events_by_row[key].append(event)
        return events_by_row

    def _get_row_label(self, worker_type: str, rank: int, stage_id: int) -> str:
        """Generate a human-readable row label."""
        stage_str = f"-S{stage_id}" if stage_id > 0 else ""
        return f"{worker_type}[{rank}]{stage_str}"

    def _calculate_row_order(self, events_by_row: dict) -> list[tuple[str, int, int]]:
        """
        Calculate the display order of rows.

        Order: runner -> env -> rollout -> actor -> reward
        Within each type, sort by rank then stage_id.
        """
        worker_order = {"runner": 0, "env": 1, "rollout": 2, "actor": 3, "reward": 4}

        def sort_key(key):
            worker_type, rank, stage_id = key
            return (worker_order.get(worker_type, 5), rank, stage_id)

        return sorted(events_by_row.keys(), key=sort_key)

    def plot(
        self,
        save_path: Optional[str] = None,
        figsize: tuple = (24, 10),
        dpi: int = 150,
        title: str = "Pipeline Execution Timeline",
        show_idle_gaps: bool = True,
        min_event_width: float = 0.01,
    ):
        """
        Plot the pipeline bubble chart.

        Args:
            save_path: Path to save the image (e.g., "timeline.png")
            figsize: Figure size (width, height) in inches
            dpi: Resolution of the saved image
            title: Title of the chart
            show_idle_gaps: Whether to highlight idle gaps between events
            min_event_width: Minimum width for displaying event labels
        """
        if not self.events:
            print("No events to visualize")
            return

        events_by_row = self._group_events_by_row()
        row_order = self._calculate_row_order(events_by_row)

        # Calculate time range
        min_time = min(e.start_time for e in self.events)
        max_time = max(e.end_time for e in self.events)
        time_range = max_time - min_time

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        num_rows = len(row_order)
        row_height = 0.8
        row_gap = 0.1

        # Track statistics for each row
        row_stats = {}

        for idx, row_key in enumerate(row_order):
            worker_type, rank, stage_id = row_key
            row_events = sorted(events_by_row[row_key], key=lambda e: e.start_time)

            y_pos = num_rows - 1 - idx
            total_busy_time = 0.0

            # Draw background for the row
            ax.axhspan(
                y_pos - row_height / 2,
                y_pos + row_height / 2,
                color=to_rgba(WORKER_COLORS.get(worker_type, "gray"), 0.1),
                zorder=0,
            )

            # Draw each event as a bubble
            for event in row_events:
                width = event.duration
                total_busy_time += width

                # Skip very small events
                if width < min_event_width * 0.001:
                    continue

                # Create simple rectangle
                rect = patches.Rectangle(
                    (event.start_time, y_pos - row_height / 2),
                    width,
                    row_height,
                    facecolor=to_rgba(WORKER_COLORS.get(worker_type, "gray"), 0.8),
                    edgecolor=WORKER_BORDER_COLORS.get(worker_type, "gray"),
                    linewidth=0.5,
                    zorder=2,
                )
                ax.add_patch(rect)

                # Add event name label for wider events
                if width > time_range * 0.02:  # Only label events wider than 2% of timeline
                    # Truncate long names
                    label = event.name
                    if len(label) > 15:
                        label = label[:12] + "..."

                    # Choose text color based on background
                    text_color = "white" if worker_type in ["env", "actor"] else "black"

                    ax.text(
                        event.start_time + width / 2,
                        y_pos,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color=text_color,
                        zorder=3,
                    )

            # Calculate idle time
            row_time_span = max(e.end_time for e in row_events) - min(e.start_time for e in row_events)
            idle_time = row_time_span - total_busy_time
            utilization = total_busy_time / row_time_span if row_time_span > 0 else 0

            row_stats[row_key] = {
                "busy_time": total_busy_time,
                "idle_time": idle_time,
                "utilization": utilization,
            }

        # Set up axes
        ax.set_xlim(min_time - time_range * 0.02, max_time + time_range * 0.02)
        ax.set_ylim(-0.5, num_rows - 0.5)

        # Y-axis labels
        y_labels = [self._get_row_label(*key) for key in row_order]
        ax.set_yticks(range(num_rows))
        ax.set_yticklabels([y_labels[num_rows - 1 - i] for i in range(num_rows)], fontsize=10)

        # X-axis
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))

        # Title
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Grid
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Add legend
        legend_patches = [
            patches.Patch(
                facecolor=to_rgba(WORKER_COLORS[wt], 0.8),
                edgecolor=WORKER_BORDER_COLORS.get(wt, "gray"),
                label=wt.capitalize(),
            )
            for wt in ["runner", "env", "rollout", "actor", "reward"]
            if wt in WORKER_COLORS
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=10)

        # Add utilization annotations on the right side
        for idx, row_key in enumerate(row_order):
            y_pos = num_rows - 1 - idx
            stats = row_stats[row_key]
            utilization_text = f"{stats['utilization'] * 100:.1f}%"
            ax.text(
                max_time + time_range * 0.03,
                y_pos,
                utilization_text,
                ha="left",
                va="center",
                fontsize=9,
                color="gray",
            )

        # Add column header for utilization
        ax.text(
            max_time + time_range * 0.03,
            num_rows + 0.3,
            "Utilization",
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="gray",
        )

        plt.tight_layout()

        # Save or show
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            print(f"[PipelineBubbleChart] Saved to {save_path}")

        plt.close()

    def plot_detailed(
        self,
        save_path: Optional[str] = None,
        figsize: tuple = (32, 12),
        dpi: int = 150,
        title: str = "Detailed Pipeline Execution Timeline",
    ):
        """
        Plot a detailed version with more information.

        Includes:
        - Event metadata in tooltips (saved separately)
        - Communication arrows between workers
        - Phase markers
        """
        # First, create the basic plot
        self.plot(save_path=save_path, figsize=figsize, dpi=dpi, title=title)

        # Create additional analysis plot
        self._plot_analysis(save_path.replace(".png", "_analysis.png") if save_path else None)

    def _plot_analysis(self, save_path: Optional[str] = None, dpi: int = 150):
        """Plot analysis charts showing timing breakdowns."""
        if not self.events:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Time distribution by worker type (pie chart)
        ax1 = axes[0, 0]
        worker_times = defaultdict(float)
        for event in self.events:
            worker_times[event.worker_type] += event.duration

        if worker_times:
            labels = list(worker_times.keys())
            sizes = list(worker_times.values())
            colors = [WORKER_COLORS.get(wt, "gray") for wt in labels]
            ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax1.set_title("Time Distribution by Worker Type", fontweight="bold")

        # 2. Event count by type (bar chart)
        ax2 = axes[0, 1]
        event_counts = defaultdict(int)
        for event in self.events:
            event_counts[event.name] += 1

        if event_counts:
            sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            names = [e[0] for e in sorted_events]
            counts = [e[1] for e in sorted_events]
            ax2.barh(names, counts, color="steelblue")
            ax2.set_xlabel("Count")
            ax2.set_title("Top 15 Events by Frequency", fontweight="bold")
            ax2.invert_yaxis()

        # 3. Duration distribution (histogram)
        ax3 = axes[1, 0]
        durations = [e.duration for e in self.events if e.duration > 0]
        if durations:
            ax3.hist(durations, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
            ax3.set_xlabel("Duration (seconds)")
            ax3.set_ylabel("Frequency")
            ax3.set_title("Event Duration Distribution", fontweight="bold")

        # 4. Timeline heatmap (events over time)
        ax4 = axes[1, 1]
        events_by_row = self._group_events_by_row()
        row_order = self._calculate_row_order(events_by_row)

        # Create a simple timeline view
        min_time = min(e.start_time for e in self.events)
        max_time = max(e.end_time for e in self.events)
        time_bins = 100
        bin_width = (max_time - min_time) / time_bins

        heatmap_data = []
        row_labels = []

        for row_key in row_order:
            worker_type, rank, stage_id = row_key
            row_events = events_by_row[row_key]
            row_label = self._get_row_label(worker_type, rank, stage_id)
            row_labels.append(row_label)

            # Calculate activity in each time bin
            activity = []
            for bin_idx in range(time_bins):
                bin_start = min_time + bin_idx * bin_width
                bin_end = bin_start + bin_width

                # Check if any event overlaps with this bin
                busy = False
                for event in row_events:
                    if event.start_time < bin_end and event.end_time > bin_start:
                        busy = True
                        break
                activity.append(1.0 if busy else 0.0)

            heatmap_data.append(activity)

        if heatmap_data:
            import numpy as np

            ax4.imshow(heatmap_data, aspect="auto", cmap="Blues", interpolation="nearest")
            ax4.set_yticks(range(len(row_labels)))
            ax4.set_yticklabels(row_labels)
            ax4.set_xlabel("Time Bin")
            ax4.set_title("Activity Heatmap", fontweight="bold")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            print(f"[PipelineBubbleChart] Analysis saved to {save_path}")

        plt.close()


def create_timeline_from_metrics(
    time_metrics: dict,
    output_dir: str,
    prefix: str = "profile",
    worker_type: str = "unknown",
    rank: int = 0,
    stage_id: int = 0,
):
    """
    Create a timeline visualization from existing time metrics.

    This is a convenience function to create visualizations from
    the existing timer data in the codebase.

    Args:
        time_metrics: Dictionary of timing metrics (e.g., {"env_step": 0.5, "predict": 0.3})
        output_dir: Directory to save output files
        prefix: Prefix for output filenames
        worker_type: Type of worker for the metrics
        rank: Worker rank
        stage_id: Pipeline stage ID
    """
    from rlinf.utils.timeline_profiler import TimelineProfiler

    profiler = TimelineProfiler()
    profiler.start_session()

    # Convert metrics to events (assuming sequential execution)
    current_time = 0.0
    for name, duration in time_metrics.items():
        if isinstance(duration, (int, float)) and duration > 0:
            profiler.add_event(
                name=name,
                start_time=current_time,
                end_time=current_time + duration,
                worker_type=worker_type,
                rank=rank,
                stage_id=stage_id,
            )
            current_time += duration

    profiler.end_session()

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    profiler.save_trace(os.path.join(output_dir, f"{prefix}_trace.json"))
    profiler.save_raw_data(os.path.join(output_dir, f"{prefix}_raw.json"))
    profiler.save_bubble_chart(os.path.join(output_dir, f"{prefix}_timeline.png"))
