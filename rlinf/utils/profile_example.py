#!/usr/bin/env python3
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
Example script demonstrating the Timeline Profiler usage.

This script simulates the env-rollout-actor interaction pattern and
generates a pipeline bubble chart visualization.

Usage:
    python -m rlinf.utils.profile_example
"""

import os
import time
import random

from rlinf.utils.timeline_profiler import TimelineProfiler, GlobalProfiler


def simulate_training_step(
    profiler: TimelineProfiler,
    step_id: int,
    num_envs: int = 2,
    num_stages: int = 2,
):
    """
    Simulate a single training step with env-rollout-actor interactions.

    This demonstrates the multi-turn interaction pattern:
    1. Env sends observation to Rollout
    2. Rollout predicts action
    3. Rollout sends action to Env
    4. Env executes action and sends new observation
    5. Repeat for multiple chunk steps
    6. Actor trains on collected data
    """
    n_chunk_steps = 4  # Number of chunk steps per epoch

    # Simulate the multi-turn env-rollout interaction
    for chunk_idx in range(n_chunk_steps):
        for stage_id in range(num_stages):
            # Env: prepare and send observation
            with profiler.record(
                "send_obs",
                "env",
                rank=0,
                stage_id=stage_id,
                metadata={"chunk_idx": chunk_idx, "step_id": step_id},
            ):
                time.sleep(random.uniform(0.01, 0.03))

            # Rollout: receive observation
            with profiler.record(
                "recv_obs",
                "rollout",
                rank=0,
                stage_id=stage_id,
                metadata={"chunk_idx": chunk_idx, "step_id": step_id},
            ):
                time.sleep(random.uniform(0.005, 0.015))

            # Rollout: model inference (predict action)
            with profiler.record(
                "predict",
                "rollout",
                rank=0,
                stage_id=stage_id,
                metadata={"chunk_idx": chunk_idx, "step_id": step_id},
            ):
                time.sleep(random.uniform(0.05, 0.15))  # Inference takes longer

            # Rollout: send action to env
            with profiler.record(
                "send_action",
                "rollout",
                rank=0,
                stage_id=stage_id,
                metadata={"chunk_idx": chunk_idx, "step_id": step_id},
            ):
                time.sleep(random.uniform(0.005, 0.01))

            # Env: receive action
            with profiler.record(
                "recv_action",
                "env",
                rank=0,
                stage_id=stage_id,
                metadata={"chunk_idx": chunk_idx, "step_id": step_id},
            ):
                time.sleep(random.uniform(0.002, 0.008))

            # Env: execute action in environment
            with profiler.record(
                "env_step",
                "env",
                rank=0,
                stage_id=stage_id,
                metadata={"chunk_idx": chunk_idx, "step_id": step_id},
            ):
                time.sleep(random.uniform(0.02, 0.08))

    # Simulate receiving rollout results
    with profiler.record(
        "recv_rollout_results",
        "env",
        rank=0,
        stage_id=0,
        metadata={"step_id": step_id},
    ):
        time.sleep(random.uniform(0.01, 0.03))

    # Simulate computing bootstrap rewards
    with profiler.record(
        "compute_rewards",
        "env",
        rank=0,
        stage_id=0,
        metadata={"step_id": step_id},
    ):
        time.sleep(random.uniform(0.005, 0.02))

    # Simulate actor training
    with profiler.record(
        "sync_weights",
        "runner",
        rank=0,
        stage_id=0,
        metadata={"step_id": step_id},
    ):
        time.sleep(random.uniform(0.1, 0.3))

    with profiler.record(
        "actor_forward",
        "actor",
        rank=0,
        stage_id=0,
        metadata={"step_id": step_id},
    ):
        time.sleep(random.uniform(0.2, 0.5))

    with profiler.record(
        "actor_backward",
        "actor",
        rank=0,
        stage_id=0,
        metadata={"step_id": step_id},
    ):
        time.sleep(random.uniform(0.3, 0.6))


def main():
    """Run the profiling example."""
    print("=" * 60)
    print("Timeline Profiler Example - Single Step")
    print("=" * 60)

    # Create output directory
    output_dir = "./profile_output"
    os.makedirs(output_dir, exist_ok=True)

    # Create profiler
    profiler = TimelineProfiler()

    # Start profiling session
    print("\n[1] Starting profiling session...")
    profiler.start_session()

    # Simulate a single training step
    print("\n[2] Simulating 1 training step...")

    # Record overall step timing
    with profiler.record(
        "step",
        "runner",
        rank=0,
        stage_id=0,
        metadata={"step_id": 0},
    ):
        simulate_training_step(profiler, step_id=0)
    print("  Step completed")

    # End profiling session
    profiler.end_session()
    print("\n[3] Profiling session completed")

    # Print summary
    print("\n[4] Event Summary:")
    summary = profiler.get_summary()
    print(f"  Total events: {summary['total_events']}")
    print(f"  Session duration: {summary['session_duration_s']:.2f}s")

    print("\n  Time by worker type:")
    for wt, data in summary["by_worker_type"].items():
        print(f"    {wt}: {data['total_duration_s']:.2f}s ({data['count']} events)")

    print("\n  Top events by duration:")
    sorted_events = sorted(
        summary["by_event_name"].items(),
        key=lambda x: x[1]["total_duration_s"],
        reverse=True,
    )[:5]
    for name, data in sorted_events:
        print(f"    {name}: {data['total_duration_s']:.2f}s (avg: {data['avg_duration_s']:.3f}s)")

    # Save all outputs
    print(f"\n[5] Saving outputs to {output_dir}/...")

    # Save Chrome trace format
    trace_path = os.path.join(output_dir, "profile_trace.json")
    profiler.save_trace(trace_path)
    print(f"  Saved: {trace_path}")

    # Save raw data
    raw_path = os.path.join(output_dir, "profile_raw.json")
    profiler.save_raw_data(raw_path)
    print(f"  Saved: {raw_path}")

    # Save bubble chart
    chart_path = os.path.join(output_dir, "profile_timeline.png")
    profiler.save_bubble_chart(chart_path)
    print(f"  Saved: {chart_path}")

    # Save summary
    import json

    summary_path = os.path.join(output_dir, "profile_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # Create detailed analysis
    from rlinf.utils.timeline_visualizer import PipelineBubbleChart

    chart = PipelineBubbleChart(profiler)
    analysis_path = os.path.join(output_dir, "profile_analysis.png")
    chart._plot_analysis(analysis_path)
    print(f"  Saved: {analysis_path}")

    print("\n" + "=" * 60)
    print("Profiling complete!")
    print("=" * 60)
    print(f"\nOutput files in {output_dir}/:")
    print("  - profile_trace.json    : Chrome Trace Format (load in chrome://tracing)")
    print("  - profile_raw.json      : Raw event data for further analysis")
    print("  - profile_timeline.png  : Pipeline bubble chart visualization")
    print("  - profile_summary.json  : Statistics summary")
    print("  - profile_analysis.png  : Detailed analysis charts")
    print("\nTo view the Chrome trace:")
    print("  1. Open Chrome browser")
    print("  2. Navigate to chrome://tracing")
    print("  3. Click 'Load' and select profile_trace.json")


if __name__ == "__main__":
    main()
