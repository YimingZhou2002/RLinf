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

"""LLM-critic-driven auto-tuner for RLinf embodied training configs.

The orchestrator iteratively proposes config deltas, validates them with
``rlinf.config.validate_cfg``, runs RLinf trials, parses ``metrics.log`` and
``timeline/*.jsonl``, and converges on a config minimising
``step_time / num_trajectories`` under memory/feasibility constraints.

The per-trial loop is plain Python; it is intentionally not an RLCR loop.
Placement-touching deltas proposed by the critic must cite both
``metrics.log`` MetricTable evidence and ``timeline/*.jsonl`` evidence.

This package must not import ``toolkits.auto_placement``; the smoke-test
suite enforces this at test time.
"""
