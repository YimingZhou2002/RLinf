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

"""Tools for benchmarking / profiling the embodied env-rollout-actor pipeline."""

from rlinf.tools.benchmark.message_capture import (
    CaptureEnvWorker,
    CaptureRolloutWorker,
    capture_enabled,
    describe,
)

__all__ = [
    "CaptureEnvWorker",
    "CaptureRolloutWorker",
    "capture_enabled",
    "describe",
]

# The sweep tools (bench_workers) pull in the actor/rollout worker modules and
# their heavy deps. Import them lazily so importing this package (e.g. for
# capture-only use) stays cheap; access them via ``rlinf.tools.benchmark.<name>``.
_LAZY = {
    "BenchRolloutWorker": "rlinf.tools.benchmark.bench_workers",
    "BenchEmbodiedActor": "rlinf.tools.benchmark.bench_workers",
    "BenchNFTEmbodiedActor": "rlinf.tools.benchmark.bench_workers",
    "BenchEnvWorker": "rlinf.tools.benchmark.bench_workers",
    "resize_batch": "rlinf.tools.benchmark.fake_messages",
    "make_sizes": "rlinf.tools.benchmark.sweeps",
    "timed_vram": "rlinf.tools.benchmark.measure",
}


def __getattr__(name):  # PEP 562 module-level lazy attribute access
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)
