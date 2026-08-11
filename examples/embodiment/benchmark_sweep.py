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

"""Entry point for the batch-size / micro-batch sweeps (Step 2).

It launches the env, rollout and actor groups, on **disjoint** GPUs, so the three
sweeps run concurrently with no waiting between them. Each worker self-loads its
real weights / builds its real env at ``init_worker`` (no cross-group sync needed
for timing), then runs its sweep on same-shape fake data built from the Step-1
capture.

This reuses the *real* training config (``--config-name``), so it works for any
env/rollout/actor combination -- it only overrides the GPU placement and reads
sweep knobs from environment variables:

    RLINF_BENCH_SEED_DIR      : where the Step-1 captures live (default /tmp/bench_msgs)
    RLINF_BENCH_OUT           : where to write results (default = seed dir)
    RLINF_BENCH_ENV_GPUS      : env placement     (default "0-1")
    RLINF_BENCH_ROLLOUT_GPUS  : rollout placement (default "2-3")
    RLINF_BENCH_ACTOR_GPUS    : actor placement   (default "4-7")
    RLINF_BENCH_MULTIPLIERS   : relative sweep multipliers (default "0.25,0.5,1,2,4")
    RLINF_BENCH_WARMUP        : warmup iters  (default 2)
    RLINF_BENCH_REPEATS       : timed iters   (default 5)

Example::

    RLINF_BENCH_SEED_DIR=/tmp/bench_msgs \
        bash examples/embodiment/run_sweep.sh maniskill_ppo_openvla
"""

import csv
import json
import os

import hydra
import torch.multiprocessing as mp
from omegaconf import open_dict

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.tools.benchmark.bench_workers import (
    BenchEmbodiedActor,
    BenchEnvWorker,
    BenchRolloutWorker,
)
from rlinf.utils.placement import HybridComponentPlacement

mp.set_start_method("spawn", force=True)


def _pick_rank0(result) -> dict:
    """Group calls return one value per worker; keep rank 0's payload."""
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, dict) and item.get("rank") == 0:
                return item
        for item in result:
            if isinstance(item, dict):
                return item
    return {"records": []}


def _print_table(title: str, size_key: str, records: list[dict]) -> None:
    print(f"\n===== {title} =====", flush=True)
    header = (
        f"{size_key:>18} | {'ms_mean':>10} | {'ms_min':>10} | "
        f"{'peak_alloc_MB':>14} | {'peak_reserved_MB':>16} | "
        f"{'gpu_util_mean':>13} | {'gpu_util_peak':>13} | note"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in records:
        if r.get("oom"):
            print(
                f"{r.get(size_key, '?'):>18} | {'OOM':>10} | {'':>10} | "
                f"{'':>14} | {'':>16} | {'':>13} | {'':>13} |",
                flush=True,
            )
            continue
        note = "default" if r.get("is_default") else ""
        print(
            f"{r.get(size_key, '?'):>18} | {r.get('ms_mean', 0):>10.3f} | "
            f"{r.get('ms_min', 0):>10.3f} | {r.get('peak_alloc_MB', 0):>14.2f} | "
            f"{r.get('peak_reserved_MB', 0):>16.2f} | "
            f"{r.get('gpu_util_mean', 0):>13.1f} | {r.get('gpu_util_peak', 0):>13.1f} | {note}",
            flush=True,
        )


def _print_env_table(title: str, records: list[dict]) -> None:
    """Env table with the extra CPU / host-RAM / GPU-util profiling columns."""
    print(f"\n===== {title} =====", flush=True)
    header = (
        f"{'num_envs':>10} | {'ms_mean':>10} | {'cpu_%_mean':>11} | "
        f"{'cpu_%_peak':>11} | {'host_rss_MB':>12} | {'gpu_util_%':>11} | "
        f"{'HBM_alloc_MB':>13} | note"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in records:
        if r.get("oom"):
            print(f"{r.get('batch_size', '?'):>10} | {'OOM':>10} |", flush=True)
            continue
        note = "default" if r.get("is_default") else ""
        print(
            f"{r.get('batch_size', '?'):>10} | {r.get('ms_mean', 0):>10.3f} | "
            f"{r.get('cpu_percent_mean', 0):>11.1f} | "
            f"{r.get('cpu_percent_peak', 0):>11.1f} | "
            f"{r.get('host_rss_peak_MB', 0):>12.1f} | "
            f"{r.get('gpu_util_mean', 0):>11.1f} | "
            f"{r.get('peak_alloc_MB', 0):>13.2f} | {note}",
            flush=True,
        )


def _write_outputs(out_dir: str, name: str, size_key: str, payload: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=2)
    records = payload.get("records", [])
    if records:
        keys = sorted({k for r in records for k in r.keys()})
        keys = [size_key] + [k for k in keys if k != size_key]
        with open(os.path.join(out_dir, f"{name}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in records:
                writer.writerow(r)


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvla"
)
def main(cfg) -> None:
    seed_dir = os.environ.get("RLINF_BENCH_SEED_DIR", "/tmp/bench_msgs")
    out_dir = os.environ.get("RLINF_BENCH_OUT", seed_dir)
    env_gpus = os.environ.get("RLINF_BENCH_ENV_GPUS", "0-1")
    rollout_gpus = os.environ.get("RLINF_BENCH_ROLLOUT_GPUS", "2-3")
    actor_gpus = os.environ.get("RLINF_BENCH_ACTOR_GPUS", "4-7")
    multipliers = os.environ.get("RLINF_BENCH_MULTIPLIERS")  # None -> builder default
    warmup = int(os.environ.get("RLINF_BENCH_WARMUP", "2"))
    repeats = int(os.environ.get("RLINF_BENCH_REPEATS", "5"))

    for name in (
        "env_to_rollout_obs.pt",
        "env_to_actor_trajectory.pt",
        "rollout_policy_output.pt",
    ):
        if not os.path.exists(os.path.join(seed_dir, name)):
            raise FileNotFoundError(
                f"Missing capture '{name}' in {seed_dir}. Run the Step-1 capture "
                "first (run_capture.sh) so the default-run message shapes exist."
            )

    cfg = validate_cfg(cfg)
    with open_dict(cfg):
        # Disjoint placement: the three sweeps run in parallel, no waiting.
        cfg.cluster.component_placement.env = env_gpus
        cfg.cluster.component_placement.rollout = rollout_gpus
        cfg.cluster.component_placement.actor = actor_gpus

    print(
        f"[bench-sweep] env gpus={env_gpus} rollout gpus={rollout_gpus} "
        f"actor gpus={actor_gpus} seed_dir={seed_dir} out={out_dir}",
        flush=True,
    )

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    placement = HybridComponentPlacement(cfg, cluster)

    env_group = BenchEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    rollout_group = BenchRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    actor_group = BenchEmbodiedActor.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )

    # Bring all up (each self-loads real weights / builds its real env); no
    # cross-group sync needed for timing.
    env_group.init_worker().wait()
    rollout_group.init_worker().wait()
    actor_group.init_worker().wait()

    sweep_kwargs = dict(seed_dir=seed_dir, multipliers=multipliers)
    # Fire all three sweeps BEFORE waiting -> they run concurrently on disjoint GPUs.
    env_handle = env_group.bench_interact_sweep(
        warmup=warmup, repeats=repeats, **sweep_kwargs
    )
    rollout_handle = rollout_group.bench_predict_sweep(
        warmup=warmup, repeats=repeats, **sweep_kwargs
    )
    actor_handle = actor_group.bench_micro_batch_sweep(
        warmup=max(1, warmup - 1), repeats=max(1, repeats - 2), **sweep_kwargs
    )

    env_res = _pick_rank0(env_handle.wait())
    rollout_res = _pick_rank0(rollout_handle.wait())
    actor_res = _pick_rank0(actor_handle.wait())

    _print_env_table(
        "env.interact_step  (parallelism sweep, CPU/GPU/HBM/host-RAM)",
        env_res["records"],
    )
    _print_table("rollout.predict  (batch-size sweep)", "batch_size", rollout_res["records"])
    _print_table(
        "actor.train_micro_batch  (micro-batch sweep, global pinned)",
        "micro_batch_size",
        actor_res["records"],
    )

    _write_outputs(out_dir, "sweep_env_interact", "batch_size", env_res)
    _write_outputs(out_dir, "sweep_rollout_predict", "batch_size", rollout_res)
    _write_outputs(out_dir, "sweep_actor_micro_batch", "micro_batch_size", actor_res)

    print(f"\n[bench-sweep] done. Results in {out_dir}/sweep_*.json|csv", flush=True)


if __name__ == "__main__":
    main()
