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

    RLINF_BENCH_SEED_DIR      : where the seed captures live (default:
                                <runner.logger.log_path>/bench_msgs, i.e. a
                                subfolder of the run's log dir; falls back to
                                /tmp/bench_msgs if no log_path is set)
    RLINF_BENCH_OUT           : where to write results (default = seed dir)
    RLINF_BENCH_CAPTURE       : inline seed capture mode (default "auto"):
                                "auto"   -> capture iff any seed is missing
                                "always" -> always (re)capture before sweeping
                                "never"  -> never capture; require pre-existing seeds
    RLINF_BENCH_ENV_GPUS      : env placement     (default "0-1")
    RLINF_BENCH_ROLLOUT_GPUS  : rollout placement (default "2-3")
    RLINF_BENCH_ACTOR_GPUS    : actor placement   (default "4-7")
    RLINF_BENCH_MULTIPLIERS   : relative sweep multipliers (default "0.25,0.5,1,2,4")
    RLINF_BENCH_ENV_MULTIPLIERS : env-sweep-only multipliers (default "0.25,0.5,1";
                                falls back to RLINF_BENCH_MULTIPLIERS if that is set).
                                Down-sweep-only by default so the env sweep does not
                                over-subscribe a single GPU past the production size.
    RLINF_BENCH_ENV_MAX       : hard absolute cap on env-sweep num_envs (default
                                unset). Bounds the sweep below the num_envs point that
                                triggers the uncatchable native SAPIEN/OIDN TLS-key
                                SIGABRT; for RoboTwin set to 64.
    RLINF_BENCH_WARMUP        : warmup iters  (default 2)
    RLINF_BENCH_REPEATS       : timed iters   (default 5)

Seeds (the intermediate env/rollout/actor messages) are produced *inline*: when
they are missing (or ``RLINF_BENCH_CAPTURE=always``), the harness first runs one
short real env<->rollout round (a single chunk-step) on the already-launched
Bench workers, which dump the captures, then loads them for the sweep. No
separate capture step is needed. ``run_capture.sh`` remains available as a
standalone schema-inspection tool.

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

from rlinf.config import DIFFUSION_MODELS, SupportedModel, validate_cfg
from rlinf.scheduler import Channel, Cluster
from rlinf.tools.benchmark.bench_workers import (
    BenchEmbodiedActor,
    BenchEnvWorker,
    BenchNFTEmbodiedActor,
    BenchRolloutWorker,
)
from rlinf.utils.placement import HybridComponentPlacement

mp.set_start_method("spawn", force=True)

# The seed captures the sweep consumes (env->rollout obs, env->actor trajectory,
# rollout->env policy output). These are produced inline by _run_capture_round.
_SEED_FILES = (
    "env_to_rollout_obs.pt",
    "env_to_actor_trajectory.pt",
    "rollout_policy_output.pt",
)


def _run_capture_round(env_group, rollout_group, actor_group) -> None:
    """Run one real, truncated env<->rollout round to dump the seed messages.

    This mirrors the generate phase of ``EmbodiedRunner.run`` exactly: env and
    rollout are fired concurrently over the shared ``Env``/``Rollout`` channels
    and the actor drains the trajectory channel. Because the Bench workers
    subclass the capture workers and ``RLINF_BENCH_CAPTURE_DIR`` is armed, the
    real ``interact``/``generate`` code paths dump the intermediate messages as a
    byproduct. The round is truncated to a single chunk-step (see the config
    override in ``main``), so it is fast while still emitting every message with
    production-faithful shapes. Unlike the sweep's direct atomic-op calls, these
    real methods manage their own weight/env onload+offload.
    """
    env_ch = Channel.create("Env")
    rollout_ch = Channel.create("Rollout")
    actor_ch = Channel.create("Actor")

    env_handle = env_group.interact(
        input_channel=env_ch,
        rollout_channel=rollout_ch,
        reward_channel=None,  # use_reward_model is off; reward branch is inert
        actor_channel=actor_ch,
    )
    rollout_handle = rollout_group.generate(
        input_channel=rollout_ch,
        output_channel=env_ch,
    )
    # Drain the trajectory channel so the env round completes (and the
    # env->actor trajectory gets built + dumped), matching the runner.
    actor_group.recv_rollout_trajectories(input_channel=actor_ch).wait()
    rollout_handle.wait()
    env_handle.wait()


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
    """Env table with the extra CPU / host-RAM / GPU-util profiling columns.

    ``bootstrap_ms`` is the cost of building+resetting the num_envs-wide sim (the
    env "onload"); ``offload_ms`` is the cost of disposing it. Both scale with
    num_envs, so they are per-row here rather than a single sweep-level value.
    """
    print(f"\n===== {title} =====", flush=True)
    header = (
        f"{'num_envs':>10} | {'step_ms':>10} | {'bootstrap_ms':>13} | "
        f"{'offload_ms':>11} | {'cpu_%_mean':>11} | {'host_rss_MB':>12} | "
        f"{'gpu_util_%':>11} | {'HBM_alloc_MB':>13} | note"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in records:
        if r.get("oom"):
            print(f"{r.get('batch_size', '?'):>10} | {'OOM':>10} |", flush=True)
            continue
        if r.get("error"):
            print(
                f"{r.get('batch_size', '?'):>10} | {'ERR':>10} | "
                f"{r['error']}",
                flush=True,
            )
            continue
        note = "default" if r.get("is_default") else ""
        print(
            f"{r.get('batch_size', '?'):>10} | {r.get('ms_mean', 0):>10.3f} | "
            f"{r.get('bootstrap_ms', 0):>13.1f} | "
            f"{r.get('offload_ms', 0):>11.1f} | "
            f"{r.get('cpu_percent_mean', 0):>11.1f} | "
            f"{r.get('host_rss_peak_MB', 0):>12.1f} | "
            f"{r.get('gpu_util_mean', 0):>11.1f} | "
            f"{r.get('peak_alloc_MB', 0):>13.2f} | {note}",
            flush=True,
        )


def _print_onload_offload(title: str, payload: dict) -> None:
    """Print the one-shot weight onload/offload timings for rollout/actor.

    These are per-sweep (not per batch size): the weights are brought on once
    before the sweep and pushed back to host once after. A ``None`` entry means
    offload was disabled for that component (weights already resident).
    """

    def _fmt(stats: dict | None) -> str:
        if not stats:
            return "n/a (offload disabled)"
        parts = [f"{stats.get('ms', 0):.1f} ms"]
        if "peak_alloc_MB" in stats:
            parts.append(f"peak_alloc {stats['peak_alloc_MB']:.0f} MB")
        if "peak_reserved_MB" in stats:
            parts.append(f"peak_reserved {stats['peak_reserved_MB']:.0f} MB")
        return ", ".join(parts)

    print(f"\n----- {title} (weight on/off GPU) -----", flush=True)
    print(f"  onload : {_fmt(payload.get('onload'))}", flush=True)
    print(f"  offload: {_fmt(payload.get('offload'))}", flush=True)


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
    # Default the seed/results dir to a subfolder of the run's log dir
    # (runner.logger.log_path, set by run_embodiment.sh to the timestamped
    # logs/<ts>-<config> dir) so seeds + sweep outputs live next to the run's
    # logs instead of a shared /tmp path. Falls back to /tmp/bench_msgs only if
    # no log_path is configured. RLINF_BENCH_SEED_DIR still overrides.
    log_path = None
    try:
        log_path = cfg.runner.logger.log_path
    except Exception:
        log_path = None
    default_seed_dir = (
        os.path.join(str(log_path), "bench_msgs") if log_path else "/tmp/bench_msgs"
    )
    seed_dir = os.environ.get("RLINF_BENCH_SEED_DIR", default_seed_dir)
    out_dir = os.environ.get("RLINF_BENCH_OUT", seed_dir)
    env_gpus = os.environ.get("RLINF_BENCH_ENV_GPUS", "0-1")
    rollout_gpus = os.environ.get("RLINF_BENCH_ROLLOUT_GPUS", "2-3")
    actor_gpus = os.environ.get("RLINF_BENCH_ACTOR_GPUS", "4-7")
    multipliers = os.environ.get("RLINF_BENCH_MULTIPLIERS")  # None -> builder default
    # The env axis is special: each point builds a fresh num_envs-wide RoboTwin
    # sim on a *single* GPU, and SAPIEN/OIDN leak TLS pthread-keys per build that
    # are never reclaimed in-process. Sweeping ABOVE the production size (2x/4x)
    # both over-subscribes one GPU (non-representative -- production spreads envs
    # across GPUs) and is what marches into the uncatchable PTHREAD_KEYS_MAX
    # (1024) SIGABRT. So the env sweep gets its own conservative default
    # (down-sweep only, tops out at the production size) plus a hard absolute cap.
    env_multipliers = (
        os.environ.get("RLINF_BENCH_ENV_MULTIPLIERS")
        or multipliers
        or "0.25,0.5,1"
    )
    env_max = os.environ.get("RLINF_BENCH_ENV_MAX")  # optional absolute num_envs cap
    env_cap = int(env_max) if env_max else None
    warmup = int(os.environ.get("RLINF_BENCH_WARMUP", "2"))
    repeats = int(os.environ.get("RLINF_BENCH_REPEATS", "5"))
    capture_mode = os.environ.get("RLINF_BENCH_CAPTURE", "auto").strip().lower()

    # Decide whether to (re)generate the seeds inline via a short real round.
    missing = [
        n for n in _SEED_FILES if not os.path.exists(os.path.join(seed_dir, n))
    ]
    if capture_mode == "never":
        if missing:
            raise FileNotFoundError(
                f"Missing seed captures {missing} in {seed_dir} and "
                "RLINF_BENCH_CAPTURE=never. Either run the standalone capture "
                "(run_capture.sh) or use RLINF_BENCH_CAPTURE=auto|always."
            )
        do_capture = False
    elif capture_mode == "always":
        do_capture = True
    else:  # auto
        do_capture = bool(missing)

    if do_capture:
        os.makedirs(seed_dir, exist_ok=True)
        # Arm the capture hooks on the workers. This must happen BEFORE the
        # Cluster is constructed: the scheduler propagates env vars the driver
        # set (that differ from the pre-ray-start defaults) to every worker
        # (see cluster/node.py:_configure_node_envs). When we do not capture we
        # leave it unset, so the Bench workers' inherited capture hooks stay
        # pure passthrough.
        os.environ["RLINF_BENCH_CAPTURE_DIR"] = seed_dir

    cfg = validate_cfg(cfg)
    with open_dict(cfg):
        # Disjoint placement: the three sweeps run in parallel, no waiting.
        # Clear any pre-existing placement first: some configs collocate the
        # components under a single combined key (e.g. "actor, env, rollout:
        # 0-7"). Leaving that key in place while adding our per-component keys
        # would make the parser register each component twice ("Component env
        # has multiple placements defined").
        placement_cfg = cfg.cluster.component_placement
        for key in list(placement_cfg.keys()):
            del placement_cfg[key]
        placement_cfg.env = env_gpus
        placement_cfg.rollout = rollout_gpus
        placement_cfg.actor = actor_gpus
        if do_capture:
            # Truncate the capture round to a single chunk-step round-trip:
            # n_train_chunk_steps = max_steps_per_rollout_epoch // num_action_chunks.
            # These are read only at init_worker and are unused by the sweeps.
            num_action_chunks = cfg.actor.model.num_action_chunks
            cfg.env.train.max_steps_per_rollout_epoch = num_action_chunks
            cfg.env.train.rollout_epoch = 1

    print(
        f"[bench-sweep] env gpus={env_gpus} rollout gpus={rollout_gpus} "
        f"actor gpus={actor_gpus} seed_dir={seed_dir} out={out_dir} "
        f"capture={'yes' if do_capture else 'no'} (mode={capture_mode})",
        flush=True,
    )
    print(
        f"[bench-sweep] env axis bounded: multipliers={env_multipliers} "
        f"cap={env_cap if env_cap is not None else 'none'} "
        "(guards the native SAPIEN/OIDN TLS-key SIGABRT on large num_envs)",
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
    # Use the NFT-specific actor class for diffusion models (e.g. Wan2.2-TI2V-5B)
    # which requires EmbodiedNFTFSDPPolicy's init_worker (with init_rollout_model)
    # and NFT-specific training loop. The standard EmbodiedFSDPActor is used for
    # all other models (openvla, openpi, gr00t, etc.).
    model_type = SupportedModel(cfg.actor.model.model_type)
    ActorClass = (
        BenchNFTEmbodiedActor
        if model_type in DIFFUSION_MODELS
        else BenchEmbodiedActor
    )
    actor_group = ActorClass.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )

    # Bring all up (each self-loads real weights / builds its real env); no
    # cross-group sync needed for timing.
    env_group.init_worker().wait()
    rollout_group.init_worker().wait()
    actor_group.init_worker().wait()

    if do_capture:
        print(
            f"[bench-sweep] capturing seeds inline (1 chunk-step round) -> {seed_dir}",
            flush=True,
        )
        _run_capture_round(env_group, rollout_group, actor_group)
        still_missing = [
            n for n in _SEED_FILES if not os.path.exists(os.path.join(seed_dir, n))
        ]
        if still_missing:
            raise RuntimeError(
                f"Inline capture round did not produce {still_missing} in "
                f"{seed_dir}. Check the env/rollout worker logs for the round."
            )

    sweep_kwargs = dict(seed_dir=seed_dir, multipliers=multipliers)
    # Ensure out_dir exists before the sweeps fire: the env sweep streams its
    # .partial.jsonl recovery file into it during the run (before _write_outputs
    # would create it), and that write is best-effort/silent on failure.
    os.makedirs(out_dir, exist_ok=True)
    # Fire all three sweeps BEFORE waiting -> they run concurrently on disjoint GPUs.
    # The env sweep is bounded (env-specific multipliers + absolute cap) so it stops
    # before the num_envs point that would trigger the uncatchable native SAPIEN/OIDN
    # TLS-key SIGABRT, and streams each measured row to a .partial.jsonl recovery file
    # so already-collected points survive even if a later build aborts natively.
    env_handle = env_group.bench_interact_sweep(
        warmup=warmup,
        repeats=repeats,
        seed_dir=seed_dir,
        multipliers=env_multipliers,
        cap=env_cap,
        partial_path=os.path.join(out_dir, "sweep_env_interact.partial.jsonl"),
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
    _print_onload_offload("rollout weights", rollout_res)
    _print_table(
        "actor.train_micro_batch  (micro-batch sweep, global pinned)",
        "micro_batch_size",
        actor_res["records"],
    )
    _print_onload_offload("actor weights + optimizer", actor_res)

    _write_outputs(out_dir, "sweep_env_interact", "batch_size", env_res)
    _write_outputs(out_dir, "sweep_rollout_predict", "batch_size", rollout_res)
    _write_outputs(out_dir, "sweep_actor_micro_batch", "micro_batch_size", actor_res)

    print(f"\n[bench-sweep] done. Results in {out_dir}/sweep_*.json|csv", flush=True)


if __name__ == "__main__":
    main()
