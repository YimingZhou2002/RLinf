# RLinf timeline sidecar

This directory is intentionally outside `/mnt/public/zengwen/RLinf`, so the
timeline tooling does not appear in normal RLinf commits.

## Enable tracing

```bash
export PYTHONPATH=/mnt/public/zengwen/rlinf_utils:$PYTHONPATH
export RLINF_TIMELINE_DEBUG=1
export RLINF_TIMELINE=1
export RLINF_TIMELINE_DIR=auto
export RLINF_TIMELINE_WORKER_TIMER=1
export RLINF_TIMELINE_ACTOR_TRAINING=1
export RLINF_TIMELINE_PATCH_FILE=/mnt/public/zengwen/rlinf_utils/timeline_patches.embodied.txt
unset RLINF_TIMELINE_PATCHES


# Then run the normal RLinf command in the same shell.
```

## Enable NVML memory sampling

The same sidecar bootstrap can also start a lightweight per-process NVML sampler.
This is useful when you want process-level GPU memory curves for runner / actor /
rollout / env workers, plus the Behavior subprocess env.
It uses Python NVML bindings (`pynvml` or Ray's vendored binding).

```bash
export PYTHONPATH=/mnt/public/zengwen/rlinf_utils:/mnt/public/zengwen/RLinf:$PYTHONPATH
export RLINF_NVML=1
export RLINF_NVML_INTERVAL=0.2

# Optional: override output directory. By default, samples go to
#   <runner.logger.log_path>/nvml
# when the runner config is available.
# export RLINF_NVML_DIR=/path/to/nvml

# Then run the normal RLinf command in the same shell.
```

Each sampled process writes one JSONL file:

```text
<log_dir>/nvml/<component>_rank<RANK>_pid<PID>.jsonl
```

The sampler does not fall back to a bare `./nvml` directory. If a process starts
before `RLINF_LOG_DIR` is available, it will skip sampling unless
`RLINF_NVML_DIR` is explicitly set.

Each line contains:

- `ts`
- `pid`
- `component`
- `rank`
- `global_step` when it can be inferred
- `devices`: per-GPU NVML memory samples for the current process
- `nvml_total_used_bytes`
- `torch_allocated/reserved/max_*` for the local PyTorch allocator when available

## Enable nvitop resource sampling

`nvitop` sampling records per-process CPU memory/utilization plus GPU memory/util
fields exposed by nvitop. It writes a separate JSONL stream from the NVML sampler:

```bash
export PYTHONPATH=/mnt/public/zengwen/rlinf_utils:/mnt/public/zengwen/RLinf:$PYTHONPATH
export RLINF_NVITOP=1
export RLINF_NVITOP_INTERVAL=0.5

# Optional: override output directory. By default, samples go to
#   <runner.logger.log_path>/nvitop
# export RLINF_NVITOP_DIR=/path/to/nvitop

# Then run the normal RLinf command in the same shell.
```

Each sampled process writes one JSONL file:

```text
<log_dir>/nvitop/<component>_rank<RANK>_pid<PID>.jsonl
```

The sampler no longer falls back to a bare `./nvitop` directory. If a process
starts before `RLINF_LOG_DIR` is available, it will skip sampling unless
`RLINF_NVITOP_DIR` is explicitly set.

Each line includes:

- process CPU and memory: `process_cpu_percent`, `process_rss_bytes`, `process_rss_gib`
- system CPU and memory: `system_cpu_percent`, `system_memory_*`
- GPU device fields when nvitop exposes them: `gpu_util_percent`, `memory_util_percent`, `memory_used_bytes`, `memory_total_bytes`
- current-process GPU entries under `gpus[].processes` when nvitop can map the PID

`sitecustomize.py` is imported automatically by Python when this directory is on
`PYTHONPATH`. If neither `RLINF_TIMELINE` nor `RLINF_NVML` is truthy, it does
nothing.

`RLINF_TIMELINE_DIR=auto` writes into `cfg.runner.logger.log_path/timeline` when
the patched object has `self.cfg`; this keeps traces under each run's log
directory. A concrete `RLINF_TIMELINE_DIR=/path/to/timeline` still overrides it.

For RLinf worker methods decorated with `@Worker.timer(...)`, enable automatic
fine-grained tracing:

```bash
export RLINF_TIMELINE_WORKER_TIMER=1
```

For embodied FSDP actor training internals, enable:

```bash
export RLINF_TIMELINE_ACTOR_TRAINING=1
```

This records `actor_forward`, `actor_policy_loss`, `actor_backward`, and
`actor_optimizer_step` under the actor lanes during `EmbodiedFSDPActor.run_training`.

Worker timer events include extra JSONL fields when they can be inferred from
the call context, such as `global_step`, `rollout_epoch`, `chunk_step`,
`stage_id`, `phase`, `mode`, and `call_index`. The Plotly HTML hover tooltip
shows these fields for each bar.

The worker timer sidecar skips noisy outer/waiting timers by default:
`interact`, `run_interact_once`, `generate_one_epoch`, and
`recv_rollout_results`. Override with:

```bash
export RLINF_TIMELINE_WORKER_TIMER_EXCLUDE_TAGS=
```

The plotter also hides full-span wrapper/waiting tags by default:
`interact`, `run_interact_once`, `generate`, `generate_one_epoch`, and
`recv_rollout_results`, `recv_rollout_trajectories`, and `run_training`. Pass
`--exclude-tags ""` to show everything, or
`--include-tags predict,env_interact_step,actor_forward` to focus on specific
tags.

Patch spec format:

```text
module:qualname:component:tag
```

You can also put one spec per line in a local file and use:

```bash
export RLINF_TIMELINE_PATCH_FILE=/mnt/public/zengwen/rlinf_utils/timeline_patches.txt
```

## Disable tracing

```bash
unset RLINF_TIMELINE
unset RLINF_NVML
unset RLINF_NVITOP
# or remove /mnt/public/zengwen/rlinf_utils from PYTHONPATH
```

## Plot

```bash
python /mnt/public/zengwen/rlinf_utils/plot_timeline.py \
  /mnt/public/zengwen/RLinf/logs/my-run/timeline

python /mnt/public/zengwen/rlinf_utils/plot_nvml.py \
  /mnt/public/zengwen/RLinf/logs/my-run/nvml

python /mnt/public/zengwen/rlinf_utils/plot_nvitop.py \
  /mnt/public/zengwen/RLinf/logs/my-run/nvitop
```

All plotters default to interactive HTML output.

`plot_nvml.py` writes `nvml_memory.html` by default and shows:

- one line per process for `nvml_total_used_gib`
- matching PyTorch `allocated` / `reserved` curves when present
- hover fields including `component`, `rank`, `pid`, `global_step`, and GPU index list

`plot_nvitop.py` writes `nvitop_resources.html` by default and shows:

- process RSS, CPU percent, and current-process GPU memory
- system memory and CPU percent
- global GPU memory and utilization, aggregated by time bucket
- a summary log at `<nvitop_dir>/nvitop_summary.log` with average/max GPU memory and GPU utilization

## Git hygiene

Because this directory is outside the RLinf repository, `git status` under
`/mnt/public/zengwen/RLinf` will not include it. If you create helper files
inside the RLinf worktree, put those paths in `.git/info/exclude` instead of
`.gitignore` so they remain local-only.
