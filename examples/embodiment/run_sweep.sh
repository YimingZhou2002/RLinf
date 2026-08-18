#! /bin/bash
# Run the batch-size / micro-batch sweeps on the live env / rollout / actor
# workers. Seeds (the intermediate env<->rollout<->actor messages that fix every
# non-batch dimension) are generated INLINE: no separate capture step is needed.
# When the seeds are missing, the harness first runs one short real env<->rollout
# round (a single chunk-step) on the already-launched workers to dump them, then
# runs the sweeps. run_capture.sh / benchmark_capture.py remain available as a
# standalone schema-inspection tool.
#
# Reuses run_embodiment.sh for all environment setup (EGL, ROBOT_PLATFORM, ...)
# but points the entry at benchmark_sweep.py.
#
# Usage:
#   RLINF_BENCH_SEED_DIR=/tmp/bench_msgs \
#     bash examples/embodiment/run_sweep.sh maniskill_ppo_openvla [ROBOT_PLATFORM]
#
# Optional knobs (env vars, all have sensible defaults):
#   RLINF_BENCH_CAPTURE      (default "auto"): auto | always | never
#                            auto=capture iff seeds missing; never=require seeds.
#   RLINF_BENCH_ENV_GPUS     (default "0-1")   RLINF_BENCH_ROLLOUT_GPUS (default "2-3")
#   RLINF_BENCH_ACTOR_GPUS   (default "4-7")
#   RLINF_BENCH_MULTIPLIERS  (default "0.25,0.5,1,2,4")
#   RLINF_BENCH_WARMUP / RLINF_BENCH_REPEATS   RLINF_BENCH_OUT (default = seed dir)
#
# Env-sweep bounding (the env axis is special -- see caveat below):
#   RLINF_BENCH_ENV_MULTIPLIERS  (default "0.25,0.5,1"): multipliers for the ENV
#                            sweep only (rollout/actor keep RLINF_BENCH_MULTIPLIERS).
#                            The default is down-sweep-only so the env sweep tops out
#                            at the production num_envs and never over-subscribes one
#                            GPU. Set e.g. "0.25,0.5,1,2,4" to force an up-sweep.
#   RLINF_BENCH_ENV_MAX      (default unset): hard absolute cap on num_envs for the
#                            env sweep. For RoboTwin set this to the largest size that
#                            builds safely (observed: 64) -- see caveat.
#
# CAVEAT (RoboTwin/SAPIEN): each env-sweep point builds a fresh num_envs-wide sim on
# a SINGLE GPU whose renderers leak Vulkan/OIDN TLS pthread-keys that are never
# reclaimed in-process. Large num_envs points accumulate keys until CPython aborts
# natively ("Couldn't create autoTSSkey mapping" / pthread_key_create failed) -- an
# UNCATCHABLE SIGABRT that kills the worker. Bounding the sweep (defaults above, or
# RLINF_BENCH_ENV_MAX) avoids it by not attempting the doomed build. As a safety net
# the env sweep also streams each measured row to
# "${RLINF_BENCH_OUT}/sweep_env_interact.partial.jsonl", so rows already measured
# survive on disk even if a later build still aborts.
#
# The three sweeps (env / rollout / actor) run concurrently on disjoint GPUs.
# The env sweep additionally profiles CPU% / host-RAM / GPU-util (env is CPU-heavy)
# and rebuilds the simulator per parallelism point.

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export SRC_FILE="${EMBODIED_PATH}/benchmark_sweep.py"
# Do NOT force a default here: when RLINF_BENCH_SEED_DIR is unset, benchmark_sweep.py
# defaults it to <runner.logger.log_path>/bench_msgs (a subfolder of this run's log
# dir). Forcing /tmp/bench_msgs here would shadow that. An explicit value is
# inherited from the environment and still honored.

CONFIG_NAME="${1:-maniskill_ppo_openvla}"

if [ -n "${RLINF_BENCH_SEED_DIR}" ]; then
  echo "Reading captures from ${RLINF_BENCH_SEED_DIR}; writing results to ${RLINF_BENCH_OUT:-${RLINF_BENCH_SEED_DIR}}"
else
  echo "RLINF_BENCH_SEED_DIR unset; using <log_dir>/bench_msgs (see benchmark_sweep.py)"
fi

exec bash "${EMBODIED_PATH}/run_embodiment.sh" "${CONFIG_NAME}" "${@:2}"
