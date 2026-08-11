#! /bin/bash
# Run the batch-size / micro-batch sweeps on the live rollout & actor workers
# (Step 2). Requires the Step-1 capture artifacts (run_capture.sh) so every
# non-batch dimension is taken from a real default run.
#
# Reuses run_embodiment.sh for all environment setup (EGL, ROBOT_PLATFORM, ...)
# but points the entry at benchmark_sweep.py.
#
# Usage:
#   RLINF_BENCH_SEED_DIR=/tmp/bench_msgs \
#     bash examples/embodiment/run_sweep.sh maniskill_ppo_openvla [ROBOT_PLATFORM]
#
# Optional knobs (env vars, all have sensible defaults):
#   RLINF_BENCH_ENV_GPUS     (default "0-1")   RLINF_BENCH_ROLLOUT_GPUS (default "2-3")
#   RLINF_BENCH_ACTOR_GPUS   (default "4-7")
#   RLINF_BENCH_MULTIPLIERS  (default "0.25,0.5,1,2,4")
#   RLINF_BENCH_WARMUP / RLINF_BENCH_REPEATS   RLINF_BENCH_OUT (default = seed dir)
#
# The three sweeps (env / rollout / actor) run concurrently on disjoint GPUs.
# The env sweep additionally profiles CPU% / host-RAM / GPU-util (env is CPU-heavy)
# and rebuilds the simulator per parallelism point.

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export SRC_FILE="${EMBODIED_PATH}/benchmark_sweep.py"
export RLINF_BENCH_SEED_DIR="${RLINF_BENCH_SEED_DIR:-/tmp/bench_msgs}"

CONFIG_NAME="${1:-maniskill_ppo_openvla}"

echo "Reading captures from ${RLINF_BENCH_SEED_DIR}; writing results to ${RLINF_BENCH_OUT:-${RLINF_BENCH_SEED_DIR}}"

exec bash "${EMBODIED_PATH}/run_embodiment.sh" "${CONFIG_NAME}" "${@:2}"
