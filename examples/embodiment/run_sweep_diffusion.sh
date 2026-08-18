#! /bin/bash
# Run the batch-size / micro-batch sweeps for the diffusion model (NFT-based
# embodied training). Mirrors run_sweep.sh but uses run_diffusion.sh so that
# the diffusion config path and model-specific env vars are set correctly.
#
# Usage:
#   RLINF_BENCH_SEED_DIR=/tmp/bench_msgs \
#     bash examples/embodiment/run_sweep_diffusion.sh wan22_ti2v_5b_nft_video_ocr
#
# Optional knobs (env vars, all have sensible defaults):
#   RLINF_BENCH_CAPTURE      (default "auto"): auto | always | never
#   RLINF_BENCH_ENV_GPUS     (default "0-1")   RLINF_BENCH_ROLLOUT_GPUS (default "2-3")
#   RLINF_BENCH_ACTOR_GPUS   (default "4-7")
#   RLINF_BENCH_MULTIPLIERS  (default "0.25,0.5,1,2,4")
#   RLINF_BENCH_WARMUP / RLINF_BENCH_REPEATS   RLINF_BENCH_OUT (default = seed dir)
#   RLINF_BENCH_ENV_MAX      (default unset): hard absolute cap on num_envs for the
#                            env sweep. For diffusion models the env axis is less
#                            prone to the SAPIEN TLS-key SIGABRT, but still useful
#                            to bound if the GPU is 80 GiB-class.

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export DIFFUSION_PATH="$(dirname "$EMBODIED_PATH")/diffusion"
export SRC_FILE="${EMBODIED_PATH}/benchmark_sweep.py"

CONFIG_NAME="${1:-wan22_ti2v_5b_nft_video_ocr}"

if [ -n "${RLINF_BENCH_SEED_DIR}" ]; then
  echo "Reading captures from ${RLINF_BENCH_SEED_DIR}; writing results to ${RLINF_BENCH_OUT:-${RLINF_BENCH_SEED_DIR}}"
else
  echo "RLINF_BENCH_SEED_DIR unset; using <log_dir>/bench_msgs (see benchmark_sweep.py)"
fi

exec bash "${DIFFUSION_PATH}/run_diffusion.sh" "${CONFIG_NAME}" "${@:2}"