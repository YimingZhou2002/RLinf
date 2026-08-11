#! /bin/bash
# Capture the intermediate messages exchanged between the embodied
# env / rollout / actor components during a single rollout step.
#
# Reuses run_embodiment.sh for all environment setup (EGL, ROBOT_PLATFORM, etc.)
# but points the entry at benchmark_capture.py and sets the capture output dir.
#
# Usage:
#   RLINF_BENCH_CAPTURE_DIR=/tmp/bench_msgs \
#     bash examples/embodiment/run_capture.sh maniskill_ppo_openvla [ROBOT_PLATFORM]
#
# Then inspect: cat /tmp/bench_msgs/*.schema.txt

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export SRC_FILE="${EMBODIED_PATH}/benchmark_capture.py"
export RLINF_BENCH_CAPTURE_DIR="${RLINF_BENCH_CAPTURE_DIR:-/tmp/bench_msgs}"

# Default to the openvla config if none is given.
CONFIG_NAME="${1:-maniskill_ppo_openvla}"

echo "Capturing messages to ${RLINF_BENCH_CAPTURE_DIR}"
mkdir -p "${RLINF_BENCH_CAPTURE_DIR}"

exec bash "${EMBODIED_PATH}/run_embodiment.sh" "${CONFIG_NAME}" "${@:2}"
