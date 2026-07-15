#!/bin/bash
# Thin wrapper around batch_run.py. Launch the batch described in batch_runs.yaml:
#   bash toolkits/batch_run/run_batch.sh [path/to/another_batch_runs.yaml]
set -o pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
REPO_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH}

python "${SCRIPT_DIR}/batch_run.py" "${1:-${SCRIPT_DIR}/batch_runs.yaml}"
