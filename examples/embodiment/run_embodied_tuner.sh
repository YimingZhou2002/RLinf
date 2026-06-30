#!/usr/bin/env bash
# Shim launcher for the embodied auto-tuner CLI.
#
# Wraps `python -m toolkits.embodied_tuner` with the PYTHONPATH /
# REPO_PATH setup the toolkit needs, matching the convention used by
# examples/embodiment/run_placement_autotune.sh.

set -euo pipefail

EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"

export REPO_PATH
export EMBODIED_PATH
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

if ! python -c "import toolkits.embodied_tuner" >/dev/null 2>&1; then
    cat >&2 <<MSG
embodied_tuner shim: cannot import 'toolkits.embodied_tuner' under PYTHONPATH=${PYTHONPATH}.
This usually means the toolkit directory is missing or PYTHONPATH did not pick up REPO_PATH=${REPO_PATH}.
Try running:
    cd ${REPO_PATH}
    PYTHONPATH=${REPO_PATH} python -m toolkits.embodied_tuner --help
MSG
    exit 4
fi

exec python -m toolkits.embodied_tuner "$@"
