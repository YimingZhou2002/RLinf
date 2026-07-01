# Enable RLinf timeline profiler.
#
# Source this script (do NOT execute) in the same shell that will launch RLinf:
#
#   source toolkits/embodied_tuner/profiler/enable_timeline.sh
#   bash examples/embodiment/run_embodiment.sh <config_name>
#
# Location-independent: derives its own directory from ${BASH_SOURCE[0]}.

_profiler_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

export PYTHONPATH="${_profiler_dir}:${PYTHONPATH}"
export RLINF_TIMELINE_DEBUG=1
export RLINF_TIMELINE=1
export RLINF_TIMELINE_DIR=auto
export RLINF_TIMELINE_WORKER_TIMER=1
export RLINF_TIMELINE_ACTOR_TRAINING=1
export RLINF_TIMELINE_PATCH_FILE="${_profiler_dir}/timeline_patches.embodied.txt"
unset RLINF_TIMELINE_PATCHES

unset _profiler_dir

# Then run the normal RLinf command in the same shell.
