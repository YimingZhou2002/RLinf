# Enable RLinf NVML sampling.
#
# Source this script (do NOT execute) in the same shell that will launch RLinf:
#
#   source toolkits/embodied_tuner/profiler/enable_nvml.sh
#   bash examples/embodiment/run_embodiment.sh <config_name>
#
# Location-independent: derives its own directory from ${BASH_SOURCE[0]}.

_profiler_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PYTHONPATH="${_profiler_dir}:${PYTHONPATH}"
export RLINF_NVITOP=1
export RLINF_NVITOP_INTERVAL=5

unset _profiler_dir

# Then run the normal RLinf command in the same shell.
