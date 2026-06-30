export PYTHONPATH=/mnt/public2/zhouyiming/humanize/rlinfutils:$PYTHONPATH
export RLINF_TIMELINE_DEBUG=1
export RLINF_TIMELINE=1
export RLINF_TIMELINE_DIR=auto
export RLINF_TIMELINE_WORKER_TIMER=1
export RLINF_TIMELINE_ACTOR_TRAINING=1
export RLINF_TIMELINE_PATCH_FILE=/mnt/public2/zhouyiming/humanize/rlinfutils/timeline_patches.embodied.txt
unset RLINF_TIMELINE_PATCHES


# Then run the normal RLinf command in the same shell.