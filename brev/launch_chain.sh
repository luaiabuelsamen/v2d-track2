#!/usr/bin/env bash
# Launch the Tier 1 generation-2 chain (docs/T1_CHAIN.md). One box, the adopted gen-1 clips,
# warm-started from gen 1's own checkpoints and trained 349,999,104 -> 399,998,976.
set -euo pipefail
S=/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad
cd /home/jetson3/projects/v2d-challenge
export V2D_SCRATCH=$S
N=$(grep -c . "$S/ws_chain/assets.warmstart.txt")
ASSETS=$S/ws_chain/assets \
PLAN=$S/ws_chain/assets.warmstart.txt \
OUT=$S/ws_chain/results \
NAME=${NAME:-v2d-t1chain} \
INSTANCE_TYPE=${INSTANCE_TYPE:-hyperstack_A6000} \
EXPECT_CLIPS=$N \
BOX_SCRIPT=run_warm_start.sh \
BOX_ENV="ARM_FLAGS=start_curriculum,orientation_reward,object_position V2D_OBJECT_POSITION_STD=0.08 V2D_OBJECT_ORIENTATION_STD=0.3 STEPS=399998976 WS_START=349999104" \
EXTRA_UPLOADS="brev/arm_flags.sh brev/patch_start_curriculum.py brev/patch_object_orientation.py brev/patch_object_position.py brev/patch_warm_start.py" \
WORK_LABEL="generation-2 fine-tunes" \
DEADLINE_H=${DEADLINE_H:-4.6} \
BOX_DEADLINE_H=${BOX_DEADLINE_H:-5.0} \
WATCHDOG_H=${WATCHDOG_H:-5.5} \
bash brev/t1_eval_clean.sh "$@"
