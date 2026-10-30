#!/usr/bin/env bash
# Adapter: brev/t1_eval_clean.sh hands its box script a PLAN env var, while run_tier2_full.sh takes
# the episode map positionally. Rather than bend either -- t1_eval_clean has brought up every box
# today and run_tier2_full has driven 30 clips -- this translates between them and nothing else.
#
# T1WEAK trains the eleven sub-0.35 Tier 1 clips from scratch on the CLEAN Tier 1 references, so
# ASSET_ROOT points at the flattened single-stage tree and no warm-start flag is passed: a clip
# below 0.35 has never adopted a warm start (0 of 4) and 12 of 12 rejected clips came back worse
# on MPPE, because 50M steps from a settled policy cannot relearn a clip it never solved.
set -uo pipefail
PLAN="${PLAN:?PLAN must point at an episode map of '<episode> <sequence_id>' lines}"
[ -s "$PLAN" ] || { echo "FATAL: empty or missing plan at $PLAN" >&2; exit 2; }
STEPS="${STEPS:-349999104}"
SAVE_EVERY="${SAVE_EVERY:-49999872}"
export ASSET_ROOT="${ASSET_ROOT:-/v2d/assets/t1weak}"
export RESULTS_DIR="${OUT:-$HOME/t1_eval_results}"

echo "T1WEAK: $(grep -c '[^[:space:]]' "$PLAN") clips, $STEPS steps, assets $ASSET_ROOT"
echo "        arm_flags=${ARM_FLAGS:-(stock)} seed=${V2D_SEED:-42}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_tier2_full.sh" "$PLAN" "$STEPS" "$SAVE_EVERY"
