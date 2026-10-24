#!/usr/bin/env bash
# Run the reference-cleanup arms pre-registered in docs/EXP_CLEANUP_CONTACT.md.
#
# Every arm keeps Tier 2's noisy human parameters, because that is what a real pipeline
# receives; only the object trajectory differs. Arms run in the order the decision rule needs
# them, so an interruption still leaves the question answerable:
#
#   tilt_fixed    the cleanup under test
#   rot_clean     ceiling for an orientation-only cleaner (the recovery denominator)
#   yaw_fixed     the control that should not work
#   object_clean  whole-object ceiling, for P1 only
#
# 2 workers is the ceiling on this 15.6 GB box; 3 lost 28 clips to OOM kills.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${1:-}"
if [ -z "$SCRATCH" ]; then
  echo "usage: $0 <scratch-dir>   # the one scripts/setup_jetson_env.sh used" >&2
  exit 2
fi
REPO="$SCRATCH/repo_src"
SOMA="$REPO/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma"
NPZ="$SCRATCH/npz3"
PY="$PROJECT/.venv311/bin/python"

for spec in \
  "tilt_fixed:--object-npz:$NPZ/tilt_fixed" \
  "rot_clean:--object-npz:$NPZ/rot_fixed" \
  "yaw_fixed:--object-npz:$NPZ/yaw_fixed" \
  "object_clean:--object-from-tier:tier1"
do
  arm="${spec%%:*}"; rest="${spec#*:}"; flag="${rest%%:*}"; value="${rest#*:}"
  base="$SCRATCH/arms/$arm"
  printf '\n===== arm %s (%s %s) =====\n' "$arm" "$flag" "$value"
  # Each arm keeps its own tree so nothing collides and each is independently resumable.
  env -u PYTHONPATH "$PY" -I "$PROJECT/scripts/retarget_all.py" \
    --dataset-root "$SCRATCH/data" --repo "$REPO" --soma-root "$SOMA" \
    --seq-root "$base/seq" --out-root "$base/motion" \
    --log-dir "$base/logs" --validation-dir "$base/validation" \
    --tiers tier2 --workers 2 --threads 2 "$flag" "$value" \
    -o "$PROJECT/results/arm_$arm.json" || echo "arm $arm exited non-zero; see $base/logs"
done

printf '\n===== all arms done =====\n'
for arm in tilt_fixed rot_clean yaw_fixed object_clean; do
  python3 -c "
import json,sys
d=json.load(open('$PROJECT/results/arm_$arm.json'))['summary']
print(f\"$arm: {d['episodes_passed']}/{d['episodes_total']} passed, worst residual {d['worst_rigid_fit_residual_mm']} mm\")
" 2>/dev/null || echo "$arm: no results file"
done
