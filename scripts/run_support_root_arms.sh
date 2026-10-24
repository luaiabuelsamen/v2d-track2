#!/usr/bin/env bash
# Run the arms pre-registered in docs/EXP_SUPPORT_AND_ROOT.md.
#
# Every arm keeps Tier 2's inputs; only what has been cleaned differs:
#
#   table        object pose cleaned against the floor AND the released support primitives
#   table_root   the same, plus the foot-planting root correction -- the candidate reference set
#   root         the root correction alone, on the released Tier 2 object pose, for attribution
#
# Ordered so an interruption still leaves the question answerable: `table_root` minus `table` is
# the paired root contribution R2 asks for, and `root` alone only refines the attribution.
#
# 2 workers is the ceiling on this 15.6 GB box; 3 lost 28 clips to OOM kills, and even 2 lost 2
# clips while other work competed for memory. Re-run stragglers with --workers 1; the driver is
# resumable, so a full re-run redoes only the failures and rewrites the complete summary.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${1:-}"
WORKERS="${2:-2}"
if [ -z "$SCRATCH" ]; then
  echo "usage: $0 <scratch-dir> [workers]   # the one scripts/setup_jetson_env.sh used" >&2
  exit 2
fi
REPO="$SCRATCH/repo_src"
SOMA="$REPO/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma"
PY="$PROJECT/.venv311/bin/python"

for spec in \
  "table:--object-npz $SCRATCH/npz_table" \
  "table_root:--object-npz $SCRATCH/npz_table --human-npz $SCRATCH/npz_root" \
  "root:--human-npz $SCRATCH/npz_root"
do
  arm="${spec%%:*}"; flags="${spec#*:}"
  base="$SCRATCH/arms/$arm"
  printf '\n===== arm %s (%s) =====\n' "$arm" "$flags"
  # shellcheck disable=SC2086  # $flags is a deliberate word list
  env -u PYTHONPATH "$PY" -I "$PROJECT/scripts/retarget_all.py" \
    --dataset-root "$SCRATCH/data" --repo "$REPO" --soma-root "$SOMA" \
    --seq-root "$base/seq" --out-root "$base/motion" \
    --log-dir "$base/logs" --validation-dir "$base/validation" \
    --tiers tier2 --workers "$WORKERS" --threads 2 $flags \
    -o "$PROJECT/results/arm_$arm.json" || echo "arm $arm exited non-zero; see $base/logs"
done

printf '\n===== all arms done =====\n'
for arm in table table_root root; do
  "$PY" -I -c "
import json
d = json.load(open('$PROJECT/results/arm_$arm.json'))['summary']
print(f\"$arm: {d['episodes_passed']}/{d['episodes_total']} passed, \"
      f\"worst residual {d['worst_rigid_fit_residual_mm']} mm, counts {d['counts']}\")
" 2>/dev/null || echo "$arm: no results file"
done
