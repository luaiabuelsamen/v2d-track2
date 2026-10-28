#!/usr/bin/env bash
# Verify one arm-CR box the way the brief asks: GPU busy, steps moving, and the new reward term
# actually non-zero and finite in the run's own log -- plus the one failure that would mean the
# arm is wrong rather than slow, `build()` refusing the object radius.
#
#   brev/verify_cr_box.sh v2d-sw-cr-b01 [--gap 60]
#
# Exit 0 = all four checks pass. Exit 4 = build() died on the radius (STOP EVERYTHING). Exit 1 =
# some check failed or could not be read, which is a reason to look rather than to delete.
set -uo pipefail

NAME="${1:?usage: verify_cr_box.sh <box> [--gap SECONDS]}"
shift || true
# The trainer logs "training progress" once per 1% of the budget -- 3,500,032 steps, about 2.5
# minutes at the 22.8k steps/s these boxes run at -- so a 60 s gap reads the same number twice on a
# perfectly healthy box and calls it stalled. The default has to be longer than one log interval.
GAP=210
while [ $# -gt 0 ]; do
  case "$1" in
    --gap) GAP="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

sshq() { ssh -n -o ConnectTimeout=20 -o BatchMode=yes "$NAME" "$1" 2>/dev/null; }

say() { printf '%-16s %s\n' "$NAME" "$*"; }

# The clip this box is on, from the newest per-clip log the runner opened.
TAG="$(sshq "ls -t ~/t2/clip_e*.log 2>/dev/null | head -1 | sed 's#.*/clip_##; s#\.log##'")"
[ -n "$TAG" ] || { say "no clip log yet -- the runner has not started a clip"; exit 1; }

# --- the failure that stops the whole arm -------------------------------------------------------
# `build()` raises before training if CommandBuffers has no object_radius, or its value is not
# finite. That is the one error that means the patch is wrong about the image, so it is checked
# first and reported verbatim.
RADIUS_ERROR="$(sshq "grep -h -m1 -A3 'contact-role arm needs' ~/t2/clip_$TAG.log \
  ~/t2_results/$TAG/train.log 2>/dev/null")"
if [ -n "$RADIUS_ERROR" ]; then
  say "BUILD FAILED on the object radius:"
  printf '%s\n' "$RADIUS_ERROR" | sed 's/^/    /'
  exit 4
fi

# --- 1. the patch, and the roles the objective is reading ---------------------------------------
PATCHED="$(sshq "grep -h -m1 'contact-role arm:' ~/t2/clip_$TAG.log")"
ROLES="$(sshq "grep -h -m1 'contact roles from' ~/t2/clip_$TAG.log")"
[ -n "$PATCHED" ] && say "patch: ${PATCHED#*| }" || say "WARN: no contact-role patch line in clip_$TAG.log"
[ -n "$ROLES" ] && say "roles: ${ROLES#*: }" || say "WARN: no contact-roles line in clip_$TAG.log"

# --- 2. GPU utilisation and the live train process ----------------------------------------------
UTIL="$(sshq "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1")"
PROCS="$(sshq "pgrep -fc train_flash_sac")"
say "gpu ${UTIL:-?}%  train processes ${PROCS:-0}"

# --- 3. the step counter, twice, GAP seconds apart ----------------------------------------------
# `training progress: 10,500,096/349,999,104 environment steps (3.0%); 22,843 env steps/s; ...`
steps() {
  sshq "grep -ho 'progress: [0-9,]*' ~/t2_results/$TAG/train.log 2>/dev/null | tail -1 \
    | sed 's/progress: //; s/,//g'"
}
FIRST="$(steps)"
sleep "$GAP"
SECOND="$(steps)"
say "steps ${FIRST:-none} -> ${SECOND:-none} over ${GAP}s"

# --- 4. the new term, from the run's own offline W&B summary -------------------------------------
# W&B runs offline on the box, and its summary file carries the last value of every logged metric,
# which is the cheapest place to read a single term without parsing the .wandb stream.
# Through the .wandb stream, not files/wandb-summary.json: offline W&B only writes the summary
# when the run closes, so while training is alive the summary file is empty or absent.
READER="$(dirname "${BASH_SOURCE[0]}")/read_wandb_terms.py"
scp -q -o ConnectTimeout=20 "$READER" "$NAME:/tmp/read_wandb_terms.py" 2>/dev/null
TERM="$(ssh -n -o ConnectTimeout=40 -o BatchMode=yes "$NAME" \
  'CID=$(docker ps -q | head -1);
   docker exec -i $CID bash -lc "cat > /tmp/read_wandb_terms.py" < /tmp/read_wandb_terms.py;
   docker exec -i $CID bash -lc "python /tmp/read_wandb_terms.py \"/workspace/outputs/t2_'"$TAG"'/wandb/offline-run-*/run-*.wandb\"" < /dev/null' \
  2>/dev/null | grep -E 'contact_role_palm|contact_consistency')"
if [ -z "$TERM" ]; then
  say "WARN: no contact_role_palm logged yet -- the first W&B history record lands a minute or two
                 after the kernels finish compiling; re-run then rather than reading this as a failure"
else
  printf '%s\n' "$TERM" | sed "s/^/$(printf '%-16s' "$NAME") term: /"
fi

FAIL=0
awk -v u="${UTIL:-0}" 'BEGIN{exit !(u >= 90)}' || { say "FAIL: gpu ${UTIL:-?}% is under 90"; FAIL=1; }
[ "${PROCS:-0}" -ge 1 ] || { say "FAIL: no train_flash_sac process"; FAIL=1; }
if [ -z "$FIRST" ] || [ -z "$SECOND" ]; then
  say "FAIL: could not read the step counter"; FAIL=1
elif [ "$SECOND" -le "$FIRST" ]; then
  say "FAIL: step counter did not advance"; FAIL=1
fi
# Numerically, not by pattern: the value arrives as a quoted JSON string, and a regex that
# "proved" 0.0178 was zero is worse than no check -- it is a check that reports the wrong answer.
case "$TERM" in
  "")
    say "FAIL: could not read contact_role_palm at all -- not the same thing as reading a good one"
    FAIL=1 ;;
  *contact_role_palm*)
    VALUE="$(printf '%s\n' "$TERM" | grep contact_role_palm | tail -1 \
      | sed 's/.*: *"*//; s/"*,*$//')"
    # nan and inf are tested as TEXT first: this awk coerces "nan" to 0, so a numeric-only test
    # reports a NaN term as merely zero and the operator fixes the wrong thing.
    VERDICT="$(awk -v v="$VALUE" 'BEGIN{
      lower = tolower(v)
      if (lower ~ /nan|inf/) { print "not finite"; exit }
      if (v+0 > 1e30 || v+0 < -1e30) { print "not finite"; exit }
      if (v+0 == 0) { print "zero" } else { print "ok" }
    }')"
    case "$VERDICT" in
      ok) say "contact_role_palm = $VALUE (non-zero, finite)" ;;
      zero) say "FAIL: contact_role_palm is zero ($VALUE) -- the roles did not load"; FAIL=1 ;;
      *) say "FAIL: contact_role_palm is not finite ($VALUE)"; FAIL=1 ;;
    esac ;;
  *)
    say "FAIL: no contact_role_palm in the term read"; FAIL=1 ;;
esac
[ "$FAIL" = 0 ] && say "OK: $TAG training, gpu ${UTIL}%, steps advancing, contact_role_palm non-zero and finite"
exit "$FAIL"
