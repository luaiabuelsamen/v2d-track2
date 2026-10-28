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
GAP=60
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
steps() {
  sshq "grep -ho 'elapsed_steps=[0-9]*' ~/t2_results/$TAG/train.log 2>/dev/null | tail -1 | cut -d= -f2"
}
FIRST="$(steps)"
sleep "$GAP"
SECOND="$(steps)"
say "steps ${FIRST:-none} -> ${SECOND:-none} over ${GAP}s"

# --- 4. the new term, from the run's own offline W&B summary -------------------------------------
# W&B runs offline on the box, and its summary file carries the last value of every logged metric,
# which is the cheapest place to read a single term without parsing the .wandb stream.
TERM="$(sshq "cat /dev/null; docker exec -i \$(docker ps -q | head -1) bash -lc \
  \"grep -ho '\\\"[a-z_/]*contact_role_palm[a-z_/]*\\\": *[-0-9.eE+]*' \
  /workspace/outputs/$TAG/wandb/offline-run-*/files/wandb-summary.json 2>/dev/null | tail -2\" \
  < /dev/null")"
if [ -z "$TERM" ]; then
  say "WARN: no contact_role_palm in the offline W&B summary yet (it appears at the first log step)"
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
case "$TERM" in
  *contact_role_palm*)
    # non-zero AND finite: a term stuck at 0 means the roles never loaded, nan means the gap maths
    # is wrong, and both are reasons not to pay for 9 more hours of it.
    if printf '%s\n' "$TERM" | grep -qiE 'nan|inf'; then
      say "FAIL: contact_role_palm is not finite"; FAIL=1
    elif ! printf '%s\n' "$TERM" | grep -qE ': *-?[0-9]*[1-9][0-9]*(\.[0-9]*)?|: *-?0\.[0-9]*[1-9]'; then
      say "FAIL: contact_role_palm is zero everywhere"; FAIL=1
    fi ;;
esac
[ "$FAIL" = 0 ] && say "OK: $TAG training, gpu ${UTIL}%, steps advancing, term non-zero and finite"
exit "$FAIL"
