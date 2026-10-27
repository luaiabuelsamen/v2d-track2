#!/usr/bin/env bash
# One watcher over the whole second wave: pull finished clips, kit-score them, decide adoption.
#
# The group launchers already pull every 10 min and score each clip as it lands, so this is not a
# second copy of that loop -- it is the layer above it, and it exists for three things the
# launchers cannot do:
#
#   1. The cross-wave decision. Adoption compares a new rollout against the one on the Kaggle
#      board, which no single launcher can see; `v2d adopt-check` needs every group at once.
#   2. Resilience. On 2026-10-25 a run_clip loop logged "interim pull: 0 trajectories" for 80 min
#      after its runner had died. If a launcher's loop is gone, its box still holds finished
#      clips, so this pulls any complete clip the local tree is missing.
#   3. One record, committed. results/second_wave_results.md and
#      results/second_wave_adoption.json are rewritten and committed as each clip lands, so the
#      state of the wave is in git rather than in a scrollback.
#
# It shares the launchers' trees rather than copying them: scoring goes through
# brev/score_pulled.sh with the same --results dir, which means the same flock, which means two
# passes never convert into the same npz directory. second_wave/results/<arm> is a symlink to
# run_<arm>/results so the requested layout exists over one copy of the data, on a disk with
# 23 GB free and a 20 GB floor.
#
# It never creates, restarts or deletes a box. A dead box is reported here and handled by its
# own launcher and watchdog.
#
#   brev/second_wave_puller.sh                  # loop until the wave is in
#   brev/second_wave_puller.sh --once           # a single pass
set -uo pipefail
REPO=/home/jetson3/projects/v2d-challenge
cd "$REPO"
S="${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}"
WAVE="$S/second_wave"
BREV=~/.local/bin/brev
PY="env -u PYTHONPATH ./.venv311/bin/python -I"
run_v2d() { $PY -c "import sys; sys.argv=['v2d']+sys.argv[1:]; from v2d.cli import main; raise SystemExit(main())" "$@"; }

# arm -> stage dir; the results dir is $WAVE/run_<arm>, the box prefix v2d-sw-<lowercase arm>-b.
ARMS="A B1 B2 B3 B4 B5 B4x"
LAST_STATE=""
FINAL_CHECKPOINT=349999104
INTERVAL="${V2D_PULL_INTERVAL:-600}"
ONCE=""
case "${1:-}" in
  --once) ONCE=1 ;;
esac

stamp() { date -u +%H:%M:%S; }
say() { echo "$(stamp) $*"; }

# v2d-sw-a-b03 -> A, v2d-sw-b1-b01 -> B1, v2d-sw-b4x-b01 -> B4x. Getting this wrong would score a
# clip against another arm's stage, so it is a named function with a test rather than an inline
# sed. The comparison is case-insensitive and returns the arm's CANONICAL spelling: box names are
# lower case, and B4x is not "B4X" -- its stage dir, map and results dir all carry the small x.
arm_of() {
  local want candidate
  want="$(printf '%s' "${1#v2d-sw-}" | sed 's/-b[0-9]*$//' | tr 'A-Z' 'a-z')"
  [ -n "$want" ] || return 1
  for candidate in $ARMS; do
    if [ "$(printf '%s' "$candidate" | tr 'A-Z' 'a-z')" = "$want" ]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

# How many clips this watch is waiting for: whatever the staged maps say, so adding an arm (B4x
# was added mid-watch) does not leave the loop waiting on a number written by hand.
expected_clips() {
  local arm n=0
  for arm in $ARMS; do
    [ -f "$WAVE/map_$arm.txt" ] || continue
    n=$(( n + $(grep -c '[^[:space:]]' "$WAVE/map_$arm.txt") ))
  done
  echo "$n"
}

# The clips a box has finished: a clip dir holding the final checkpoint's metrics.json. Anything
# earlier is still being written, and an interim pull already has it.
finished_clips_on() {
  ssh -n -o ConnectTimeout=25 -o BatchMode=yes "$1" \
    "ls -d ~/t2_results/e[0-9][0-9]/policy_$FINAL_CHECKPOINT/metrics.json 2>/dev/null" 2>/dev/null \
    | sed 's#.*/t2_results/##; s#/policy_.*##'
}

# A self-test entry point: `--arm-of v2d-sw-b1-b01` prints B1 and exits, so the mapping can be
# checked without a box, a launcher or an ssh.
if [ "${1:-}" = "--arm-of" ]; then
  arm_of "${2:-}" || { echo "no arm owns ${2:-}" >&2; exit 1; }
  echo
  exit 0
fi

EXPECTED_CLIPS="$(expected_clips)"

live_boxes() {
  # One brev ls per pass. A transient empty answer must not read as "every box is gone", so an
  # empty result is reported as unknown and the caller keeps going.
  $BREV ls 2>/dev/null | awk 'NR>2 && $1 ~ /^v2d-sw-/ {print $1, $2, $3}'
}

pull_missing() {
  # $1 arm, $2 box: bring home any finished clip whose final checkpoint is not local yet.
  local ARM="$1" NAME="$2" RESULTS="$WAVE/run_$1/results" TAG
  for TAG in $(finished_clips_on "$NAME"); do
    [ -f "$RESULTS/$TAG/policy_$FINAL_CHECKPOINT/metrics.json" ] && continue
    say "   $NAME $TAG: final checkpoint is on the box but not here -- pulling"
    mkdir -p "$RESULTS/$TAG"
    for attempt in 1 2 3; do
      rsync -a -c --partial --timeout=180 "$NAME:t2_results/$TAG/" "$RESULTS/$TAG/" \
        >/dev/null 2>&1 && break
      sleep 10
    done
    local got; got=$(find "$RESULTS/$TAG" -name objects.parquet 2>/dev/null | wc -l)
    say "   $NAME $TAG: $got trajectories local"
  done
}

score_arm() {
  # $1 arm. score_pulled.sh picks the z-drop checkpoint first and only then kit-scores, so the
  # choice provably cannot have been made on the score; see its header.
  local ARM="$1" STAGE="$S/sw_$1" RESULTS="$WAVE/run_$1/results"
  [ -d "$RESULTS" ] || return 0
  ls -d "$RESULTS"/e[0-9][0-9] >/dev/null 2>&1 || return 0
  local NPZ_ARGS=()
  while read -r flag item; do
    [ -n "${item:-}" ] && NPZ_ARGS+=("$flag" "$item")
  done < <($PY -c "
import json
stage = json.load(open('$STAGE/provenance.json'))
for label, path in sorted(stage.get('npz_by_source', {}).items()):
    print(f'--npz {label}={path}')
for label, path in sorted(stage.get('ref_dataset_by_source', {}).items()):
    print(f'--ref-dataset {label}={path}')
" 2>/dev/null)
  V2D_SCORE_LOCK_WAIT=60 bash "$REPO/brev/score_pulled.sh" --results "$RESULTS" \
    --provenance "$STAGE/provenance.json" "${NPZ_ARGS[@]+${NPZ_ARGS[@]}}" 2>&1 | sed 's/^/   /'
}

publish_layout() {
  # The requested path, over one copy of the data.
  mkdir -p "$WAVE/results"
  local ARM
  for ARM in $ARMS; do
    [ -d "$WAVE/run_$ARM/results" ] || continue
    [ -e "$WAVE/results/$ARM" ] || ln -s "../run_$ARM/results" "$WAVE/results/$ARM"
  done
}

# What the decision depends on: which checkpoint each clip selected, and which scores exist. The
# decision re-reads two parquets per clip, so on a 13 h watch it is worth not redoing it 80 times
# over an unchanged tree -- this is a Jetson with a two-worker budget.
state_fingerprint() {
  find "$WAVE"/run_*/results -maxdepth 3 \( -name selection.json -o -name score.json \) \
    -printf '%p %T@ %s\n' 2>/dev/null | sort | md5sum | cut -d' ' -f1
}

decide_and_commit() {
  local ARM_GROUPS=() ARM
  # $ARMS is ordered deliberately: B4x last, because it is judged against the others' winners.
  for ARM in $ARMS; do
    [ -d "$WAVE/run_$ARM/results" ] && ARM_GROUPS+=(--group "$ARM=$WAVE/run_$ARM/results")
  done
  [ "${#ARM_GROUPS[@]}" -gt 0 ] || return 0
  # Group order IS the chain order: the registered wave's arms first, then B4x, which re-runs six
  # Part A clips and one B3 clip. A B4x rollout therefore has to beat what Part A or B3 put on the
  # board, not the first-wave record it would also beat (docs/SECOND_WAVE.md).
  #
  # e01 is the one clip whose reference changed, so it is measured against the reference the
  # shipped entry used. docs/SECOND_WAVE.md registers that caveat; adopt-check enforces it.
  run_v2d adopt-check "${ARM_GROUPS[@]}" \
    --old "$S/t2_combined_run/results" \
    --record results/t2_combined_run_record.json \
    --reference-from-old e01 \
    -o results/second_wave_adoption.json \
    --markdown results/second_wave_results.md 2>&1 | tail -n +1 | sed 's/^/   /'

  git add results/second_wave_adoption.json results/second_wave_results.md >/dev/null 2>&1
  git diff --cached --quiet results/second_wave_adoption.json results/second_wave_results.md \
    && return 0
  local LANDED
  LANDED=$($PY -c "
import json
print(json.load(open('results/second_wave_adoption.json'))['summary']['clips_landed'])
" 2>/dev/null || echo "?")
  # --only those two paths: this runs in the background for hours, and must never sweep up
  # whatever else happens to be staged in the working tree at the time.
  git commit -q --only results/second_wave_adoption.json results/second_wave_results.md -F - <<EOF
Second wave, $LANDED of $EXPECTED_CLIPS clips in

Written by brev/second_wave_puller.sh as the clips land. The adopt column is the registered
input-only rule; the kit columns are reported, not read by the decision.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
  say "   committed: $(git log -1 --format=%h%x20%s)"
}

landed_count() {
  local ARM N=0
  for ARM in $ARMS; do
    N=$((N + $(ls -d "$WAVE/run_$ARM/results"/e[0-9][0-9]/policy_$FINAL_CHECKPOINT/metrics.json \
      2>/dev/null | wc -l)))
  done
  echo "$N"
}

pass() {
  local ROWS BOXES=() NAME STATE BUILD ARM
  ROWS="$(live_boxes)"
  if [ -z "$ROWS" ]; then
    say "brev ls returned no v2d-sw-* row; treating as unknown, not as 'all gone'"
  else
    while read -r NAME STATE BUILD; do
      [ -n "$NAME" ] || continue
      case "$STATE$BUILD" in
        *FAILURE*|*TERMINAT*) say "!! $NAME is $STATE $BUILD -- its launcher owns the restart, not me" ;;
        *) BOXES+=("$NAME") ;;
      esac
    done <<< "$ROWS"
  fi

  for NAME in ${BOXES[@]+"${BOXES[@]}"}; do
    ARM="$(arm_of "$NAME")" || { say "!! $NAME: no arm of this wave owns that name, skipped"; continue; }
    pull_missing "$ARM" "$NAME"
  done

  for ARM in $ARMS; do score_arm "$ARM"; done
  publish_layout
  local NOW; NOW="$(state_fingerprint)"
  if [ "$NOW" != "${LAST_STATE:-}" ]; then
    LAST_STATE="$NOW"
    decide_and_commit
  fi
  say "pass done: $(landed_count) of $EXPECTED_CLIPS clips have a final checkpoint, "\
"${#BOXES[@]} boxes live"
}

say "second-wave puller: $EXPECTED_CLIPS clips expected, every ${INTERVAL}s, no launches"
while :; do
  pass
  [ -n "$ONCE" ] && break
  if [ "$(landed_count)" -ge "$EXPECTED_CLIPS" ]; then
    say "all $EXPECTED_CLIPS clips are in; final pass done, stopping"
    break
  fi
  if [ -z "$(live_boxes)" ] && [ "$(landed_count)" -gt 0 ]; then
    # Two empty readings 60 s apart, because one transient brev ls must not end the watch.
    sleep 60
    if [ -z "$(live_boxes)" ]; then
      say "no v2d-sw-* box left and $(landed_count) of $EXPECTED_CLIPS clips in; stopping"
      break
    fi
  fi
  sleep "$INTERVAL"
done
