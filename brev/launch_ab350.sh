#!/usr/bin/env bash
# Orchestrate the six cells of the 3-clip full-budget A/B: one cell per box, N boxes at a time.
#
# Read-only without --go. The bring-up is not reimplemented here: it calls
# brev/gate_launch.sh --phase build --dev-box, which created all three boxes running today and
# already handles the parts that bit us -- watchdog armed BEFORE create, creation judged by
# brev ls rather than create's exit code, SSH readiness proved by an actual ssh, resolved G1
# meshes shipped to dodge the LFS problem.
#
# Per cell: bring up, stage that arm's reference and verify it by md5, run one cell, pull, delete.
# --max-concurrent exists because this org is only MEASURED to run 4 A6000s at once; 6 is
# untested, and a refused create must queue rather than lose the cell.
set -uo pipefail

GO=0
MAX_CONCURRENT=6
TYPE=massedcompute_A6000_plus
MAX_HOURS=7                 # ~4.8 h of work; the in-box watchdog must not fire mid-run
BACKSTOP_HOURS=7.5          # my own, independent of anything gate_launch.sh does
STEPS=349999104
CLIPS="3 9 18"
ARMS="raw clean"      # step 1 is the RAW baseline alone; CLEAN is held until it verifies
while [ $# -gt 0 ]; do
  case "$1" in
    --go) GO=1; shift ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --type) TYPE="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --clips) CLIPS="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

S="${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}"
REPO=/home/jetson3/projects/v2d-challenge
OUT=$S/ab350
# One id per launcher invocation, stamped into each cell directory so a reader can tell this run's
# results from a previous run's at the same path.
RUN_ID="ab350-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$OUT"
declare -A SEQ=(
  [3]=2026-04-27_11-50-10_contigo_coffee_mug_handle_table_05
  [9]=2026-06-04_16-45-04_big_red_bowl_under_skinny_wood_chair_03
  [18]=2026-05-06_10-00-19_blue_trash_can_drag_sideways_08
)

CELLS=()
for EP in $CLIPS; do for ARM in $ARMS; do CELLS+=("$ARM:$EP"); done; done

# Refuse to start on references that are not staged and verified. The arm IS the parquet, so an
# unstaged or half-copied reference does not fail loudly, it trains the wrong arm.
for ARM in $ARMS; do
  [ -s "$S/ab_stage_${ARM}.md5" ] || { echo "FATAL: $S/ab_stage_${ARM}.md5 missing; run stage_ab_refs.sh" >&2; exit 2; }
  for EP in $CLIPS; do
    P="$S/ab_stage_${ARM}/whole_body/soma/sequence_id=${SEQ[$EP]}/robot_name=g1/data.parquet"
    [ -s "$P" ] || { echo "FATAL: no staged $ARM reference for e$EP at $P" >&2; exit 2; }
  done
done

EST_HOURS=4.8
TOTAL_USD="$(awk -v n="${#CELLS[@]}" -v h="$EST_HOURS" 'BEGIN{printf "%.2f", n*h*0.68}')"
cat <<PLAN

== Plan: ${#CELLS[@]} cells, $MAX_CONCURRENT at a time, $TYPE at \$0.68/h
   cells        : ${CELLS[*]}
   steps/cell   : $STEPS   (checkpoint every 49999872)
   est per cell : ${EST_HOURS} h  -> projected total \$${TOTAL_USD}
   watchdogs    : in-box ${MAX_HOURS} h, my backstop ${BACKSTOP_HOURS} h, per box
   arms         : $ARMS
   references   : $(for a in $ARMS; do printf "%s=%s " "$a" "$(wc -l < "$S/ab_stage_${a}.md5")"; done)files, md5-verified after transfer
   decision rule: docs/AB350_BAR.md  (not editable after this runs)
PLAN
[ "$GO" = 1 ] || { echo "
Read-only. Re-run with --go to create instances."; exit 0; }

run_cell() {
  local ARM="${1%%:*}" EP="${1##*:}"
  local NAME="v2d-ab-${ARM}-e$(printf '%02d' "$EP")"
  local LOG="$OUT/${NAME}.log"
  local SEQID="${SEQ[$EP]}"
  {
    echo "=== $NAME: arm $ARM, e$EP, $SEQID ($(date -u +%H:%M:%S)) ==="
    setsid nohup bash "$S/backstop_watchdog.sh" "$NAME" \
      "$(awk -v h="$BACKSTOP_HOURS" 'BEGIN{printf "%d", h*3600}')" "$S/backstop.log" >/dev/null 2>&1 &

    bash "$REPO/brev/gate_launch.sh" --phase build --dev-box --name "$NAME" \
         --type "$TYPE" --max-hours "$MAX_HOURS" --go || { echo "!! bring-up failed for $NAME"; return 1; }

    # rsync will not create intermediate directories, which cost a transfer earlier today.
    ssh -n "$NAME" "mkdir -p ~/v2d_assets/ab_${ARM} ~/ab" 2>/dev/null
    local ok=0
    for attempt in 1 2 3 4 5 6; do
      rsync -a -c --partial --timeout=120 -e 'ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=4' \
        "$S/ab_stage_${ARM}/" "$NAME:v2d_assets/ab_${ARM}/" >/dev/null 2>&1
      # By md5, not by file count: a truncated parquet passes a count and fails with
      # "Parquet magic bytes not found in footer" four hours into a run.
      if ssh -n "$NAME" "cd ~/v2d_assets/ab_${ARM} && find . -type f | sort | xargs md5sum" 2>/dev/null \
           | grep -E '^[0-9a-f]{32} ' | sort -k2 > "$OUT/${NAME}.remote.md5" \
         && diff -q <(sort -k2 "$S/ab_stage_${ARM}.md5") "$OUT/${NAME}.remote.md5" >/dev/null; then
        ok=1; echo "reference verified by md5 on attempt $attempt"; break
      fi
      sleep 5
    done
    [ "$ok" = 1 ] || { echo "!! $NAME: reference never matched by md5; NOT training"; return 1; }

    # Count the deletion authorities and check the earliest one. gate_launch.sh arms a DETACHED
    # host-side watchdog, which is a second authority on top of my backstop -- and on 2026-10-24
    # that is exactly what deleted the sweep box at 3 h with 8 of 10 runs done and nothing pulled,
    # because I extended my backstop and never looked at the launcher's. Mine must fire LAST.
    local earliest
    earliest=$(ps -eo etimes,cmd 2>/dev/null | grep -F "brev delete '$NAME'" | grep -v grep \
      | awk '{for(i=1;i<=NF;i++) if($i=="sleep"){print $(i+1)-$1; break}}' | sort -n | head -1)
    if [ -n "$earliest" ]; then
      echo "earliest deletion authority for $NAME fires in $((earliest/60)) min"
      if [ "$earliest" -lt "$(awk -v h="$EST_HOURS" 'BEGIN{printf "%d", h*3600*1.1}')" ]; then
        echo "!! $NAME: a watchdog fires in $((earliest/60)) min, under the ${EST_HOURS} h this cell needs"
        echo "!! refusing to train; the box will be deleted rather than billed for a run that cannot finish"
        ~/.local/bin/brev delete "$NAME" </dev/null >/dev/null 2>&1
        return 1
      fi
    fi

    rsync -a --timeout=60 "$REPO/brev/run_ab350.sh" "$NAME:ab/" >/dev/null 2>&1
    ssh -n "$NAME" "setsid nohup bash ~/ab/run_ab350.sh $ARM $EP $SEQID $STEPS \
                      > ~/ab/cell.log 2>&1 < /dev/null & echo started" 2>/dev/null

    local CELL_DIR="${ARM}_e$(printf '%02d' "$EP")"
    # Move any previous run's results aside before pulling this run's, and stamp the directory with
    # this run's id. A relaunch of the same cell writes to the same local path, so without this the
    # old metrics.json sits there looking exactly like a fresh result -- which on 2026-10-25 made a
    # verifier read the broken run's 50M AUC and delete both new boxes three minutes after they
    # were created. A result file must never be older than the run it is attributed to.
    if [ -d "$OUT/$CELL_DIR" ] && [ -n "$(ls -A "$OUT/$CELL_DIR" 2>/dev/null)" ]; then
      local stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
      mv "$OUT/$CELL_DIR" "$OUT/${CELL_DIR}.superseded.$stamp"
      echo "moved a previous $CELL_DIR aside as ${CELL_DIR}.superseded.$stamp"
    fi
    mkdir -p "$OUT/$CELL_DIR"
    printf '%s %s %s\n' "$RUN_ID" "$NAME" "$(date -u +%FT%TZ)" > "$OUT/$CELL_DIR/RUN_ID"
    # Pull every ~10 min rather than only at DONE. Each checkpoint's trajectory exists from the
    # moment it is evaluated, and the sweep box proved that a run whose results are only collected
    # at the end loses everything if the box goes away first.
    local limit=$(( MAX_HOURS * 60 * 2 ))
    local tick=0
    for _ in $(seq 1 $limit); do
      if ssh -n -o ConnectTimeout=20 "$NAME" "test -f ~/ab350_results/$CELL_DIR/DONE" 2>/dev/null; then break; fi
      tick=$((tick + 1))
      if [ $((tick % 20)) = 0 ]; then
        rsync -a --partial --timeout=120 "$NAME:ab350_results/$CELL_DIR/" "$OUT/$CELL_DIR/" >/dev/null 2>&1 \
          && echo "interim pull at $((tick / 2)) min: $(find "$OUT/$CELL_DIR" -name objects.parquet | wc -l) trajectories"
      fi
      sleep 30
    done
    for attempt in 1 2 3; do
      rsync -a -c --partial --timeout=180 "$NAME:ab350_results/$CELL_DIR/" \
        "$OUT/$CELL_DIR/" >/dev/null 2>&1 && break
      sleep 10
    done
    echo "pulled: $(find "$OUT/$CELL_DIR" -name objects.parquet | wc -l) trajectories"
    ~/.local/bin/brev delete "$NAME" </dev/null >/dev/null 2>&1 && echo "deleted $NAME" || echo "!! delete $NAME BY HAND"
    pkill -f "backstop_watchdog[.]sh $NAME " 2>/dev/null
    echo "=== $NAME finished ($(date -u +%H:%M:%S)) ==="
  } >> "$LOG" 2>&1
}

running=0
for CELL in "${CELLS[@]}"; do
  while [ "$running" -ge "$MAX_CONCURRENT" ]; do wait -n; running=$((running - 1)); done
  run_cell "$CELL" & running=$((running + 1))
  sleep 20          # stagger: two launchers polling brev ls at once hit the rate limit
done
wait
echo "all cells finished; brev ls:"; ~/.local/bin/brev ls 2>&1 | tail -8
