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
ARM_FLAGS=""          # recipe changes, comma-separated; see brev/arm_flags.sh. Empty = stock.
while [ $# -gt 0 ]; do
  case "$1" in
    --go) GO=1; shift ;;
    --arm-flags) ARM_FLAGS="$2"; shift 2 ;;
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
# The same helper the runner uses, so the launcher and the box agree on what a flag set is called
# and which files it needs. An unknown flag fails here, before a single box is created.
# shellcheck source=brev/arm_flags.sh
. "$REPO/brev/arm_flags.sh"
SLUG="$(arm_flags_slug "$ARM_FLAGS")" || exit 2
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
   arm flags    : ${ARM_FLAGS:-(stock recipe)}$([ -n "$SLUG" ] && echo "  -> cell suffix $SLUG")
   references   : $(for a in $ARMS; do printf "%s=%s " "$a" "$(wc -l < "$S/ab_stage_${a}.md5")"; done)files, md5-verified after transfer
   decision rule: docs/AB350_BAR.md  (not editable after this runs)
PLAN
[ "$GO" = 1 ] || { echo "
Read-only. Re-run with --go to create instances."; exit 0; }

run_cell() {
  local ARM="${1%%:*}" EP="${1##*:}"
  local NAME="v2d-ab-${ARM}-e$(printf '%02d' "$EP")${SLUG}"
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

    # Ship the runner AND every file it sources. Shipping run_ab350.sh alone is what killed the
    # 2026-10-25 relaunch: the runner sources support_flag.sh, the box did not have it, both cells
    # died on line 50 inside a second, and the boxes then idled for 80 min. The list is derived
    # from the runner itself so adding a new `source` line cannot silently repeat this.
    local -a HELPERS=()
    while read -r helper; do
      [ -n "$helper" ] && HELPERS+=("$REPO/brev/$helper")
    done < <(grep -oE '\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/[A-Za-z0-9_.-]+' "$REPO/brev/run_ab350.sh" \
             | sed 's#.*/##' | sort -u)
    # Plus the patch script behind each arm flag. arm_flags.sh is sourced by name above, but the
    # patches it runs are resolved at run time, so they have to be asked for rather than grepped.
    while read -r patch; do
      [ -n "$patch" ] && HELPERS+=("$REPO/brev/$patch")
    done < <(arm_flags_files "$ARM_FLAGS")
    for helper in "${HELPERS[@]}"; do
      [ -f "$helper" ] || { echo "!! $NAME: run_ab350.sh sources $(basename "$helper"), which does not exist"; return 1; }
    done
    rsync -a --timeout=60 "$REPO/brev/run_ab350.sh" "${HELPERS[@]}" "$NAME:ab/" >/dev/null 2>&1 \
      || { echo "!! $NAME: could not ship the runner and its helpers"; return 1; }
    # Prove they landed, by name, before starting anything.
    local shipped want
    want=$(( 1 + ${#HELPERS[@]} ))
    shipped=$(ssh -n -o ConnectTimeout=20 "$NAME" "cd ~/ab 2>/dev/null && ls -1 \
      $(printf '%s ' run_ab350.sh $(for h in "${HELPERS[@]}"; do basename "$h"; done)) 2>/dev/null | wc -l" 2>/dev/null)
    if [ "${shipped:-0}" != "$want" ]; then
      echo "!! $NAME: only ${shipped:-0} of $want runner files are on the box; NOT training"
      ~/.local/bin/brev delete "$NAME" </dev/null >/dev/null 2>&1
      return 1
    fi
    echo "shipped $want runner files: run_ab350.sh $(for h in "${HELPERS[@]}"; do basename "$h"; done | tr '\n' ' ')"
    ssh -n "$NAME" "setsid nohup env ARM_FLAGS='$ARM_FLAGS' bash ~/ab/run_ab350.sh $ARM $EP $SEQID $STEPS \
                      > ~/ab/cell.log 2>&1 < /dev/null & echo started" 2>/dev/null

    local CELL_START; CELL_START=$(date +%s)
    local CELL_DIR="${ARM}_e$(printf '%02d' "$EP")${SLUG}"
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
    # "interim pull: 0 trajectories" is not a heartbeat. On 2026-10-25 this loop logged it every
    # 10 min for 80 min while the runner had already exited and the GPU sat at 0%. Two liveness
    # probes now run every 2 min, and either one ends the cell loudly:
    #   - the runner process is gone and there is no DONE file  -> it exited early;
    #   - the box-side train.log has not grown in 10 min while the runner is still up -> hung.
    # train.log is the right file to watch, not cell.log: the runner pipes training through
    # `tee train.log | tail -20`, so cell.log legitimately stays silent for the whole run.
    local last_size=-1 last_growth; last_growth=$(date +%s)
    local DEAD=""
    for _ in $(seq 1 $limit); do
      if ssh -n -o ConnectTimeout=20 "$NAME" "test -f ~/ab350_results/$CELL_DIR/DONE" 2>/dev/null; then break; fi
      tick=$((tick + 1))

      if [ $((tick % 4)) = 0 ]; then
        local probe size alive now
        probe="$(ssh -n -o ConnectTimeout=20 "$NAME" "printf '%s %s' \
          \"\$(stat -c %s ~/ab350_results/$CELL_DIR/train.log 2>/dev/null || echo 0)\" \
          \"\$(pgrep -cf 'ab/run_ab350[.]sh' 2>/dev/null || echo 0)\"" 2>/dev/null)"
        size="${probe%% *}"; alive="${probe##* }"
        now=$(date +%s)
        case "$size" in ''|*[!0-9]*) size=-1 ;; esac
        case "$alive" in ''|*[!0-9]*) alive=1 ;; esac
        if [ "$size" -gt "$last_size" ]; then last_size=$size; last_growth=$now; fi

        if [ "$alive" = 0 ]; then
          echo "!! $NAME: run_ab350.sh is NOT running and there is no DONE file -- it exited early"
          echo "!! last 15 lines of ~/ab/cell.log:"
          ssh -n -o ConnectTimeout=20 "$NAME" "tail -15 ~/ab/cell.log" 2>/dev/null | sed 's/^/!!   /'
          DEAD="runner exited early"; break
        fi
        if [ $(( now - last_growth )) -gt 600 ]; then
          echo "!! $NAME: train.log has not grown in $(( (now - last_growth) / 60 )) min while the runner is up -- hung"
          ssh -n -o ConnectTimeout=20 "$NAME" "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" 2>/dev/null | sed 's/^/!!   gpu: /'
          DEAD="train.log stalled"; break
        fi
      fi

      if [ $((tick % 20)) = 0 ]; then
        rsync -a --partial --timeout=120 "$NAME:ab350_results/$CELL_DIR/" "$OUT/$CELL_DIR/" >/dev/null 2>&1 \
          && echo "interim pull at $((tick / 2)) min: $(find "$OUT/$CELL_DIR" -name objects.parquet | wc -l) trajectories, train.log $last_size B"
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
    # Stop this box's clock in the ledger the moment it is deleted. brev ls cannot tell us about a
    # deleted box, so a cell that ends without this line silently drops out of the spend total.
    printf '%s %s\n' "$NAME" "$(( ( $(date +%s) - CELL_START + 59 ) / 60 ))" >> "$S/spend_closed.txt"
    if [ -n "$DEAD" ]; then
      echo "=== $NAME FAILED: $DEAD ($(date -u +%H:%M:%S)) ==="
      return 1
    fi
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
