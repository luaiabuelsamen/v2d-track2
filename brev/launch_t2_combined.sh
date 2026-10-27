#!/usr/bin/env bash
# The 30-clip Tier 2 run with the combined recipe. One seed, full budget, host-side scoring.
#
# Boxes pull clips from a shared queue rather than being handed a fixed list, because we do not
# know how many boxes the org will give us. Ask for 8; if 5 come up, those 5 work through the queue
# and the launcher reports which clips are left for a second wave. A box stops claiming when its
# 24 h deletion authority can no longer fit another clip, which is what makes "3 to 4 clips per
# box" true rather than hoped for: 4 clips is ~20.4 h of the 24.
#
# Nothing here decides anything. The recipe is fixed before launch (docs/T2_COMBINED_RUN.md), the
# checkpoint per clip is chosen by the z-drop rule on the input reference alone, and the kit score
# is computed after that choice, for the record.
#
#   brev/launch_t2_combined.sh                    # plan only, creates nothing
#   brev/launch_t2_combined.sh --go               # Luai's approval
set -uo pipefail
cd /home/jetson3/projects/v2d-challenge

GO=0
ONLY_BOX=0                    # --add-box N: one replacement box joining a run already in flight
BOXES=8
BOX_PREFIX="v2d-t2-b"      # one prefix per arm group; two groups must not share a box name
OUT_OVERRIDE=""
MAX_REPLACEMENTS=6            # extra creates a run may spend on boxes that fail during bring-up
MAX_CLIPS_PER_BOX=4
TYPE=massedcompute_A6000_plus
STEPS=349999104
SAVE_EVERY=49999872
MAX_HOURS=24                  # the box-side deletion authority gate_launch.sh arms
BACKSTOP_HOURS=24.5           # mine, which must fire LAST
EST_HOURS_PER_CLIP=5.1        # 4.4 h train + 7 evaluations, measured on full350
BRINGUP_HOURS=0.3
ARM_FLAGS="start_curriculum,orientation_reward"
STD=0.3
# Extra K=V pairs forwarded verbatim to the runner's env, for arms whose patch takes a parameter
# (V2D_OBJECT_POSITION_STD, V2D_OBJECT_POSITION_WEIGHT, V2D_VOC_ZERO_AT). Kept generic so a new
# parameterised patch needs no launcher change.
ARM_ENV=""
CREATE_STAGGER=20
FIRST="3 9 18 24"             # the four clips with a stock record, so the first read is comparable
S="${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}"
STAGE="$S/t2_combined"
MAP="$S/tier2_episode_map.txt"
# The tier the pulled rollouts are scored against. tier2 for every run so far; a Tier 1 run
# trains on the clean tier_1 references and has to be scored against those.
TIER="tier2"
NPZ_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --go) GO=1; shift ;;
    --add-box) ONLY_BOX="$2"; shift 2 ;;
    --boxes) BOXES="$2"; shift 2 ;;
    --max-replacements) MAX_REPLACEMENTS="$2"; shift 2 ;;
    --clips-per-box) MAX_CLIPS_PER_BOX="$2"; shift 2 ;;
    --type) TYPE="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --map) MAP="$2"; shift 2 ;;
    --tier) TIER="$2"; shift 2 ;;
    --out) OUT_OVERRIDE="$2"; shift 2 ;;
    --name-prefix) BOX_PREFIX="$2"; shift 2 ;;
    --arm-flags) ARM_FLAGS="$2"; shift 2 ;;
    --std) STD="$2"; shift 2 ;;
    --arm-env) ARM_ENV="$2"; shift 2 ;;
    --create-stagger) CREATE_STAGGER="$2"; shift 2 ;;
    --first) FIRST="$2"; shift 2 ;;
    --max-hours) MAX_HOURS="$2"; shift 2 ;;
    --npz) NPZ_ARGS+=(--npz "$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

export V2D_SCRATCH="$S"
REPO=/home/jetson3/projects/v2d-challenge
# One run dir and one box-name prefix per arm group: two groups running at once must not share a
# queue, a claims dir, a results tree or a box name.
OUT="${OUT_OVERRIDE:-$S/t2_combined_run}"
RESULTS="$OUT/results"
RUN_ID="t2-$(date -u +%Y%m%dT%H%M%SZ)-$$"
# shellcheck source=brev/arm_flags.sh
. "$REPO/brev/arm_flags.sh"
# shellcheck source=brev/box_names.sh
. "$REPO/brev/box_names.sh"
# shellcheck source=brev/budget_guard.sh
. "$REPO/brev/budget_guard.sh"

arm_flags_normalise "$ARM_FLAGS" > /dev/null || exit 2
[ -f "$MAP" ] || { echo "FATAL: no episode map at $MAP" >&2; exit 2; }
[ -d "$STAGE" ] || { echo "FATAL: no staged references at $STAGE -- run brev/stage_t2_combined.sh" >&2; exit 2; }
[ -f "$STAGE.md5" ] || { echo "FATAL: no md5 manifest at $STAGE.md5" >&2; exit 2; }
[ -f "$STAGE/provenance.json" ] || { echo "FATAL: $STAGE has no provenance.json" >&2; exit 2; }
MOUNT="$(basename "$STAGE")"

# The reference each source must be kit-scored against comes from the stage itself, so the launch
# command carries no path and the two cannot drift apart. An explicit --npz still wins.
if [ "${#NPZ_ARGS[@]}" = 0 ]; then
  while read -r flag item; do
    [ -n "${item:-}" ] && NPZ_ARGS+=("$flag" "$item")
  done < <(env -u PYTHONPATH ./.venv311/bin/python -I -c "
import json, sys
stage = json.load(open('$STAGE/provenance.json'))
for label, path in sorted(stage.get('npz_by_source', {}).items()):
    print(f'--npz {label}={path}')
for label, path in sorted(stage.get('ref_dataset_by_source', {}).items()):
    print(f'--ref-dataset {label}={path}')
" 2>/dev/null)
  [ "${#NPZ_ARGS[@]}" -gt 0 ] && echo "scoring references from the stage: ${NPZ_ARGS[*]}"
fi

# One tree can carry two references (interp's finger tree does), so the convert root is recorded
# per clip in the stage's provenance and score_pulled.sh reads it there. Printed in the plan
# because a run that scores against two roots should not look like a run that scores against one.
CONVERT_ROOTS="$(env -u PYTHONPATH ./.venv311/bin/python -I -c "
import collections, json
stage = json.load(open('$STAGE/provenance.json'))
per_clip = stage.get('ref_dataset_by_clip', {})
lines = [f'   convert root   : {n} clips against {root}'
         for root, n in sorted(collections.Counter(per_clip.values()).items())]
if per_clip and len(per_clip) < len(stage['clips']):
    lines.append(f'   convert root   : {len(stage[\"clips\"]) - len(per_clip)} clips against the release')
print('\n'.join(lines))
" 2>/dev/null)"


# The queue: the four clips with a stock record first, then the rest in map order.
mkdir -p "$OUT" "$RESULTS" "$OUT/claims"
QUEUE="$OUT/queue.txt"
if [ "$ONLY_BOX" != 0 ]; then
  # A replacement box joins the queue and the claims of the run already in flight. The queue is
  # NOT rewritten: claim_next reads it on every claim, so truncating it under the running boxes
  # could show one of them a short file and retire it early with clips left.
  [ -s "$QUEUE" ] || { echo "FATAL: --add-box needs the queue of a run in flight: $QUEUE" >&2; exit 2; }
  CLIPS=$(grep -c '[^[:space:]]' "$QUEUE")
else
: > "$QUEUE"
for EP in $FIRST; do
  # A --first episode the map does not have would be dropped silently, and the ordering that is
  # there to give an early comparable read would quietly not happen.
  awk -v e="$EP" '$1==e' "$MAP" | grep -q . \
    || { echo "FATAL: --first names e$EP, which is not in $MAP" >&2; exit 2; }
  awk -v e="$EP" '$1==e' "$MAP" >> "$QUEUE"
done
awk -v first="$(echo "$FIRST" | tr ' ' ',')" '
  BEGIN { n = split(first, f, ","); for (i = 1; i <= n; i++) skip[f[i]] = 1 }
  $1 != "" && !($1 in skip)' "$MAP" >> "$QUEUE"
CLIPS=$(grep -c '[^[:space:]]' "$QUEUE")
[ "$CLIPS" = "$(grep -c '[^[:space:]]' "$MAP")" ] \
  || { echo "FATAL: queue has $CLIPS clips, the map has $(grep -c '[^[:space:]]' "$MAP")" >&2; exit 2; }
fi

CAPACITY=$(( BOXES * MAX_CLIPS_PER_BOX ))
HOURS_PER_BOX="$(awk -v c="$MAX_CLIPS_PER_BOX" -v h="$EST_HOURS_PER_CLIP" -v b="$BRINGUP_HOURS" \
                   'BEGIN{printf "%.1f", c*h+b}')"
TOTAL_HOURS="$(awk -v n="$CLIPS" -v h="$EST_HOURS_PER_CLIP" -v b="$BRINGUP_HOURS" -v x="$BOXES" \
                 'BEGIN{printf "%.1f", n*h + x*b}')"
TOTAL_USD="$(awk -v h="$TOTAL_HOURS" -v r="$RATE_USD_PER_HOUR" 'BEGIN{printf "%.2f", h*r}')"
WORST_USD="$(awk -v x="$BOXES" -v h="$MAX_HOURS" -v r="$RATE_USD_PER_HOUR" 'BEGIN{printf "%.2f", x*h*r}')"

cat <<PLAN
== The 30-clip Tier 2 run, combined recipe ($RUN_ID)
   clips          : $CLIPS, queue order $(head -4 "$QUEUE" | awk '{printf "e%02d ", $1}')... ($STAGE)
   map            : $MAP
   tier           : $TIER  (what the pulled checkpoints are scored against)
   box names      : $(for i in $(seq 1 "$BOXES"); do printf '%s%02d ' "$BOX_PREFIX" "$i"; done)
                    replacements up to ${BOX_PREFIX}$(printf '%02d' $(( BOXES + MAX_REPLACEMENTS )))
   references     : $(awk '{print $3}' "$STAGE/sources.txt" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
$CONVERT_ROOTS
   recipe         : seed 42, $STEPS steps, checkpoint every $SAVE_EVERY, support per clip
   arm flags      : $ARM_FLAGS  (overrides: $(arm_flags_overrides "$ARM_FLAGS"), object_orientation_std=$STD)
   arm env        : ${ARM_ENV:-(none)}
   boxes          : up to $BOXES x $TYPE, <= $MAX_CLIPS_PER_BOX clips each, sequential
   capacity       : $CAPACITY clip-slots for $CLIPS clips$([ "$CAPACITY" -lt "$CLIPS" ] && echo "  -- NOT ENOUGH, a second wave will be needed")
   per box        : ~$HOURS_PER_BOX h of work against a ${MAX_HOURS} h authority
   expected spend : \$$TOTAL_USD   (worst case if every box runs its full authority: \$$WORST_USD)
   scoring        : kit-score every checkpoint on pull, then z-drop selection per clip
   bar            : kit AUC > 0.36635 on e03 e09 e18 e24, no extra clip without a drop-free
                    checkpoint (docs/T2_COMBINED_RUN.md). A shipping run, not an attribution run.
PLAN

if [ "$GO" != 1 ]; then
  echo "   plan only. Nothing created. Add --go to launch."
  exit 0
fi

if [ "$ONLY_BOX" != 0 ]; then
  UNCLAIMED=$(comm -23 <(awk '{printf "e%02d\n", $1}' "$QUEUE" | sort) \
                       <(ls "$OUT/claims" 2>/dev/null | sort) | tr '\n' ' ')
  echo "== adding $BOX_PREFIX$(printf '%02d' "$ONLY_BOX") to the run in $OUT"
  echo "   unclaimed now  : ${UNCLAIMED:-none}"
  echo "   it will take   : up to $MAX_CLIPS_PER_BOX clips, ~$HOURS_PER_BOX h, \$$(awk -v h="$HOURS_PER_BOX" -v r="$RATE_USD_PER_HOUR" 'BEGIN{printf "%.2f", h*r}')"
  budget_ok_for_hours "$HOURS_PER_BOX" "one replacement box" || exit 3
else
  budget_ok_for_hours "$TOTAL_HOURS" "the whole $CLIPS-clip run" || exit 3
fi

# claim_next -- the next unclaimed clip, atomically. mkdir is the lock; all boxes claim locally.
claim_next() {
  local EP SEQ
  while read -r EP SEQ; do
    [ -n "${EP:-}" ] || continue
    if mkdir "$OUT/claims/e$(printf '%02d' "$EP")" 2>/dev/null; then printf '%s %s\n' "$EP" "$SEQ"; return 0; fi
  done < "$QUEUE"
  return 1
}

# run_clip <name> <episode> <sequence> -- one clip on one box, watched, pulled and scored.
# Returns 0 if a trajectory came back, 1 otherwise; the caller decides whether to keep the box.
run_clip() {
  local NAME="$1" EP="$2" SEQ="$3"
  local TAG="e$(printf '%02d' "$EP")"
  echo "-- $NAME: $TAG $SEQ starting ($(date -u +%FT%TZ))"
  ssh -n "$NAME" "mkdir -p ~/t2 && printf '%s %s\n' $EP $SEQ > ~/t2/map_$TAG.txt" 2>/dev/null \
    || { echo "!! $NAME: could not write the clip map"; return 1; }
  ssh -n "$NAME" "setsid nohup env ARM_FLAGS='$ARM_FLAGS' V2D_OBJECT_ORIENTATION_STD='$STD' \
      $ARM_ENV \
      ASSET_ROOT='/v2d/assets/$MOUNT' RESULTS_DIR=\$HOME/t2_results \
      bash ~/t2/run_tier2_full.sh ~/t2/map_$TAG.txt $STEPS $SAVE_EVERY \
      > ~/t2/clip_$TAG.log 2>&1 < /dev/null & echo started" 2>/dev/null \
    || { echo "!! $NAME: could not start the runner"; return 1; }

  local LOCAL="$RESULTS/$TAG"
  mkdir -p "$LOCAL"
  printf '%s %s %s\n' "$RUN_ID" "$NAME" "$(date -u +%FT%TZ)" > "$LOCAL/RUN_ID"
  local started; started=$(date +%s)
  local limit=$(( (${MAX_HOURS%.*} + 1) * 60 * 2 ))
  local tick=0 last_size=-1 last_growth dead=""
  last_growth=$(date +%s)
  for _ in $(seq 1 $limit); do
    if ssh -n -o ConnectTimeout=20 "$NAME" "test -f ~/t2_results/$TAG/DONE" 2>/dev/null; then break; fi
    tick=$((tick + 1))

    # Liveness every 2 min. "interim pull: 0 trajectories" is not a heartbeat: on 2026-10-25 that
    # line was logged for 80 min while the runner had already exited and the GPU sat at 0%.
    if [ $((tick % 4)) = 0 ]; then
      local probe size alive now
      probe="$(ssh -n -o ConnectTimeout=20 "$NAME" "printf '%s %s' \
        \"\$(stat -c %s ~/t2_results/$TAG/train.log 2>/dev/null || echo 0)\" \
        \"\$(pgrep -cf 'run_tier2_full[.]sh' 2>/dev/null || echo 0)\"" 2>/dev/null)"
      size="${probe%% *}"; alive="${probe##* }"; now=$(date +%s)
      case "$size" in ''|*[!0-9]*) size=-1 ;; esac
      case "$alive" in ''|*[!0-9]*) alive=1 ;; esac
      [ "$size" -gt "$last_size" ] && { last_size=$size; last_growth=$now; }
      if [ "$alive" = 0 ]; then
        echo "!! $NAME $TAG: the runner is NOT running and there is no DONE -- it exited early"
        ssh -n -o ConnectTimeout=20 "$NAME" "tail -15 ~/t2/clip_$TAG.log" 2>/dev/null | sed 's/^/!!   /'
        dead="runner exited early"; break
      fi
      if [ $(( now - last_growth )) -gt 900 ]; then
        echo "!! $NAME $TAG: train.log has not grown in $(( (now - last_growth) / 60 )) min -- hung"
        ssh -n -o ConnectTimeout=20 "$NAME" \
          "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" 2>/dev/null \
          | sed 's/^/!!   gpu: /'
        dead="train.log stalled"; break
      fi
    fi

    # A pull every 10 min. Every evaluated checkpoint exists from the moment it is written, and the
    # sweep box proved that results collected only at the end are lost with the box.
    if [ $((tick % 20)) = 0 ]; then
      rsync -a --partial --timeout=120 "$NAME:t2_results/$TAG/" "$LOCAL/" >/dev/null 2>&1 \
        && echo "   $NAME $TAG: pull at $((tick / 2)) min, $(find "$LOCAL" -name objects.parquet | wc -l) trajectories, train.log $last_size B"
    fi
    sleep 30
  done

  for attempt in 1 2 3; do
    rsync -a -c --partial --timeout=180 "$NAME:t2_results/$TAG/" "$LOCAL/" >/dev/null 2>&1 && break
    sleep 10
  done
  local got; got=$(find "$LOCAL" -name objects.parquet | wc -l)
  echo "-- $NAME: $TAG finished after $(( ($(date +%s) - started) / 60 )) min, $got trajectories${dead:+ ($dead)}"
  bash "$REPO/brev/score_pulled.sh" --results "$RESULTS" --tier "$TIER" --provenance "$STAGE/provenance.json" \
    "${NPZ_ARGS[@]+${NPZ_ARGS[@]}}" 2>&1 | sed 's/^/   /'
  [ -n "$dead" ] && return 1
  [ "$got" -gt 0 ] || return 1
  return 0
}

run_box() {
  local INDEX="$1"
  local NAME="$BOX_PREFIX$(printf '%02d' "$INDEX")"
  local LOG="$OUT/${NAME}.log"
  {
    echo "=== $NAME ($(date -u +%FT%TZ)) ==="
    setsid nohup bash "$REPO/brev/backstop_watchdog.sh" "$NAME" \
      "$(awk -v h="$BACKSTOP_HOURS" 'BEGIN{printf "%d", h*3600}')" "$OUT/backstop.log" >/dev/null 2>&1 &

    local box_start; box_start=$(date +%s)
    bash "$REPO/brev/gate_launch.sh" --phase build --dev-box --name "$NAME" \
         --type "$TYPE" --max-hours "$MAX_HOURS" --go \
      || { echo "!! $NAME: bring-up failed (quota, availability, or image build)"
           # Brev bills from creation, so a box that failed during bring-up still cost something
           # unless it never existed. Recording it is the difference between a $230 cap that
           # means something and one that quietly drifts.
           ab4_ledger_add "$NAME" "$(( ($(date +%s) - box_start) / 60 ))"
           return 1; }

    ssh -n "$NAME" "mkdir -p ~/v2d_assets/$MOUNT ~/t2" 2>/dev/null
    local ok=0
    for attempt in 1 2 3 4 5 6; do
      rsync -a -c --partial --timeout=180 \
        -e 'ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=4' \
        "$STAGE/" "$NAME:v2d_assets/$MOUNT/" >/dev/null 2>&1
      # By md5, not by file count: a truncated parquet passes a count and then fails with
      # "Parquet magic bytes not found in footer" four hours into a run.
      if ssh -n "$NAME" "cd ~/v2d_assets/$MOUNT && find . -type f | sort | xargs md5sum" 2>/dev/null \
           | grep -E '^[0-9a-f]{32} ' | sort -k2 > "$OUT/${NAME}.remote.md5" \
         && diff -q <(sort -k2 "$STAGE.md5") "$OUT/${NAME}.remote.md5" >/dev/null; then
        ok=1; echo "references verified by md5 on attempt $attempt"; break
      fi
      echo "   md5 attempt $attempt: $(wc -l < "$OUT/${NAME}.remote.md5") files on the box, $(wc -l < "$STAGE.md5") in the manifest"
      sleep 5
    done
    if [ "$ok" != 1 ]; then
      echo "!! $NAME: references never matched by md5; NOT training"
      # Say WHY, here, now. The first 30-clip launch retried blind six times on each of five boxes
      # and the reason -- one file in the tree that the manifest did not list -- was only found
      # afterwards, by diffing the two listings by hand, after every box had been deleted.
      diff <(sort -k2 "$STAGE.md5") "$OUT/${NAME}.remote.md5" | head -6 | sed 's/^/!!   /'
      ~/.local/bin/brev delete "$NAME" </dev/null >/dev/null 2>&1
      ab4_ledger_add "$NAME" "$(( ($(date +%s) - box_start) / 60 ))"
      return 1
    fi

    # Ship the runner, every file it sources, and the patch behind every arm flag. Shipping the
    # runner alone is what killed both cells of the 2026-10-25 relaunch inside a second.
    local -a FILES=("$REPO/brev/run_tier2_full.sh")
    while read -r helper; do
      [ -n "$helper" ] && FILES+=("$REPO/brev/$helper")
    done < <(grep -oE '\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/[A-Za-z0-9_.-]+' \
               "$REPO/brev/run_tier2_full.sh" | sed 's#.*/##' | sort -u)
    while read -r patch; do
      [ -n "$patch" ] && FILES+=("$REPO/brev/$patch")
    done < <(arm_flags_files "$ARM_FLAGS")
    for f in "${FILES[@]}"; do
      [ -f "$f" ] || { echo "!! $NAME: the runner needs $(basename "$f"), which does not exist"; \
                       ~/.local/bin/brev delete "$NAME" </dev/null >/dev/null 2>&1; return 1; }
    done
    rsync -a --timeout=60 "${FILES[@]}" "$NAME:t2/" >/dev/null 2>&1 \
      || { echo "!! $NAME: could not ship the runner"; return 1; }
    local want shipped
    want=${#FILES[@]}
    shipped=$(ssh -n -o ConnectTimeout=20 "$NAME" "cd ~/t2 2>/dev/null && ls -1 \
      $(for f in "${FILES[@]}"; do basename "$f"; done | tr '\n' ' ') 2>/dev/null | wc -l" 2>/dev/null)
    if [ "${shipped:-0}" != "$want" ]; then
      echo "!! $NAME: only ${shipped:-0} of $want runner files landed; NOT training"
      ~/.local/bin/brev delete "$NAME" </dev/null >/dev/null 2>&1
      ab4_ledger_add "$NAME" "$(( ($(date +%s) - box_start) / 60 ))"
      return 1
    fi
    echo "shipped $want files: $(for f in "${FILES[@]}"; do basename "$f"; done | tr '\n' ' ')"

    local done_clips=0 claim
    while [ "$done_clips" -lt "$MAX_CLIPS_PER_BOX" ]; do
      # Never start a clip that the deletion authority cannot let finish. A box deleted mid-clip
      # bills for the whole clip and returns nothing -- which is how the sweep box was lost.
      local spent_h left_h
      spent_h="$(awk -v s="$(( $(date +%s) - box_start ))" 'BEGIN{printf "%.2f", s/3600}')"
      left_h="$(awk -v m="$MAX_HOURS" -v s="$spent_h" 'BEGIN{printf "%.2f", m-s-0.4}')"
      if awk -v l="$left_h" -v n="$EST_HOURS_PER_CLIP" 'BEGIN{exit !(l < n)}'; then
        echo "$NAME: ${left_h} h left of the ${MAX_HOURS} h authority, a clip needs $EST_HOURS_PER_CLIP -- stopping"
        break
      fi
      claim="$(claim_next)" || { echo "$NAME: queue empty"; break; }
      run_clip "$NAME" ${claim} || echo "!! $NAME: clip ${claim%% *} did not complete cleanly"
      done_clips=$((done_clips + 1))
    done

    local mins; mins=$(( ($(date +%s) - box_start) / 60 ))
    ~/.local/bin/brev delete "$NAME" </dev/null >/dev/null 2>&1 \
      && echo "deleted $NAME after $mins min" || echo "!! delete $NAME BY HAND"
    ab4_ledger_add "$NAME" "$mins"
    pkill -f "backstop_watchdog[.]sh $NAME " 2>/dev/null
    echo "=== $NAME done: $done_clips clips, $mins min, \$$(awk -v m="$mins" -v r="$RATE_USD_PER_HOUR" 'BEGIN{printf "%.2f", m/60*r}')"
  } >> "$LOG" 2>&1
}

if [ "$ONLY_BOX" != 0 ]; then
  box_name_reserve "$OUT/boxnames" "$ONLY_BOX" \
    || echo "   (name $ONLY_BOX was already handed out in this run; continuing, it is your call)"
  echo "adding one box; log in $OUT/$BOX_PREFIX$(printf '%02d' "$ONLY_BOX").log"
  run_box "$ONLY_BOX" &
  wait
  echo "== replacement box done"
  exit 0
fi

# One worker slot: take a free name, run a box on it, and if that box dies during bring-up, take
# the next free name and try again. Three of eight creates failed provider-side on the first
# attempt and nothing replaced them, which is how 30 clips ended up with 28 slots.
#
# It stops replacing for any of four reasons, and says which: no name left under the cap, no clip
# left unclaimed, the dollar cap, or a box that did work (whose own exit is not a failure to
# replace). A short pause first, so a provider outage is not hammered.
NAMES="$OUT/boxnames"
MAX_BOX_NAMES=$(( BOXES + MAX_REPLACEMENTS ))

unclaimed_clips() {
  comm -23 <(awk '{printf "e%02d\n", $1}' "$QUEUE" | sort) \
           <(ls "$OUT/claims" 2>/dev/null | sort) | grep -c '[^[:space:]]'
}

box_slot() {
  local slot="$1" index left replacements=0
  while :; do
    if ! index="$(box_name_claim "$NAMES" "$MAX_BOX_NAMES")"; then
      echo "slot $slot: no box name left under the cap of $MAX_BOX_NAMES; not replacing"
      return 1
    fi
    if run_box "$index"; then return 0; fi

    # It failed. Everything below decides whether another box would have anything to do.
    left="$(unclaimed_clips)"
    if [ "$left" = 0 ]; then
      echo "slot $slot: $BOX_PREFIX$(printf '%02d' "$index") failed, but every clip is claimed; not replacing"
      return 1
    fi
    if ! budget_ok_for_hours "$HOURS_PER_BOX" "a replacement for the failed $BOX_PREFIX$(printf '%02d' "$index")"; then
      echo "slot $slot: $BOX_PREFIX$(printf '%02d' "$index") failed and the budget will not fund a replacement"
      return 1
    fi
    replacements=$((replacements + 1))
    echo "slot $slot: $BOX_PREFIX$(printf '%02d' "$index") failed with $left clips unclaimed; replacing it (attempt $((replacements + 1)))"
    sleep 30
  done
}

echo "launching up to $BOXES boxes; per-box logs in $OUT/$BOX_PREFIX*.log"
echo "failed boxes are replaced automatically, up to $MAX_BOX_NAMES boxes in total"
for i in $(seq 1 "$BOXES"); do
  box_slot "$i" &
  sleep "$CREATE_STAGGER"   # stagger the creates: a quota refusal stays attributable to one
                            # box, and `brev create` is rate limited at roughly three a minute --
                            # three in one minute returned "rate limit exceeded" on 2026-10-26.
                            # Raise it when several launchers are creating at the same time.
done
wait
echo "boxes created in total: $(box_names_taken "$NAMES") of a cap of $MAX_BOX_NAMES"

LEFT=$(comm -23 <(awk '{printf "e%02d\n", $1}' "$QUEUE" | sort) \
                <(ls "$OUT/claims" 2>/dev/null | sort) | tr '\n' ' ')
echo "== run finished. clips never claimed: ${LEFT:-none}"
bash "$REPO/brev/score_pulled.sh" --results "$RESULTS" --tier "$TIER" --provenance "$STAGE/provenance.json" \
  "${NPZ_ARGS[@]+${NPZ_ARGS[@]}}"
echo "spend committed so far: \$$(ab4_spent_usd) of \$$NEW_BUDGET_USD"
~/.local/bin/brev ls 2>&1 | tail -3
