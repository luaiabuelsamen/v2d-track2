#!/usr/bin/env bash
# Stage (and, only with --go, launch) the second wave: six arm groups, one variable each.
#
# The first wave ran one recipe over 30 clips. This one runs SIX recipes over 28, each aimed at a
# failure mode the metrics deep dive identified, and each a single change against the shipping
# recipe (START + ORI 0.3 + FING + CLEAN) that produced the clip's record. Groups are kept apart
# end to end -- own stage dir, own map, own run dir, own box-name prefix -- because a box applies
# one patch set to every clip it takes, so two arms on one box would be two variables on one cell.
#
#   A   object_position std 0.2 -> 0.08 m          18 clips   MP-SR, then AUC
#   B1  object_position weight 1 -> 5              e04 e25    keep the object at all
#   B2  VOC spring zero point 60% -> 85%           e14 e19 e28  survive first contact
#   B3  revert the START patch                     e05 e07 e09  recover late in the clip
#   B4  orientation std 0.3 -> 0.8                 e17        angle, where 0.3 is inert
#   B5  bridge-smoothed reference instead of raw   e01        a reference it can track
#
# And one follow-up, staged after B4 won e17 by +0.2004 and asked for by name, never by default:
#
#   B4x orientation std 0.3 -> 0.8                 the 9 orientation-limited clips
#       e00 e06 e08 e09 e10 e13 e20 e21 e26        9 boxes, 1 clip each, ~$30.60
#
# e13 and e21 are deliberately not run: both sit within 0.03 of their own reference ceiling, so
# there is nothing a policy change can win there (docs/METRICS_DEEP_DIVE.md).
#
# Staging is free and idempotent. Launching is not, so it needs --go AND an explicit group list.
set -uo pipefail

S="${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}"
REPO=/home/jetson3/projects/v2d-challenge
ROOT="$S/second_wave"
# The shipping recipe's PATCH flags. FING (interp's finger parquets) and CLEAN (tetool's
# bridge-cleaned poses) are properties of the staged references, not patches, which is why they do
# not appear here -- the first wave shipped exactly these two patch scripts.
SHIPPING="start_curriculum,orientation_reward"
GO=0
ONLY=""
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --go) GO=1; shift ;;
    --groups) ONLY="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# group | clips | arm flags | env assignments | boxes | why
# NOT `GROUPS`: bash reserves that for the caller's group ids, so assigning to it silently
# does nothing and every field below comes back empty.
ARM_GROUPS=$(cat <<'EOF'
A|0 2 3 6 8 10 11 12 15 16 18 20 22 23 24 26 27 29|start_curriculum,orientation_reward,object_position|V2D_OBJECT_POSITION_STD=0.08|6|object_position std 0.08 m
B1|4 25|start_curriculum,orientation_reward,object_position|V2D_OBJECT_POSITION_STD=0.2 V2D_OBJECT_POSITION_WEIGHT=5.0|1|object_position weight 5
B2|14 19 28|start_curriculum,orientation_reward,voc_curriculum|V2D_VOC_ZERO_AT=0.85|1|VOC zero point 85%
B3|5 7 9|orientation_reward||1|START reverted
B4|17|start_curriculum,orientation_reward|V2D_OBJECT_ORIENTATION_STD=0.8|1|orientation std 0.8
B5|1|start_curriculum,orientation_reward||1|bridge-smoothed reference
B4x|0 6 8 9 10 13 20 21 26|start_curriculum,orientation_reward|V2D_OBJECT_ORIENTATION_STD=0.8|9|orientation std 0.8 on the 9 orientation-limited clips
EOF
)

# B4x is the follow-up to B4's +0.2004 on e17, not part of the registered second wave, so a bare
# run never includes it: it has to be asked for by name. See docs/SECOND_WAVE.md.
REGISTERED="A B1 B2 B3 B4 B5"
[ -n "$ONLY" ] || ONLY="$REGISTERED"

# Substring matching would make `--groups B4` select B4x as well -- nine boxes nobody asked for.
selected() {
  case " $ONLY " in *" $1 "*) return 0 ;; esac
  return 1
}

# shellcheck source=brev/arm_flags.sh
. "$REPO/brev/arm_flags.sh"

CAP=90
BOX_CAP=11
RATE=0.68
EST_HOURS_PER_CLIP=4.4
BRINGUP_HOURS=0.6
total_cost=0
total_boxes=0
total_clips=0
plan=""

mkdir -p "$ROOT"
while IFS='|' read -r group clips flags envs boxes why; do
  [ -n "$group" ] || continue
  selected "$group" || continue

  map="$ROOT/map_$group.txt"
  : > "$map"
  for ep in $clips; do
    awk -v e="$ep" '$1==e' "$S/tier2_episode_map.txt" >> "$map"
  done
  want=$(echo "$clips" | wc -w)
  got=$(grep -c '[^[:space:]]' "$map")
  [ "$want" = "$got" ] || { echo "FATAL: group $group: asked for $want clips, map has $got" >&2; exit 2; }

  arm_flags_normalise "$flags" >/dev/null || exit 2
  slug="$(arm_flags_slug "$flags")"
  per_box=$(( (got + boxes - 1) / boxes ))
  hours="$(awk -v c="$per_box" -v h="$EST_HOURS_PER_CLIP" -v b="$BRINGUP_HOURS" 'BEGIN{printf "%.1f", b + c*h}')"
  cost="$(awk -v n="$boxes" -v h="$hours" -v r="$RATE" 'BEGIN{printf "%.2f", n*h*r}')"
  total_cost="$(awk -v a="$total_cost" -v b="$cost" 'BEGIN{printf "%.2f", a+b}')"
  total_boxes=$(( total_boxes + boxes ))
  total_clips=$(( total_clips + got ))

  plan="$plan$(printf '%-3s %2d clips  %d box(es) x %s clips  %5s h/box  $%6s  %s\n' \
    "$group" "$got" "$boxes" "$per_box" "$hours" "$cost" "$why")"$'\n'

  printf '%s\n' "$group $flags${slug:+ (slug $slug)} ${envs:-no env} -> $map" >&2
done <<< "$ARM_GROUPS"

echo
echo "== second wave plan"
printf '%s' "$plan"
printf '%-3s %2d clips  %d boxes%*s$%6s\n' "all" "$total_clips" "$total_boxes" 26 "" "$total_cost"
echo "   budget: \$$total_cost of the \$$CAP cap; boxes: $total_boxes of $BOX_CAP"
if awk -v c="$total_cost" -v cap="$CAP" 'BEGIN{exit !(c > cap)}'; then
  echo "REFUSED: \$$total_cost is over the \$$CAP cap. Cut clips or boxes before asking for a go." >&2
  exit 3
fi
if [ "$total_boxes" -gt "$BOX_CAP" ]; then
  echo "REFUSED: $total_boxes boxes is over the cap of $BOX_CAP." >&2
  exit 3
fi
echo "   maps in $ROOT; staging and launching are separate -- this script only wrote maps."
[ "$GO" = 1 ] || { echo "   (no --go: nothing staged or launched)"; exit 0; }

# ---------------------------------------------------------------------------- stage, then launch
# Creates are rate limited at roughly three a minute, and several launchers create at once, so the
# groups start spaced: the stagger inside a launcher plus the gap between launchers has to keep
# the GLOBAL create rate under the limit. 30 s inside and 90 s between is ~2 creates a minute.
STAGGER="${V2D_CREATE_STAGGER:-30}"
GAP="${V2D_GROUP_GAP:-90}"

while IFS='|' read -r group clips flags envs boxes why; do
  [ -n "$group" ] || continue
  selected "$group" || continue
  map="$ROOT/map_$group.txt"
  mount="sw_$group"
  got=$(grep -c '[^[:space:]]' "$map")
  per_box=$(( (got + boxes - 1) / boxes ))

  echo "== staging $group ($got clips) into $S/$mount"
  if [ "$DRY" = 1 ]; then
    echo "   DRY: $REPO/brev/stage_t2_combined.sh --from-briefs --map $map --mount $mount"
  else
    "$REPO/brev/stage_t2_combined.sh" --from-briefs --map "$map" --mount "$mount" \
      > "$ROOT/stage_$group.log" 2>&1 || {
        echo "FATAL: staging $group failed; see $ROOT/stage_$group.log" >&2
        tail -5 "$ROOT/stage_$group.log" >&2
        exit 3
      }
    tail -2 "$ROOT/stage_$group.log" | sed 's/^/   /'
  fi

  echo "== launching $group: $boxes box(es) x $per_box clips, flags $flags, env ${envs:-none}"
  launch=("$REPO/brev/launch_t2_combined.sh" --go
          --map "$map" --stage "$S/$mount" --out "$ROOT/run_$group"
          --name-prefix "v2d-sw-$(echo "$group" | tr 'A-Z' 'a-z')-b"
          --arm-flags "$flags" --boxes "$boxes" --clips-per-box "$per_box"
          --first "" --create-stagger "$STAGGER")
  [ -n "$envs" ] && launch+=(--arm-env "$envs")
  if [ "$DRY" = 1 ]; then
    printf '   DRY: %s\n' "${launch[*]}"
  else
    nohup "${launch[@]}" > "$ROOT/launch_$group.log" 2>&1 &
    echo "   started, pid $!, log $ROOT/launch_$group.log"
    sleep "$GAP"
  fi
done <<< "$ARM_GROUPS"

echo
echo "== all selected groups launched; watch $ROOT/launch_*.log and $ROOT/run_*/\*.log"
