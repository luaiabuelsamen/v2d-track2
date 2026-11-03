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
# And two follow-ups, asked for by name, never by default. B4x was staged after B4 won e17 by
# +0.2004; CR is the contact-role reward, the first arm that adds a reward term of our own:
#
#   B4x orientation std 0.3 -> 0.8                 the orientation-limited clips
#       e00 e06 e08 e09 e10 e13 e20                8 boxes, 1 clip each, ~$27.20
#       e26                                        e21 dropped: already above its ceiling
#
#   A29 arm A, one clip, to replace what a reboot took   e29   1 box, ~$3.40
#       The 2026-10-28 reboot wiped the scratchpad and e29's rollout and checkpoint existed only
#       there (docs/RECOVERY_2026-10-28.md). Same recipe, same selection rule; it is a re-roll, so
#       it will not reproduce the lost rollout bit for bit.
#
#   T1P Tier 1 precision, from scratch on the clean tier_1 references   7 boxes, ~$24
#       e02 e09 e11 e13 e17 e21 e29  -- the seven Tier 1 clips sitting just outside the 3 cm
#       MP-SR gate (3.85-5.15 cm). It is the only untried lever there: the warm start was run on
#       six of these seven and every one came back WORSE on MPPE (e02 4.57 -> 35.05,
#       e29 3.85 -> 8.72), because re-fitting a converged actor to a different trajectory trades
#       position for reference tracking. Training from scratch on the clean reference instead
#       removes the ~1.94 cm of noise these policies currently chase. Three Tier 1 rungs are one
#       clip wide (mp-sr 1 clip, mppe 1.9 cm summed, auc +1.03 summed), and Tier 1 adoption reads
#       the permitted input, which IS the scoring reference, so a clip ships only if it improves.
#   CR  contact-role reward (docs/ARM_CR.md)       six of the eight clips under 0.30
#       e04 e05 e07 e09 e19 e25                    3 boxes, 2 clips each, ~$19.18
#
# e01 and e28 are deliberately not run: the roles measure 43 contact-role frames of 747 and 12 of
# 988, so the arm's two terms apply to about 1% of e28 and 6% of e01 and cannot move either clip
# (results/contact_roles.json). Two clips per box, not three: the cost model prices a group by its
# FULLEST box, so six over three boxes is 2 per box at $19.18 and finishes in 9.4 h.
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
    # The cap this invocation is allowed to spend. The registered second wave's is $90; an arm
    # launched on its own gets its own, so the refusal below is checked against what was
    # authorised for THIS launch rather than against the whole wave's envelope.
    --cap) V2D_CAP="$2"; shift 2 ;;
    # The instance type, forwarded to the launcher. It is NOT left to the launcher's default,
    # because Brev's catalogue churns: the launcher's built-in default
    # (massedcompute_A6000_plus) was listed but failing to create boxes on 2026-11-02, and a
    # pinned type that silently stops creating is a wave that launches nothing. Check with
    # `v2d brev-pick --show 6` every time and pass what it names.
    --type) V2D_TYPE="$2"; shift 2 ;;
    --box-cap) V2D_BOX_CAP="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# group | clips | arm flags | env assignments | boxes | stage | tier | steps | why
# stage: a tree under $V2D_SCRATCH to launch from, instead of staging sw_<group> from
#        the briefs. tier: what the pulled checkpoints are scored against (default
#        tier2). steps: the training budget (default 349999104), optionally STEPS:SAVE when the
#        checkpoint interval has to change with it. All three empty for every second-wave arm,
#        which is what keeps those rows reading as they did.
# A group whose clip list is not known yet reads it from $ROOT/clips_<group>.txt when that file
# exists -- T3-FULL's order comes from interp's reconstruction quality, which lands later.
# NOT `GROUPS`: bash reserves that for the caller's group ids, so assigning to it silently
# does nothing and every field below comes back empty.
ARM_GROUPS=$(cat <<'EOF'
A|0 2 3 6 8 10 11 12 15 16 18 20 22 23 24 26 27 29|start_curriculum,orientation_reward,object_position|V2D_OBJECT_POSITION_STD=0.08|6||||object_position std 0.08 m
B1|4 25|start_curriculum,orientation_reward,object_position|V2D_OBJECT_POSITION_STD=0.2 V2D_OBJECT_POSITION_WEIGHT=5.0|1||||object_position weight 5
B2|14 19 28|start_curriculum,orientation_reward,voc_curriculum|V2D_VOC_ZERO_AT=0.85|1||||VOC zero point 85%
B3|5 7 9|orientation_reward||1||||START reverted
B4|17|start_curriculum,orientation_reward|V2D_OBJECT_ORIENTATION_STD=0.8|1||||orientation std 0.8
B5|1|start_curriculum,orientation_reward||1||||bridge-smoothed reference
B4x|0 6 8 9 10 13 20 26|start_curriculum,orientation_reward|V2D_OBJECT_ORIENTATION_STD=0.8|8||||orientation std 0.8 on the orientation-limited clips
T1|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29|start_curriculum,orientation_reward,object_position|V2D_OBJECT_POSITION_STD=0.08|8|t1_precision|tier1||Tier 1: the clean tier_1 references, precision recipe, one seed
W3A|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29|start_curriculum,orientation_reward,object_position|V2D_OBJECT_POSITION_STD=0.08|8|w3_bridge_v2|||wave 3: precision recipe on bridge_v2, seed 42
W3B|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29|start_curriculum,orientation_reward,object_position|V2D_OBJECT_POSITION_STD=0.08 V2D_SEED=43|8|w3_bridge_v2|||wave 3: the same recipe and tree, seed 43
W3R|1 4 5 7 13 14 19 25 28|start_curriculum,orientation_reward,object_position,body_joint_position|V2D_OBJECT_POSITION_STD=0.08 V2D_OBJECT_POSITION_WEIGHT=5.0 V2D_BODY_JOINT_POSITION_WEIGHT=1.0|5|w3_regrasp||699998208|wave 3: the nine clips under 0.30, object over body, 700M steps
T3-VALID|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29|start_curriculum,orientation_reward,object_position|V2D_OBJECT_POSITION_STD=0.08|2|t3_recon|tier3|409600:409600|Tier 3: a valid entry on every clip at the insurance budget
T3-FULL|3 24 18 12 27 8 22 15 20|start_curriculum,orientation_reward,object_position|V2D_OBJECT_POSITION_STD=0.08|3|t3_recon|tier3||Tier 3: full-length precision on as many clips as the credit buys
A29|29|start_curriculum,orientation_reward,object_position|V2D_OBJECT_POSITION_STD=0.08|1|a29_refs|||re-roll e29 on arm A after the 2026-10-28 scratchpad loss
T1P|2 9 11 13 17 21 29|start_curriculum,orientation_reward,object_position|V2D_OBJECT_POSITION_STD=0.08|7|t1_precision|tier1||Tier 1 precision, from scratch on the clean tier_1 references, on the seven clips at the MP-SR gate
CR|4 5 7 9 19 25|start_curriculum,orientation_reward,contact_role|V2D_CONTACT_ROLES=/v2d/assets/cr_refs/contact_roles|3|cr_refs|||contact-role reward on the six clips the roles can reach
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

CAP="${V2D_CAP:-90}"
BOX_CAP="${V2D_BOX_CAP:-11}"
#: `v2d brev-pick` named this on 2026-11-03 at $0.60/h, the cheapest of 86 types clearing
#: the 46 GB floor. Re-check before every launch; the catalogue churns.
TYPE="${V2D_TYPE:-hyperstack_A6000}"
RATE=0.68
EST_HOURS_PER_CLIP=4.4
#: The budget every arm so far has trained to; a group's `steps` field is priced against it.
STOCK_STEPS=349999104
BRINGUP_HOURS=0.6
total_cost=0
total_boxes=0
total_clips=0
plan=""

mkdir -p "$ROOT"
while IFS='|' read -r group clips flags envs boxes stage tier steps why; do
  [ -n "$group" ] || continue
  selected "$group" || continue

  # A clip list that arrives later: one line or one space-separated list of episode numbers.
  if [ -f "$ROOT/clips_$group.txt" ]; then
    clips="$(tr '\n' ' ' < "$ROOT/clips_$group.txt")"
    printf '%s\n' "   $group takes its clips from $ROOT/clips_$group.txt: $clips" >&2
  fi
  # STEPS:SAVE, where the two can be equal -- a 409,600-step run checkpoints once, at the end.
  case "$steps" in *:*) save="${steps#*:}" ;; *) save="" ;; esac
  steps="${steps%%:*}"

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
  # A longer budget costs proportionally longer: W3R's 700M steps is two clips' worth of GPU in
  # one cell, and a plan that priced it as one would understate the arm by half.
  hours="$(awk -v c="$per_box" -v h="$EST_HOURS_PER_CLIP" -v b="$BRINGUP_HOURS" \
             -v s="${steps:-$STOCK_STEPS}" -v d="$STOCK_STEPS" \
             'BEGIN{printf "%.1f", b + c*h*(s/d)}')"
  cost="$(awk -v n="$boxes" -v h="$hours" -v r="$RATE" 'BEGIN{printf "%.2f", n*h*r}')"
  total_cost="$(awk -v a="$total_cost" -v b="$cost" 'BEGIN{printf "%.2f", a+b}')"
  total_boxes=$(( total_boxes + boxes ))
  total_clips=$(( total_clips + got ))

  plan="$plan$(printf '%-4s %2d clips  %d box(es) x %s clips  %5s h/box  $%6s  %s\n' \
    "$group" "$got" "$boxes" "$per_box" "$hours" "$cost" "$why")"$'\n'
  [ -n "$stage$tier$steps" ] && plan="$plan$(printf '     %s%s%s\n' \
    "${stage:+stage $S/$stage  }" "${tier:+tier $tier  }" \
    "${steps:+steps $steps}${save:+ (checkpoint every $save)}")"$'\n'

  printf '%s\n' "$group $flags${slug:+ (slug $slug)} ${envs:-no env} -> $map" >&2
  if [ -n "$stage" ] && [ ! -d "$S/$stage" ]; then
    printf '%s\n' "   NOT STAGED YET: $group wants $S/$stage, which does not exist" >&2
  fi
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

while IFS='|' read -r group clips flags envs boxes stage tier steps why; do
  [ -n "$group" ] || continue
  selected "$group" || continue
  # STEPS:SAVE, where the two can be equal -- a 409,600-step run checkpoints once, at the end.
  case "$steps" in *:*) save="${steps#*:}" ;; *) save="" ;; esac
  steps="${steps%%:*}"
  map="$ROOT/map_$group.txt"
  mount="${stage:-sw_$group}"
  got=$(grep -c '[^[:space:]]' "$map")
  per_box=$(( (got + boxes - 1) / boxes ))

  # A group with its own `stage` was built elsewhere -- t1_precision by stage_t2_combined.sh
  # against the tier_1 retarget, w3_bridge_v2 and t3_recon by whoever owns that tree. It is used
  # as it stands, and its absence is a refusal rather than a silent fall back to the briefs.
  if [ -n "$stage" ]; then
    builder=""
    [ "$group" = CR ] && builder=" -- build it with brev/stage_cr.sh"
    [ -d "$S/$mount" ] \
      || { echo "FATAL: $group: no staged tree at $S/$mount$builder" >&2; exit 3; }
    [ -f "$S/$mount.md5" ] \
      || { echo "FATAL: $group: no manifest at $S/$mount.md5$builder" >&2; exit 3; }
    echo "== $group uses the tree already staged at $S/$mount"
  elif [ "$DRY" = 1 ]; then
    echo "== staging $group ($got clips) into $S/$mount"
    echo "   DRY: $REPO/brev/stage_t2_combined.sh --from-briefs --map $map --mount $mount"
  else
    echo "== staging $group ($got clips) into $S/$mount"
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
          --first "" --create-stagger "$STAGGER" --type "$TYPE")
  [ -n "$envs" ] && launch+=(--arm-env "$envs")
  [ -n "$tier" ] && launch+=(--tier "$tier")
  [ -n "$steps" ] && launch+=(--steps "$steps")
  [ -n "$save" ] && launch+=(--save-every "$save")
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
