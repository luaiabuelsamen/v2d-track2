#!/usr/bin/env bash
# Day-4 two-clip FlashCHORD gate: host-side launcher for NVIDIA Brev.
#
# Without --go this is read-only: it prints the live price table, the GPU it would pick and
# the commands it would run, and exits. It never creates an account, redeems a code, submits,
# posts or pushes. Luai redeems the credit and approves the run.
#
# With --go it creates ONE instance, runs brev/gate_on_instance.sh on it, copies the results
# back, and DELETES the instance on any exit path. A watchdog deletes it after --max-hours
# even if this script is killed.
#
# Usage:
#   brev/gate_launch.sh                               # dry run: prices + plan
#   brev/gate_launch.sh --phase probe --go
#   brev/gate_launch.sh --phase full --max-hours 10 --go
set -uo pipefail

PHASE=probe
MAX_HOURS=10
GO=0
TYPE=""
NAME="v2d-gate-$(date +%m%d-%H%M)"
MIN_VRAM_GB=46          # FlashCHORD needs >=46,000 MiB free on a 48 GiB card
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/results/gate"

while [ $# -gt 0 ]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --max-hours) MAX_HOURS="$2"; shift 2 ;;
    --type) TYPE="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --go) GO=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$PHASE" in probe|full) ;; *) echo "--phase must be probe or full" >&2; exit 2 ;; esac

say() { printf '\n== %s\n' "$*"; }

command -v brev >/dev/null || {
  echo "The brev CLI is not installed. Install it and run 'brev login' first:" >&2
  echo "  curl -fsSL https://brev.nvidia.com/install.sh | bash   # see docs.nvidia.com/brev" >&2
  exit 1
}

say "Brev account"
if ! brev ls >/dev/null 2>&1; then
  echo "Not logged in (or no org). Run 'brev login' and redeem the credit code first." >&2
  [ "$GO" = 1 ] && exit 1
fi
brev ls 2>/dev/null | head -20 || true

# ---------------------------------------------------------------- price discovery
say "Live Brev prices for cards that meet the ${MIN_VRAM_GB} GB floor"
SEARCH_RAW="$(brev search --sort price 2>/dev/null)" || SEARCH_RAW=""
if [ -z "$SEARCH_RAW" ]; then
  echo "WARNING: 'brev search' returned nothing; falling back to the Oct 2026 snapshot in brev/README.md."
  SEARCH_RAW=$'hyperstack_A6000\thyperstack\tA6000\t48\t0.60\nmassedcompute_A6000_plus\tmassedcompute\tA6000\t48\t0.68\nmassedcompute_L40S\tmassedcompute\tL40S\t48\t1.06\nmassedcompute_RTX6000Ada\tmassedcompute\tRTX6000Ada\t48\t1.16'
fi
printf '%s\n' "$SEARCH_RAW" | head -40

# Keep rows whose VRAM column is >= the floor, then take the cheapest. Brev's table layout has
# changed before, so this parses defensively: any row with a GB figure and a $ figure.
PICK="$(printf '%s\n' "$SEARCH_RAW" | awk -v floor="$MIN_VRAM_GB" '
  {
    vram=""; price=""; type=$1
    for (i=1;i<=NF;i++) {
      t=$i; gsub(/[$,]/,"",t)
      if (t ~ /^[0-9]+(\.[0-9]+)?$/) {
        if (t+0 >= 8 && t+0 <= 400 && t ~ /^[0-9]+$/ && vram=="") vram=t+0
        else if (t+0 < 50 && t ~ /\./ && price=="") price=t+0
      }
    }
    if (vram != "" && price != "" && vram >= floor) printf "%s %s %s\n", price, type, vram
  }' | sort -n | head -1)"

if [ -n "$TYPE" ]; then
  INSTANCE_TYPE="$TYPE"; HOURLY="unknown"; VRAM="unknown"
  echo "Using --type override: $INSTANCE_TYPE"
elif [ -n "$PICK" ]; then
  HOURLY="$(echo "$PICK" | cut -d' ' -f1)"
  INSTANCE_TYPE="$(echo "$PICK" | cut -d' ' -f2)"
  VRAM="$(echo "$PICK" | cut -d' ' -f3)"
else
  INSTANCE_TYPE="hyperstack_A6000"; HOURLY="0.60"; VRAM="48"
  echo "Could not parse a candidate; defaulting to $INSTANCE_TYPE at \$$HOURLY/h."
fi

if [ "$PHASE" = probe ]; then STEPS=10000896; EST_HOURS=0.7; else STEPS=349999104; EST_HOURS=6; fi
EST_COST="$(awk -v h="$EST_HOURS" -v r="${HOURLY/unknown/0.60}" 'BEGIN{printf "%.2f", h*r}')"

cat <<PLAN

== Plan
  phase            $PHASE  (${STEPS} environment steps per clip)
  instance type    $INSTANCE_TYPE   (${VRAM} GB, \$${HOURLY}/h)
  instance name    $NAME
  clips            e24 watering_can (easy) and e05 big_black_box (hard)
  watchdog         delete after ${MAX_HOURS} h
  rough cost       ~\$${EST_COST} at \$${HOURLY}/h for ~${EST_HOURS} h
  results to       $OUT_DIR

== Commands it would run
  brev create "$NAME" --type "$INSTANCE_TYPE"
  brev refresh && ssh "$NAME" 'bash -s' < brev/gate_on_instance.sh $PHASE $STEPS
  scp "$NAME":~/gate_results/* $OUT_DIR/
  brev delete "$NAME" --yes
PLAN

if [ "$GO" != 1 ]; then
  echo
  echo "DRY RUN. Nothing was created. Re-run with --go to execute."
  exit 0
fi

# ---------------------------------------------------------------- create, run, tear down
mkdir -p "$OUT_DIR"
CREATED=0
teardown() {
  local rc=$?
  if [ "$CREATED" = 1 ]; then
    say "Deleting $NAME"
    brev delete "$NAME" --yes 2>/dev/null || brev delete "$NAME" 2>/dev/null || {
      echo "!! COULD NOT DELETE. Run this yourself, now:" >&2
      echo "     brev delete $NAME" >&2
      echo "   then confirm with: brev ls" >&2
    }
  fi
  exit $rc
}
trap teardown EXIT INT TERM

say "Creating $NAME ($INSTANCE_TYPE)"
brev create "$NAME" --type "$INSTANCE_TYPE" || { echo "create failed" >&2; exit 1; }
CREATED=1

# Independent watchdog: deletes the instance even if this script is killed.
( sleep "$((MAX_HOURS * 3600))"; brev delete "$NAME" --yes >/dev/null 2>&1 ) &
WATCHDOG=$!
trap 'kill $WATCHDOG 2>/dev/null; teardown' EXIT INT TERM

say "Waiting for RUNNING"
for _ in $(seq 1 60); do
  brev ls 2>/dev/null | grep -q "$NAME.*RUNNING" && break
  sleep 15
done
brev refresh >/dev/null 2>&1 || true

say "Running the gate on the instance (phase=$PHASE)"
ssh -o StrictHostKeyChecking=accept-new "$NAME" 'bash -s' -- "$PHASE" "$STEPS" \
  < "$REPO_ROOT/brev/gate_on_instance.sh" 2>&1 | tee "$OUT_DIR/gate_${PHASE}_console.log"

say "Copying results back"
scp -r "$NAME":'~/gate_results/*' "$OUT_DIR/" 2>/dev/null || echo "no results copied; see the console log"

say "Done. Results in $OUT_DIR"
ls -la "$OUT_DIR"
