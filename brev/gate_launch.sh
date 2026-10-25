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
#   brev/gate_launch.sh --phase probe --max-usd 2 --go
#   brev/gate_launch.sh --phase full --max-hours 10 --go
set -uo pipefail

PHASE=probe
MAX_HOURS=10
MAX_USD=0               # 0 = no cap. With a cap, the watchdog is bounded by it too.
GO=0
TYPE=""
G1_PARQUET=""           # retargeted G1 reference for the gate clip; required for --phase full
EVAL_EP=""              # B1: also evaluate the probe's checkpoint on this episode
EVAL_WORLDS=64
NAME="v2d-gate-$(date +%m%d-%H%M)"
MIN_VRAM_GB=46          # FlashCHORD needs >=46,000 MiB free on a 48 GiB card
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/results/gate"

while [ $# -gt 0 ]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --max-hours) MAX_HOURS="$2"; shift 2 ;;
    --max-usd) MAX_USD="$2"; shift 2 ;;
    --type) TYPE="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --g1-parquet) G1_PARQUET="$2"; shift 2 ;;
    --eval-episode) EVAL_EP="$2"; shift 2 ;;
    --eval-worlds) EVAL_WORLDS="$2"; shift 2 ;;
    --allow-raw-reference) ALLOW_RAW=1; shift ;;
    # Check the reference provenance and exit, touching nothing else. Added because verifying
    # the guard by running the script reaches the brev CLI, which prompts to log in; the guard
    # is testable without going anywhere near an account.
    --check-provenance-only) CHECK_ONLY=1; shift ;;
    # Keep the instance after this script exits, so a debugging session does not pay for
    # provisioning (~6 min) and a Docker build (~6 min) on every script bug. Four attempts went
    # that way. The watchdog still deletes it, and is detached so it outlives this process.
    --dev-box) DEV_BOX=1; shift ;;
    --go) GO=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$PHASE" in probe|full|build) ;; *) echo "--phase must be build, probe or full" >&2; exit 2 ;; esac
if [ "$PHASE" = full ] && [ -z "$G1_PARQUET" ]; then
  echo "--phase full needs --g1-parquet <retargeted G1 parquet for the gate clip>." >&2
  echo "Produce it with soma_to_g1.py first; the probe phase does not need it." >&2
  exit 2
fi
[ -n "$G1_PARQUET" ] && [ ! -f "$G1_PARQUET" ] && { echo "no such file: $G1_PARQUET" >&2; exit 2; }

# The gate must train on a cleaned reference. The support-contact cleaner lifts the Tier 2 input
# ceiling from AUC 0.4247 to 0.4660 and SP-SR to 1.000 (results/cleanup_ladder.json), so a run
# against the raw noisy reference spends money measuring the wrong thing. retarget_all.py drops a
# provenance.json at the motion root; walk up from the parquet to find it.
if [ -n "$G1_PARQUET" ] && [ "${ALLOW_RAW:-0}" != 1 ]; then
  prov=""; dir="$(cd "$(dirname "$G1_PARQUET")" && pwd)"
  while [ "$dir" != "/" ]; do
    [ -f "$dir/provenance.json" ] && { prov="$dir/provenance.json"; break; }
    dir="$(dirname "$dir")"
  done
  if [ -z "$prov" ]; then
    echo "No provenance.json above $G1_PARQUET, so I cannot tell which reference it was" >&2
    echo "retargeted from. Re-run scripts/retarget_all.py (it writes one), or pass" >&2
    echo "--allow-raw-reference to override deliberately." >&2
    exit 2
  fi
  if ! grep -q '"cleaned_reference": true' "$prov"; then
    echo "$prov says this motion came from the raw released reference:" >&2
    sed -n 's/^/    /p' "$prov" >&2
    echo "Retarget from the cleaned reference first:" >&2
    echo "    python scripts/clean_support_contact.py --dataset-root DATA --npz-out NPZ_CLEAN \\" >&2
    echo "        --support-primitives" >&2
    echo "    python scripts/retarget_all.py ... --object-npz NPZ_CLEAN" >&2
    echo "Or pass --allow-raw-reference to override deliberately." >&2
    exit 2
  fi
  # Print both halves of the provenance. The object reference is what the guard enforces; the
  # human root is reported so a run is never ambiguous about which cleaners produced it.
  python3 -c 'import json,sys
r=json.load(open(sys.argv[1]))
print("reference provenance: object=%s" % r["object_source"])
if "human_source" in r:
    print("                      human =%s%s" % (
        r["human_source"], "" if r.get("cleaned_human") else "  (released, not cleaned)"))
' "$prov"
fi
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  echo "provenance check only; not contacting Brev."
  exit 0
fi

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

# Pick the GPU with `v2d brev-pick`, which parses the table by its header offsets and is
# covered by tests/test_brev.py. The awk that used to live here guessed which column was which
# and guessed wrong: it read TARGET_DISK as the VRAM and CAPABILITY as the price, so a 16 GB
# V100 at $2.81/h sorted ahead of a 48 GB A6000 and the dry run proposed a card that cannot run
# the job. There is deliberately no fallback default -- guessing is what broke it.
V2D_BIN="$REPO_ROOT/.venv311/bin/v2d"
[ -x "$V2D_BIN" ] || V2D_BIN="$(command -v v2d || true)"
if [ -z "$V2D_BIN" ]; then
  echo "v2d is not installed; run 'uv pip install -e .' in $REPO_ROOT" >&2
  exit 2
fi

PICK_ARGS=(brev-pick --min-vram "$MIN_VRAM_GB" --show 5)
[ -n "$TYPE" ] && PICK_ARGS+=(--type "$TYPE")
if ! PICK="$("$V2D_BIN" "${PICK_ARGS[@]}")"; then
  echo "No usable GPU was chosen, so nothing will be created." >&2
  exit 2
fi
HOURLY="$(echo "$PICK" | cut -d' ' -f1)"
INSTANCE_TYPE="$(echo "$PICK" | cut -d' ' -f2)"
VRAM="$(echo "$PICK" | cut -d' ' -f3)"
if [ -z "$INSTANCE_TYPE" ] || [ -z "$HOURLY" ]; then
  echo "brev-pick returned nothing usable: '$PICK'" >&2
  exit 2
fi
# Belt and braces: the floor is enforced in the picker and again here, because this is the line
# between a $0.60 card that works and a $7 card that cannot.
if awk -v v="$VRAM" -v f="$MIN_VRAM_GB" 'BEGIN{exit !(v+0 < f+0)}'; then
  echo "refusing $INSTANCE_TYPE: ${VRAM} GB per GPU is below the ${MIN_VRAM_GB} GB floor" >&2
  exit 2
fi

B1_DESCRIPTION="none"
[ -n "$EVAL_EP" ] && B1_DESCRIPTION="e$EVAL_EP at ${EVAL_WORLDS} worlds"
case "$PHASE" in
  # build: provisioning plus one Docker image build, measured at 229-336 s over four attempts.
  build) STEPS=0;         EST_HOURS=0.4 ;;
  probe) STEPS=10000384;  EST_HOURS=0.7 ;;
  full)  STEPS=349999104; EST_HOURS=6   ;;
esac
EST_COST="$(awk -v h="$EST_HOURS" -v r="$HOURLY" 'BEGIN{printf "%.2f", h*r}')"
# The watchdog is bounded by the money cap as well as by --max-hours, so a hung run cannot spend
# past the cap even if this script dies and the estimate was wrong.
WATCHDOG_HOURS="$MAX_HOURS"
if awk -v c="$MAX_USD" 'BEGIN{exit !(c+0 > 0)}'; then
  CAP_HOURS="$(awk -v c="$MAX_USD" -v r="$HOURLY" 'BEGIN{printf "%.4f", c/r}')"
  WATCHDOG_HOURS="$(awk -v a="$MAX_HOURS" -v b="$CAP_HOURS" 'BEGIN{print (a<b)?a:b}')"
  if awk -v e="$EST_COST" -v c="$MAX_USD" 'BEGIN{exit !(e+0 > c+0)}'; then
    echo >&2
    echo "ABORT: the estimate is \$$EST_COST but --max-usd is \$$MAX_USD." >&2
    echo "Nothing was created. Raise --max-usd, shorten the phase, or pick a cheaper card." >&2
    exit 3
  fi
fi
WATCHDOG_SECONDS="$(awk -v h="$WATCHDOG_HOURS" 'BEGIN{printf "%d", h*3600}')"

cat <<PLAN

== Plan
  phase            $PHASE  (${STEPS} environment steps per clip)
  instance type    $INSTANCE_TYPE   (${VRAM} GB, \$${HOURLY}/h)
  instance name    $NAME
  clips            e24 watering_can (easy) and e05 big_black_box (hard)
  watchdog         delete after ${WATCHDOG_HOURS} h
  rough cost       ~\$${EST_COST} at \$${HOURLY}/h for ~${EST_HOURS} h
  hard cost cap    ${MAX_USD:+\$}${MAX_USD:-none}
  B1 eval          $B1_DESCRIPTION
  results to       $OUT_DIR

== Commands it would run
  brev create "$NAME" --type "$INSTANCE_TYPE"
  brev refresh && ssh "$NAME" 'bash -s' < brev/gate_on_instance.sh $PHASE $STEPS [g1-parquet]
  scp "$NAME":~/gate_results/* $OUT_DIR/
  brev delete "$NAME"
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
    brev delete "$NAME" 2>/dev/null || {
      echo "!! COULD NOT DELETE. Run this yourself, now:" >&2
      echo "     brev delete $NAME" >&2
      echo "   then confirm with: brev ls" >&2
    }
  fi
  exit $rc
}
trap teardown EXIT INT TERM

# Armed BEFORE the box exists, deliberately. `brev create` blocks until the instance is ready,
# about six minutes, and arming after it left that whole window uncovered: kill the launcher
# during the build and the box bills until someone notices. Deleting by name needs no instance,
# so arming early costs nothing -- if create fails, the delete simply finds nothing to do.
# Independent watchdog: deletes the instance even if this script is killed.
#
# `brev delete` takes names POSITIONALLY and has no --yes flag. Passing one makes the CLI exit
# with "unknown flag: --yes" having deleted nothing, so every watchdog here was a silent no-op
# until this was found -- the teardown's `|| brev delete "$NAME"` fallback had been masking it.
# It also retries, because hammering the API earns "rate limit exceeded" and the call just fails. In dev-box mode it
# is fully detached, because it has to outlive this process -- it is the only thing standing
# between a forgotten box and a day of A6000 billing.
if [ "${DEV_BOX:-0}" = 1 ]; then
  setsid nohup bash -c "sleep $WATCHDOG_SECONDS; \
    for _ in 1 2 3 4 5; do brev delete '$NAME' </dev/null >/dev/null 2>&1 && break; sleep 60; done" \
    >/dev/null 2>&1 < /dev/null &
  echo "detached watchdog armed: deletes $NAME in ${WATCHDOG_HOURS} h"
  # No teardown trap: the point of this mode is that the box survives.
  trap - EXIT INT TERM
else
  ( sleep "$WATCHDOG_SECONDS"
    for _ in 1 2 3 4 5; do brev delete "$NAME" </dev/null >/dev/null 2>&1 && break; sleep 60; done ) &
  WATCHDOG=$!
  trap 'kill $WATCHDOG 2>/dev/null; teardown' EXIT INT TERM
fi

say "Creating $NAME ($INSTANCE_TYPE)"
brev create "$NAME" --type "$INSTANCE_TYPE" || { echo "create failed" >&2; exit 1; }
CREATED=1


# STATUS=RUNNING is not the question, and assuming it was cost the first probe run: brev reports
# RUNNING while the SHELL column still reads NOT READY, ssh then dies with
# "kex_exchange_identification: Connection closed by remote host", and the run tears down having
# measured nothing. Wait for the shell, then prove it with an actual ssh before sending the
# payload. The readiness predicate lives in v2d.brev so it can be tested (tests/test_brev.py).
say "Waiting for the shell to accept SSH"
brev refresh >/dev/null 2>&1 || true
READY=0
WAITED=0
for _ in $(seq 1 80); do          # 80 x 15 s = 20 minutes
  ROW="$(brev ls 2>/dev/null | grep -E "^[[:space:]]*${NAME}[[:space:]]" || true)"
  if [ -n "$ROW" ] && "$V2D_BIN" brev-ready --row "$ROW" --dead-check >/dev/null 2>&1; then
    echo "!! the instance reached a state it will not return from:" >&2
    echo "   $ROW" >&2
    exit 4
  fi
  if [ -n "$ROW" ] && "$V2D_BIN" brev-ready --row "$ROW" >/dev/null 2>&1; then
    brev refresh >/dev/null 2>&1 || true
    if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes \
           "$NAME" true >/dev/null 2>&1; then
      READY=1; break
    fi
  fi
  sleep 15
  WAITED=$((WAITED + 15))
done
if [ "$READY" != 1 ]; then
  echo "!! the instance never accepted SSH after ${WAITED}s; tearing down without running." >&2
  exit 4
fi
say "SSH is up after ~${WAITED}s"

# Belt and braces for the LFS problem. gate_on_instance.sh resolves pointers from GitHub's
# /raw/<sha>/ endpoint, which is the primary mechanism and covers the ONNX policies too; this
# ships this host's already-resolved meshes as well, because they are known good (they are what
# retargeted all 60 clips) and the cost is one 23 MB transfer.
G1_MESHES="$(find "$REPO_ROOT" /tmp/claude-1000 -maxdepth 12 -type d -path "*assets/meshes/g1" \
  2>/dev/null | head -1)"
if [ -n "$G1_MESHES" ] && [ -z "$(grep -rl "git-lfs.github.com/spec" "$G1_MESHES" 2>/dev/null)" ]; then
  say "Shipping resolved G1 meshes ($(du -sh "$G1_MESHES" | cut -f1))"
  tar -C "$(dirname "$G1_MESHES")" -czf - g1 \
    | ssh "${SSH_KEEPALIVE[@]}" "$NAME" 'cat > /tmp/g1_meshes.tgz' </dev/null
  ssh "${SSH_KEEPALIVE[@]}" "$NAME" 'bash -s' </dev/null <<'REMOTE'
set -uo pipefail
SRC=~/v2d/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/meshes
FC=~/v2d/robotic_grounding/flash_chord/src/flash_chord/assets/meshes
[ -d "$SRC" ] || { echo "no source mesh tree on the instance yet" >&2; exit 0; }
tar -C "$SRC" -xzf /tmp/g1_meshes.tgz
# The g1_dex3 embodiment loads the flash_chord copy, so mirror by file name into it.
copied=0
while IFS= read -r f; do
  base=$(basename "$f")
  src=$(find "$SRC/g1" -name "$base" -type f | head -1)
  [ -n "$src" ] && { cp -f "$src" "$f"; copied=$((copied+1)); }
done < <(find "$FC/g1" -type f \( -name "*.STL" -o -name "*.stl" -o -name "*.obj" \) 2>/dev/null)
echo "resolved meshes installed: source=$(find "$SRC/g1" -type f | wc -l) flash_chord=$copied"
REMOTE
else
  echo "WARNING: no resolved G1 mesh tree found locally; the instance must resolve LFS itself." >&2
fi

REMOTE_PARQUET=""
if [ -n "$G1_PARQUET" ]; then
  say "Uploading the retargeted reference"
  # It has to land inside the directory the container mounts at /v2d/assets, or the training
  # step cannot see it.
  ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$NAME" 'mkdir -p ~/v2d_assets'
  scp -o StrictHostKeyChecking=accept-new "$G1_PARQUET" "$NAME":'~/v2d_assets/gate_reference.parquet'
  REMOTE_PARQUET=/v2d/assets/gate_reference.parquet
fi

# Run it DETACHED and poll, rather than holding the work on one ssh channel. Attempt 3 was lost
# exactly there: the image built, the container came up, jax saw the GPU, and then the session
# died during the first long silent step (the hydra config resolve), taking the run with it. A
# detached job survives a dropped link, and the poll reconnects.
SSH_KEEPALIVE=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30
               -o ServerAliveCountMax=10 -o ConnectTimeout=30)
say "Uploading the gate script"
scp "${SSH_KEEPALIVE[@]}" "$REPO_ROOT/brev/gate_on_instance.sh" "$NAME":'~/gate_on_instance.sh'

say "Starting the gate detached (phase=$PHASE)"
ssh "${SSH_KEEPALIVE[@]}" "$NAME" \
  "mkdir -p ~/gate_results && rm -f ~/gate_results/DONE && \
   setsid nohup bash ~/gate_on_instance.sh '$PHASE' '$STEPS' '$REMOTE_PARQUET' \
     '$EVAL_EP' '$EVAL_WORLDS' > ~/gate_results/launch.log 2>&1 < /dev/null & \
   echo started \$!" </dev/null

say "Polling until it finishes (watchdog still bounds this at ${WATCHDOG_HOURS} h)"
POLL_LIMIT="$(awk -v h="$WATCHDOG_HOURS" 'BEGIN{printf "%d", (h*3600)/30}')"
SEEN=0
for _ in $(seq 1 "$POLL_LIMIT"); do
  if ssh "${SSH_KEEPALIVE[@]}" "$NAME" 'test -f ~/gate_results/DONE' </dev/null 2>/dev/null; then
    SEEN=1; break
  fi
  # Show the tail so a stuck run is visible rather than silent.
  ssh "${SSH_KEEPALIVE[@]}" "$NAME" 'tail -3 ~/gate_results/on_instance.log 2>/dev/null' \
    </dev/null 2>/dev/null | sed 's/^/   | /' || true
  sleep 30
done
[ "$SEEN" = 1 ] || echo "!! the gate did not finish before the watchdog bound; copying what exists." >&2

ssh "${SSH_KEEPALIVE[@]}" "$NAME" 'cat ~/gate_results/on_instance.log 2>/dev/null' \
  </dev/null > "$OUT_DIR/gate_${PHASE}_console.log" 2>/dev/null || true

say "Copying results back"
scp -r -o StrictHostKeyChecking=accept-new "$NAME":'~/gate_results/*' "$OUT_DIR/" 2>/dev/null || echo "no results copied; see the console log"

if [ "${DEV_BOX:-0}" = 1 ]; then
  cat <<DEV

== Dev box is up and NOT scheduled for deletion by this script
  ssh $NAME
  container:  ssh $NAME "docker ps --filter name=flash_chord"
  run a step: ssh $NAME "docker exec -i \$(docker ps -q --filter name=flash_chord | head -1) bash -lc 'CMD'"
  watchdog:   deletes it in ${WATCHDOG_HOURS} h (~\$${MAX_USD} at \$${HOURLY}/h)
  DELETE IT:  brev delete $NAME && brev ls
DEV
  exit 0
fi

say "Done. Results in $OUT_DIR"
ls -la "$OUT_DIR"
