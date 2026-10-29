#!/usr/bin/env bash
# Re-evaluate the EXISTING wave-2 checkpoints against the clean Tier 1 trajectory. Eval only.
#
# The question this answers: our Tier 1 board entry is the Tier 2 entry's own rows, because the
# policies were driven by Tier 2's noisy object track (proved by rigid fit: every shipped rollout's
# reference is an exact rigid transform of the released Tier 2 trajectory and ~1 cm off Tier 1
# truth). Tier 1 permits the clean track as input. This drives the same policies with it.
#
# No training. One box, 30 `evaluate_policy.py` runs at world_count=1, each rebuilding its own
# trained scene from its checkpoint, with the clean parquet staged at the path the checkpoint
# already records (see brev/stage_t1_eval.sh for why that is the only lever that works).
#
# DRY RUN BY DEFAULT: without --go it prints every step and the estimate, and creates nothing.
#
#   bash brev/t1_eval_clean.sh            # dry run
#   bash brev/t1_eval_clean.sh --go       # spends money
set -uo pipefail

GO=0
[[ "${1:-}" == "--go" ]] && GO=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export V2D_SCRATCH="${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}"
S=$V2D_SCRATCH
ASSETS=${ASSETS:-$S/t1_eval/assets}
PLAN=${PLAN:-$S/t1_eval/assets.sources.txt}
OUT=${OUT:-$S/t1_eval/results}
NAME=${NAME:-v2d-t1eval-01}
INSTANCE_TYPE=${INSTANCE_TYPE:-massedcompute_A6000_plus}
RATE_USD_PER_HOUR=${RATE_USD_PER_HOUR:-0.68}
BOX_DEADLINE_H=${BOX_DEADLINE_H:-2.5}
WATCHDOG_H=${WATCHDOG_H:-3.2}
MAX_USD=$(awk -v h="$WATCHDOG_H" -v r="$RATE_USD_PER_HOUR" 'BEGIN{printf "%.2f", h*r}')
LOG=$OUT/driver.log
mkdir -p "$OUT"

say() { printf '\n== %s\n' "$*"; }
# This instance type refuses sessions intermittently -- "kex_exchange_identification: Connection
# closed by remote host" killed a whole run after the 1 GB upload had already succeeded. Every
# remote call goes through here.
rssh() {
  local i
  for i in 1 2 3 4 5 6; do
    ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 "$NAME" "$1" && return 0
    sleep $((i * 15))
  done
  echo "!! ssh to $NAME kept failing: $1" >&2
  return 1
}
step() {
  local desc=$1; shift; [[ ${1:-} == -- ]] && shift
  printf '   %-34s %s\n' "$desc" "$*"
  [[ $GO == 1 ]] || return 0
  "$@" >>"$LOG" 2>&1 || { echo "!! FAILED: $desc (see $LOG)" >&2; return 1; }
}

# --------------------------------------------------------------------------- preflight (free)
say "Preflight (free)"
PRE_OK=1
STAGED=$(find "$ASSETS" -name data.parquet 2>/dev/null | wc -l)
printf '   %-15s %s\n' "staged refs" "$STAGED/30 clean Tier 1 parquets in $ASSETS"
[[ $STAGED == 30 ]] || { echo "   staged refs     INCOMPLETE -- run brev/stage_t1_eval.sh first"; PRE_OK=0; }
CKPTS=$(find "$ASSETS/ckpt" -name policy.safetensors 2>/dev/null | wc -l)
printf '   %-15s %s\n' "checkpoints" "$CKPTS/30"
[[ $CKPTS == 30 ]] || PRE_OK=0
[[ -f "$ASSETS/MD5SUMS" ]] && printf '   %-15s %s\n' "manifest" "$(wc -l < "$ASSETS/MD5SUMS") lines" \
  || { echo "   manifest        MISSING"; PRE_OK=0; }
[[ -f "$PLAN" ]] && printf '   %-15s %s\n' "plan" "$(grep -c . "$PLAN") clips" \
  || { echo "   plan            MISSING"; PRE_OK=0; }
~/.local/bin/brev ls >/dev/null 2>&1 && printf '   %-15s %s\n' "brev" "logged in" \
  || { echo "   brev            NOT logged in"; PRE_OK=0; }

say "Estimate (one $INSTANCE_TYPE at \$$RATE_USD_PER_HOUR/h)"
cat <<EST
   create + SSH ready      8 min
   image build            10 min  (cached per box, first build is the slow one)
   upload assets           3 min  ($(du -sh "$ASSETS" 2>/dev/null | cut -f1) staged tree + 30 checkpoints)
   30 evaluations         60 min  (1-2 min/clip once warp's kernels are cached; the insurance run
                                   did train AND eval for 30 clips in 114 box-minutes)
   pull + delete           3 min
   TOTAL                 ~1.4 h  ->  \$$(awk -v r="$RATE_USD_PER_HOUR" 'BEGIN{printf "%.2f", 1.4*r}')
   hard caps       box deadline ${BOX_DEADLINE_H} h, backstop delete ${WATCHDOG_H} h = \$$MAX_USD
EST

# --------------------------------------------------------------------------- the run
say "Steps ($([[ $GO == 1 ]] && echo EXECUTING || echo 'dry run: printed, not executed'))"
if [[ $GO == 1 && $PRE_OK != 1 ]]; then
  echo "!! preflight failed; refusing to create a box" >&2; exit 2
fi

source "$REPO_ROOT/brev/budget_guard.sh"
step "1  budget guard"                 -- budget_ok_for_hours "$WATCHDOG_H" "$NAME"
arm_backstop() {  # detached: outlives this script, deletes the box at WATCHDOG_H regardless
  setsid nohup bash "$REPO_ROOT/brev/backstop_watchdog.sh" "$NAME" \
    "$(awk -v h="$WATCHDOG_H" 'BEGIN{printf "%d", h*3600}')" "$LOG" </dev/null >/dev/null 2>&1 &
}
step "2  arm backstop (detached)"      -- arm_backstop
CREATED=0; T0=$SECONDS
teardown() {
  if [[ $CREATED == 1 ]]; then
    for _ in 1 2 3 4 5; do ~/.local/bin/brev delete "$NAME" </dev/null >>"$LOG" 2>&1 && break; sleep 60; done
    ab4_ledger_add "$NAME" "$(( (SECONDS - T0) / 60 + 1 ))"
    ~/.local/bin/brev ls 2>/dev/null | grep -q "^[[:space:]]*$NAME[[:space:]]" \
      && echo "!! $NAME STILL LISTED -- run: brev delete $NAME" >&2
  fi
}
[[ $GO == 1 ]] && trap teardown EXIT INT TERM
step "3  create box"                   -- ~/.local/bin/brev create "$NAME" --type "$INSTANCE_TYPE"
[[ $GO == 1 ]] && CREATED=1
wait_ssh() {
  for _ in $(seq 1 80); do
    ~/.local/bin/brev refresh >/dev/null 2>&1
    ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes "$NAME" true \
      >/dev/null 2>&1 && return 0
    sleep 15
  done
  echo "!! $NAME never accepted SSH" >&2; return 1
}
step "4  wait for SSH (<=20 min)"      -- wait_ssh

upload() {
  ssh -o BatchMode=yes "$NAME" 'mkdir -p ~/v2d_assets' || return 1
  # The CONTENTS of the staged tree, so /v2d/assets/<stage>/... matches what the checkpoints record.
  rsync -a --info=stats1 -e "ssh -o ServerAliveInterval=30" "$ASSETS"/ "$NAME":~/v2d_assets/ || return 1
  scp -o ServerAliveInterval=30 "$PLAN" "$NAME":~/t1_plan.txt || return 1
  # Everything the box runner sources or calls, or it dies on the box with "No such file or
  # directory" after the image has already been paid for -- which is how the first wave of the
  # 30-clip run was lost.
  scp -o ServerAliveInterval=30 "$REPO_ROOT/brev/gate_on_instance.sh" \
      "$REPO_ROOT/brev/t1_eval_box.sh" "$REPO_ROOT/brev/arm_flags.sh" \
      "$REPO_ROOT/brev/patch_object_orientation.py" "$NAME":~/ || return 1
}
step "5  upload assets + runners"      -- upload
# The manifest gate, on the box, before any GPU time is spent on a tree that travelled badly.
step "6  verify the manifest on the box" -- ssh -o BatchMode=yes "$NAME" \
       'cd ~/v2d_assets && md5sum -c --quiet MD5SUMS && echo "manifest ok: $(wc -l < MD5SUMS) files"'
# Detached on the box, with a marker, then polled: the work outlives any ssh session, which is
# the only way to survive this instance type's refusals. Same shape as the Tier 3 runner.
start_remote() {
  rssh 'cat > ~/t1_runner.sh <<"EOS"
#!/usr/bin/env bash
set -uo pipefail
R=$HOME/t1_eval_results
mkdir -p "$R"; rm -f "$R/FINISHED" "$R/FAILED"
bash ~/gate_on_instance.sh build > "$R/build.log" 2>&1 || { echo build >"$R/FAILED"; exit 1; }
PLAN=~/t1_plan.txt DEADLINE_H=__DEADLINE__ bash ~/t1_eval_box.sh > "$R/run.log" 2>&1 \
  || { echo evaluate >"$R/FAILED"; exit 1; }
touch "$R/FINISHED"
EOS
sed -i s/__DEADLINE__/'"$BOX_DEADLINE_H"'/ ~/t1_runner.sh
setsid nohup bash ~/t1_runner.sh </dev/null >/dev/null 2>&1 &
echo started'
}
wait_remote() {
  local waited=0 limit
  limit=$(awk -v h="$BOX_DEADLINE_H" 'BEGIN{printf "%d", h*3600 + 1800}')
  while [ "$waited" -lt "$limit" ]; do
    if rssh 'test -f ~/t1_eval_results/FINISHED' 2>/dev/null; then return 0; fi
    if rssh 'test -f ~/t1_eval_results/FAILED' 2>/dev/null; then
      rssh 'echo "stage: $(cat ~/t1_eval_results/FAILED)"; tail -25 ~/t1_eval_results/run.log 2>/dev/null \
            || tail -25 ~/t1_eval_results/build.log' >&2
      return 1
    fi
    sleep 120; waited=$((waited + 120))
    rssh 'echo "  progress: $(ls ~/t1_eval_results/rollouts/*.parquet 2>/dev/null | wc -l)/30 clips"' || true
  done
  echo "!! remote run did not finish inside ${limit}s" >&2; return 1
}
step "7  start build+eval detached"    -- start_remote

# A failure here used to fall through to "pull" and "delete", which pulled nothing and deleted the
# evidence: the first run died on every clip with an unexpected-keyword error and still tore the
# box down. Keep the box, inside the backstop, and say how to look.
if ! step "8  wait for the 30 evaluations" -- wait_remote; then
  if [[ $GO == 1 ]]; then
    # Disarm first: `exit` fires the EXIT trap, which calls teardown, which deletes the box this
    # branch exists to keep. That bug ate the evidence from the second attempt.
    CREATED=0; trap - EXIT INT TERM
    echo "!! evaluation failed. The box $NAME is KEPT (backstop deletes it at ${WATCHDOG_H} h)." >&2
    echo "   Look with:  ssh $NAME 'tail -40 ~/t1_eval_results/eval_*.log'" >&2
    exit 3
  fi
fi

# Pull before deleting, and only delete if something came back: an empty pull with a live box is
# recoverable, an empty pull with a deleted box is not.
step "9  pull results"                 -- rsync -a -e "ssh -o ServerAliveInterval=30" \
       "$NAME":~/t1_eval_results/ "$OUT/"
if [[ $GO == 1 ]]; then
  PULLED=$(find "$OUT/rollouts" -name 'episode_*.parquet' 2>/dev/null | wc -l)
  if [[ $PULLED == 0 ]]; then
    CREATED=0; trap - EXIT INT TERM
    echo "!! pulled 0 rollouts; KEEPING $NAME so the run can be inspected or resumed" >&2
    exit 3
  fi
fi
step "10 delete box"                   -- teardown
[[ $GO == 1 ]] && { CREATED=0; trap - EXIT INT TERM; }

if [[ $GO == 1 ]]; then
  COUNT=$(find "$OUT/rollouts" -name 'episode_*.parquet' 2>/dev/null | wc -l)
  say "Pulled $COUNT of 30 clips to $OUT/rollouts"
  echo "Next, free on the Jetson: kit-score these against the clean truth and compare with 0.46523."
else
  say "Exact command"
  echo "   bash $REPO_ROOT/brev/t1_eval_clean.sh --go"
  [[ $PRE_OK == 1 ]] || echo "   (would refuse right now: preflight failed above)"
fi
