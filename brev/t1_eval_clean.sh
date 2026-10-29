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
# massedcompute_A6000_plus disappeared from the catalog mid-run on 2026-10-29 and three creates
# failed with "not a recognized type" before the message was read. hyperstack_A6000 is the cheapest
# 48 GB card still listed, and cheaper than what we had been paying.
INSTANCE_TYPE=${INSTANCE_TYPE:-hyperstack_A6000}
RATE_USD_PER_HOUR=${RATE_USD_PER_HOUR:-0.60}
BOX_DEADLINE_H=${BOX_DEADLINE_H:-2.5}
WATCHDOG_H=${WATCHDOG_H:-3.2}
# Hooks, so a second job that is "create a box, upload a staged tree, run one script per clip,
# pull, delete" reuses this driver instead of forking it. Four attempts' worth of fixes live in
# here (SSH refusals, docker exec eating the plan, the EXIT trap deleting a kept box), and a fork
# would not inherit the next one.
EXPECT_CLIPS=${EXPECT_CLIPS:-30}          # preflight: refs and checkpoints both have to match
BOX_SCRIPT=${BOX_SCRIPT:-t1_eval_box.sh}  # the per-clip runner, uploaded and run on the box
BOX_ENV=${BOX_ENV:-}                      # extra VAR=VAL passed to it (no spaces in values)
EXTRA_UPLOADS=${EXTRA_UPLOADS:-}          # whitespace-separated repo-relative files it sources
WORK_LABEL=${WORK_LABEL:-evaluations}
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
printf '   %-15s %s\n' "staged refs" "$STAGED/$EXPECT_CLIPS parquets in $ASSETS"
[[ $STAGED == "$EXPECT_CLIPS" ]] || { echo "   staged refs     INCOMPLETE -- stage first"; PRE_OK=0; }
CKPTS=$(find "$ASSETS/ckpt" -name policy.safetensors 2>/dev/null | wc -l)
printf '   %-15s %s\n' "checkpoints" "$CKPTS/$EXPECT_CLIPS"
[[ $CKPTS == "$EXPECT_CLIPS" ]] || PRE_OK=0
[[ -f "$ASSETS/MD5SUMS" ]] && printf '   %-15s %s\n' "manifest" "$(wc -l < "$ASSETS/MD5SUMS") lines" \
  || { echo "   manifest        MISSING"; PRE_OK=0; }
[[ -f "$PLAN" ]] && printf '   %-15s %s\n' "plan" "$(grep -c . "$PLAN") clips" \
  || { echo "   plan            MISSING"; PRE_OK=0; }
~/.local/bin/brev ls >/dev/null 2>&1 && printf '   %-15s %s\n' "brev" "logged in" \
  || { echo "   brev            NOT logged in"; PRE_OK=0; }
# An instance type can vanish from the catalog between runs, and then `brev create` fails with
# "not a recognized type" AFTER the budget guard has committed the spend and the backstop is armed.
if ~/.local/bin/brev search 2>/dev/null | grep -qE "^[[:space:]]*$INSTANCE_TYPE[[:space:]]"; then
  printf '   %-15s %s\n' "instance type" "$INSTANCE_TYPE is in the catalog"
else
  echo "   instance type   $INSTANCE_TYPE IS NOT IN THE CATALOG -- 48 GB cards still listed:"
  ~/.local/bin/brev search 2>/dev/null | awk '$6 >= 48 {printf "                     %-32s %s\n", $1, $NF}' | head -5
  PRE_OK=0
fi

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

# Every other remote call goes through rssh, which retries this instance type's intermittent
# "kex_exchange_identification: Connection closed by remote host". The upload did not, and on
# v2d-t3ws-a it cost the whole box: the 1 GB rsync of the asset tree succeeded, the scp of the
# runner scripts was refused, the driver carried on, and the box sat at 0% GPU reporting
# "gate_on_instance.sh: No such file or directory". Retried now, per transfer.
retry() {
  local label=$1 i; shift
  for i in 1 2 3 4 5; do
    "$@" && return 0
    echo "   $label refused (attempt $i); retrying" >&2
    sleep $((i * 15))
  done
  echo "!! $label failed five times" >&2
  return 1
}
upload() {
  retry "mkdir" ssh -o BatchMode=yes -o ConnectTimeout=30 "$NAME" 'mkdir -p ~/v2d_assets' || return 1
  # The CONTENTS of the staged tree, so /v2d/assets/<stage>/... matches what the checkpoints record.
  retry "asset rsync" rsync -a --info=stats1 -e "ssh -o ServerAliveInterval=30" \
      "$ASSETS"/ "$NAME":~/v2d_assets/ || return 1
  retry "plan scp" scp -o ServerAliveInterval=30 -o ConnectTimeout=30 \
      "$PLAN" "$NAME":~/t1_plan.txt || return 1
  # Everything the box runner sources or calls, or it dies on the box with "No such file or
  # directory" after the image has already been paid for -- which is how the first wave of the
  # 30-clip run was lost.
  local extra=()
  for f in $EXTRA_UPLOADS; do
    [[ -f "$REPO_ROOT/$f" ]] || { echo "!! EXTRA_UPLOADS: no $REPO_ROOT/$f" >&2; return 1; }
    extra+=("$REPO_ROOT/$f")
  done
  retry "runner scp" scp -o ServerAliveInterval=30 -o ConnectTimeout=30 \
      "$REPO_ROOT/brev/gate_on_instance.sh" \
      "$REPO_ROOT/brev/$BOX_SCRIPT" "$REPO_ROOT/brev/arm_flags.sh" \
      "$REPO_ROOT/brev/patch_object_orientation.py" "${extra[@]}" "$NAME":~/ || return 1
  # And prove every one of them arrived, because a partial upload is what actually happened.
  retry "runner check" ssh -o BatchMode=yes -o ConnectTimeout=30 "$NAME" \
      "test -s ~/gate_on_instance.sh && test -s ~/$BOX_SCRIPT && test -s ~/arm_flags.sh" || return 1
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
PLAN=~/t1_plan.txt OUT="$R" DEADLINE_H=__DEADLINE__ __BOXENV__ bash ~/__BOXSCRIPT__ > "$R/run.log" 2>&1 \
  || { echo evaluate >"$R/FAILED"; exit 1; }
touch "$R/FINISHED"
EOS
sed -i s/__DEADLINE__/'"$BOX_DEADLINE_H"'/ ~/t1_runner.sh
sed -i s@__BOXSCRIPT__@'"$BOX_SCRIPT"'@ ~/t1_runner.sh
# QUOTED, and it has to be: BOX_ENV holds several VAR=VAL pairs separated by spaces, so an
# unquoted sed expression splits and sed treats the remaining pairs as filenames. It then fails
# while the placeholder survives, and the runner dies on `__BOXENV__: command not found` AFTER the
# image build has already been paid for. That is exactly what happened on v2d-t1ws-01.
sed -i "s@__BOXENV__@'"$BOX_ENV"'@" ~/t1_runner.sh
grep -q __BOXENV__ ~/t1_runner.sh && { echo "FATAL: BOX_ENV never substituted" >&2; exit 3; }
grep -q __BOXSCRIPT__ ~/t1_runner.sh && { echo "FATAL: BOX_SCRIPT never substituted" >&2; exit 3; }
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
    rssh 'echo "  progress: $(ls ~/t1_eval_results/rollouts/*.parquet 2>/dev/null | wc -l)/'"$EXPECT_CLIPS"' clips"' || true
  done
  echo "!! remote run did not finish inside ${limit}s" >&2; return 1
}
step "7  start build+eval detached"    -- start_remote

# A failure here used to fall through to "pull" and "delete", which pulled nothing and deleted the
# evidence: the first run died on every clip with an unexpected-keyword error and still tore the
# box down. Keep the box, inside the backstop, and say how to look.
if ! step "8  wait for the $EXPECT_CLIPS $WORK_LABEL" -- wait_remote; then
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
  WANTED=$(grep -c . "$PLAN")
  # "> 0" is not the bar. A run that exported 1 of 30 and reported success deleted its own box
  # under that test; the box is only disposable once every planned clip is home.
  if [[ $PULLED -lt $WANTED ]]; then
    CREATED=0; trap - EXIT INT TERM
    echo "!! pulled $PULLED of $WANTED rollouts; KEEPING $NAME to finish or inspect" >&2
    echo "   resume with: ssh $NAME 'PLAN=~/t1_plan.txt bash ~/t1_eval_box.sh'  (done clips skip)" >&2
    exit 3
  fi
  echo "pulled $PULLED of $WANTED rollouts"
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
