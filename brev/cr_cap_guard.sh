#!/usr/bin/env bash
# A spend cap that deletes boxes, and therefore has to doubt its own arithmetic first.
#
# WHY THIS FILE EXISTS. The first version of this guard was written inline with the box start time
# hardcoded as an epoch, and the epoch was a year out -- 1761612220 is 2025-10-28, not 2026-10-28.
# It computed 26,280 box-hours, read $17,871 against a $20 cap, and deleted three boxes that were
# 14M steps into training and passing every check. It cost about $0.95 and an hour of wall clock.
#
# The bug was not the arithmetic. The bug was that a process allowed to delete running work had no
# opinion about whether its own numbers made sense. So:
#
#   * Box clocks come from `brev ls` observations made by THIS process -- first seen, now -- never
#     from a constant typed by hand.
#   * MAX_PLAUSIBLE_HOURS: any total above it is treated as a bug in the guard, reported, and
#     acted on by exiting rather than by deleting. A cap is a ceiling on spend, not a licence.
#   * A box is only deleted once its measured age exceeds MIN_AGE_HOURS, so a guard that somehow
#     fires in its first minutes cannot take down a box that has barely started.
#   * --dry-run (the default is to act) prints what it would delete and touches nothing.
#
#   brev/cr_cap_guard.sh --prefix v2d-sw-cr- --cap 20 --rate 0.68 [--dry-run] [--once]
set -uo pipefail

PREFIX=v2d-sw-cr-
CAP=20
RATE=0.68
INTERVAL=300
WARN_AT=0.85
DRY=0
ONCE=0
KEEP=0
#: A 3-box arm under a $20 cap cannot legitimately reach 60 box-hours; anything past it is a bug.
MAX_PLAUSIBLE_HOURS=60
#: Never delete a box younger than this, whatever the sum says.
MIN_AGE_HOURS=0.25
LEDGER="${V2D_SCRATCH:-/tmp}/cr_cap_guard_seen.txt"

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --cap) CAP="$2"; shift 2 ;;
    --rate) RATE="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --ledger) LEDGER="$2"; shift 2 ;;
    --max-plausible-hours) MAX_PLAUSIBLE_HOURS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    # Keep the box clocks already on disk instead of starting fresh. For resuming after a restart,
    # and for exercising the plausibility bound against the ledger that caused the incident.
    --keep-ledger) KEEP=1; shift ;;
    --once) ONCE=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Overridable so the guard can be exercised against a fake `brev` with no cloud account: a
# process allowed to delete running work has to be testable, and the incident this file
# exists for was never caught because nothing could run it offline.
BREV="${V2D_BREV:-$HOME/.local/bin/brev}"
[ "$KEEP" = 1 ] || : > "$LEDGER"
WARNED=0

while true; do
  NOW=$(date +%s)
  LIVE=$($BREV ls 2>/dev/null | awk -v p="$PREFIX" 'index($1,p)==1{print $1}')
  # A transient empty listing is not an empty org: only believe it when brev itself answered.
  if [ -z "$LIVE" ] && $BREV ls >/dev/null 2>&1; then
    if [ -s "$LEDGER" ]; then
      echo "every $PREFIX box is gone"
      exit 0
    fi
  fi
  for box in $LIVE; do
    grep -q "^$box " "$LEDGER" 2>/dev/null || printf '%s %s\n' "$box" "$NOW" >> "$LEDGER"
  done

  TOTAL=$(awk -v now="$NOW" '{h+=(now-$2)/3600} END{printf "%.3f", h+0}' "$LEDGER")
  SPEND=$(awk -v h="$TOTAL" -v r="$RATE" 'BEGIN{printf "%.2f", h*r}')

  # The check the first version did not have.
  if awk -v h="$TOTAL" -v m="$MAX_PLAUSIBLE_HOURS" 'BEGIN{exit !(h > m)}'; then
    echo "GUARD BUG: computed $TOTAL box-hours (\$$SPEND), which is past the plausible $MAX_PLAUSIBLE_HOURS h."
    echo "  Not deleting anything. Ledger: $LEDGER"
    awk '{print "  " $0}' "$LEDGER"
    exit 3
  fi

  if awk -v s="$SPEND" -v c="$CAP" 'BEGIN{exit !(s >= c)}'; then
    echo "CAP: $PREFIX has billed ~\$$SPEND of \$$CAP over $TOTAL box-hours"
    for box in $LIVE; do
      AGE=$(awk -v b="$box" -v now="$NOW" '$1==b{printf "%.3f", (now-$2)/3600}' "$LEDGER")
      if awk -v a="${AGE:-0}" -v m="$MIN_AGE_HOURS" 'BEGIN{exit !(a < m)}'; then
        echo "  keeping $box: only ${AGE}h old, which is under the ${MIN_AGE_HOURS}h floor"
        continue
      fi
      if [ "$DRY" = 1 ]; then
        echo "  DRY: would delete $box (${AGE}h)"
      else
        $BREV delete "$box" </dev/null >/dev/null 2>&1 && echo "  deleted $box (${AGE}h)"
      fi
    done
    exit 0
  fi

  if [ "$WARNED" = 0 ] && awk -v s="$SPEND" -v c="$CAP" -v f="$WARN_AT" 'BEGIN{exit !(s >= c*f)}'; then
    WARNED=1
    echo "$PREFIX has billed ~\$$SPEND of \$$CAP over $TOTAL box-hours"
  fi
  [ "$ONCE" = 1 ] && { echo "$PREFIX: ~\$$SPEND of \$$CAP over $TOTAL box-hours"; exit 0; }
  sleep "$INTERVAL"
done
