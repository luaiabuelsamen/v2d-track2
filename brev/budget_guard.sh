#!/usr/bin/env bash
# The hard cap, in one place. Sourced by every launcher before it creates a box.
#
# Luai authorized $230 of NEW spend on 2026-10-26, on top of the $21.13 already spent under the
# earlier $30 standing budget. NEW_BUDGET_USD is that $230: the ledger below counts only boxes
# created from the AB4 work onward, so prior spend is not double-counted.
#
# A box costs money from the moment it is created, so the guard is asked BEFORE create, with the
# hours that box is expected to bill. Refusing is cheap; an over-cap box is not.
set -uo pipefail

NEW_BUDGET_USD="${NEW_BUDGET_USD:-230}"
RATE_USD_PER_HOUR="${RATE_USD_PER_HOUR:-0.68}"
# The ledger has ONE canonical path. It used to default to $HOME/spend_ab4.txt and only move to
# the scratch dir when V2D_SCRATCH happened to be exported, so an invocation without that variable
# read a file that does not exist, counted zero closed minutes, and reported $94.08 against a true
# $153.90 -- a guard that under-reads by $60 is not a cap. The launchers export V2D_SCRATCH before
# sourcing this, and anything else gets the same default they use.
V2D_LEDGER_DEFAULT="${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}/spend_ab4.txt"
LEDGER="${LEDGER:-$V2D_LEDGER_DEFAULT}"
# A box whose uptime cannot be read is billing anyway. Counting it as zero is the one error a cap
# must not make, so it counts as a full expected cell instead and says which box it guessed for.
UNREADABLE_BOX_MINUTES="${UNREADABLE_BOX_MINUTES:-300}"

# Minutes already committed by AB4 boxes: closed ones from the ledger, live ones from their uptime.
ab4_committed_minutes() {
  local total=0 mins
  if [ -f "$LEDGER" ]; then
    while read -r _name mins _rest; do
      case "${_name:-}" in ''|'#'*) continue ;; esac
      [[ "$mins" =~ ^[0-9]+$ ]] && total=$((total + mins))
    done < "$LEDGER"
  fi
  # `brev ls` sometimes answers short -- an empty or partial listing is a documented failure mode
  # on this org, and it is how a $157.60 commitment read as $134.52. A partial listing is always a
  # SUBSET of the truth, so three reads are unioned and the largest set wins.
  local names
  names="$(
    for _ in 1 2 3; do
      ~/.local/bin/brev ls 2>/dev/null | awk 'NR>2 && $1 ~ /^v2d-/ {print $1}'
      sleep 1
    done | sort -u
  )"
  for n in $names; do
    mins="$(ssh -n -o ConnectTimeout=15 -o BatchMode=yes "$n" \
             "awk '{print int(\$1/60)}' /proc/uptime" 2>/dev/null | grep -v Pseudo | tail -1)"
    if [[ "$mins" =~ ^[0-9]+$ ]]; then
      total=$((total + mins))
    else
      echo "   budget: $n did not answer; counting it as $UNREADABLE_BOX_MINUTES min" >&2
      total=$((total + UNREADABLE_BOX_MINUTES))
    fi
  done
  echo "$total"
}

ab4_spent_usd() {
  awk -v m="$(ab4_committed_minutes)" -v r="$RATE_USD_PER_HOUR" 'BEGIN{printf "%.2f", m/60*r}'
}

# budget_ok_for_hours <hours> [label] -- 0 to proceed, 1 to refuse, and says which on stderr.
budget_ok_for_hours() {
  local hours="${1:?hours this box is expected to bill}" label="${2:-a box}"
  local spent projected
  spent="$(ab4_spent_usd)"
  projected="$(awk -v h="$hours" -v r="$RATE_USD_PER_HOUR" 'BEGIN{printf "%.2f", h*r}')"
  if awk -v s="$spent" -v p="$projected" -v b="$NEW_BUDGET_USD" 'BEGIN{exit !(s+p > b)}'; then
    printf 'BUDGET REFUSAL: %s would bill ~$%s on top of $%s already committed, over the $%s cap.\n' \
      "$label" "$projected" "$spent" "$NEW_BUDGET_USD" >&2
    return 1
  fi
  printf 'budget ok: %s ~$%s, committed $%s of $%s\n' "$label" "$projected" "$spent" "$NEW_BUDGET_USD" >&2
  return 0
}

# Record a box's billed minutes the moment it is deleted, on the success AND failure paths.
ab4_ledger_add() {
  printf '%s %s\n' "${1:?name}" "${2:?minutes}" >> "$LEDGER"
}
