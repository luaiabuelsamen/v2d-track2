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
LEDGER="${LEDGER:-$(dirname "${BASH_SOURCE[0]}")/../../../spend_ab4.txt}"
[ -n "${V2D_SCRATCH:-}" ] && LEDGER="$V2D_SCRATCH/spend_ab4.txt"

# Minutes already committed by AB4 boxes: closed ones from the ledger, live ones from their uptime.
ab4_committed_minutes() {
  local total=0 mins
  if [ -f "$LEDGER" ]; then
    while read -r _name mins _rest; do
      case "${_name:-}" in ''|'#'*) continue ;; esac
      [[ "$mins" =~ ^[0-9]+$ ]] && total=$((total + mins))
    done < "$LEDGER"
  fi
  for n in $(~/.local/bin/brev ls 2>/dev/null | awk 'NR>2 && $1 ~ /^v2d-/ {print $1}'); do
    mins="$(ssh -n -o ConnectTimeout=15 -o BatchMode=yes "$n" \
             "awk '{print int(\$1/60)}' /proc/uptime" 2>/dev/null | grep -v Pseudo | tail -1)"
    [[ "$mins" =~ ^[0-9]+$ ]] && total=$((total + mins))
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
