# Hand out box names so that no two boxes can ever take the same one, and so the total number of
# boxes a run may create is bounded by something other than hope.
#
# Needed because boxes are now replaced when they fail: at 10:05 on 2026-10-26 three of eight
# creates failed provider-side (one `internal server error`, two that never reached READY), the
# launcher started no replacements, and coverage fell from 32 clip-slots to 28 for 30 clips. A
# replacement needs a name no other slot will pick, chosen without a lock file to clean up.
#
# `mkdir` is the lock, exactly as the clip queue claims clips: it is atomic on a local filesystem
# and it leaves the record of who took what behind for the report.

# box_name_claim <names-dir> <max-index> -- print the lowest free index, or fail if there is none.
box_name_claim() {
  local dir="${1:?names dir}" max="${2:?max index}" i
  mkdir -p "$dir" || return 2
  for i in $(seq 1 "$max"); do
    if mkdir "$dir/$(printf '%02d' "$i")" 2>/dev/null; then
      printf '%d\n' "$i"
      return 0
    fi
  done
  return 1
}

# box_name_reserve <names-dir> <index> -- take one specific index, e.g. for --add-box. Succeeds if
# it was free, fails if somebody already holds it.
box_name_reserve() {
  local dir="${1:?names dir}" index="${2:?index}"
  mkdir -p "$dir" || return 2
  mkdir "$dir/$(printf '%02d' "$index")" 2>/dev/null
}

# box_names_taken <names-dir> -- how many have been handed out, for the spend report.
box_names_taken() {
  local dir="${1:?names dir}"
  [ -d "$dir" ] && find "$dir" -mindepth 1 -maxdepth 1 -type d | wc -l || echo 0
}
