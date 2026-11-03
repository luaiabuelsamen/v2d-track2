#!/usr/bin/env bash
# Stage arm CR: the bridge_v1 references every second-wave arm trained on, plus the contact roles.
#
# Arm CR needs one thing no other arm needs: a per-clip data file the OBJECTIVE reads at training
# time. The container only ever sees ~/v2d_assets, mounted at /v2d/assets, so the roles have to
# travel inside the staged reference tree -- and they have to be in the tree BEFORE its md5
# manifest is written, because the launcher refuses a box whose files do not match the manifest.
# Writing a file after the manifest is exactly what killed the first 30-clip launch: the manifest
# listed 181 files, the tree had 182, and eight boxes were deleted for "references never matched".
#
# So this is stage_t2_combined.sh --from-briefs (the same bridge_v1 + fingers tree as arm A, which
# is what keeps the reward the only variable) followed by the roles, followed by one manifest over
# the lot, followed by a count check.
#
#   brev/stage_cr.sh                       # map from $ROOT/map_CR.txt, roles from results/
#   brev/stage_cr.sh --map <f> --mount cr_refs --roles results/contact_roles
set -uo pipefail

S="${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}"
REPO=/home/jetson3/projects/v2d-challenge
ROOT="$S/second_wave"
MAP="$ROOT/map_CR.txt"
MOUNT=cr_refs
ROLES="$REPO/results/contact_roles"
SOURCES=()

while [ $# -gt 0 ]; do
  case "$1" in
    --map) MAP="$2"; shift 2 ;;
    --mount) MOUNT="$2"; shift 2 ;;
    --roles) ROLES="$2"; shift 2 ;;
    --root) ROOT="$2"; MAP="$ROOT/map_CR.txt"; shift 2 ;;
    # Name the reference tree directly instead of reading the briefs. The briefs' clean_refs_path
    # names a DATASET root, which stage_t2_combined has to retarget first (~70 min); interp's
    # finger tree is already the retarget of that same bridge_v1 plus its finger solve, which is
    # exactly what every second-wave arm trained on, so passing it keeps the reward the only
    # variable without paying for the retarget again.
    --source) SOURCES+=("$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

OUT="$S/$MOUNT"
[ -f "$MAP" ] || { echo "FATAL: no clip map at $MAP -- run brev/second_wave.sh --groups CR first" >&2; exit 2; }
[ -d "$ROLES" ] || { echo "FATAL: no contact roles at $ROLES -- run 'v2d contact-roles' first" >&2; exit 2; }

echo "== staging the bridge_v1 references for $(grep -c '[^[:space:]]' "$MAP") clips into $OUT"
if [ "${#SOURCES[@]}" -gt 0 ]; then
  ARGS=()
  for src in "${SOURCES[@]}"; do ARGS+=(--source "$src"); done
  "$REPO/brev/stage_t2_combined.sh" "${ARGS[@]}" --map "$MAP" --mount "$MOUNT" || {
    echo "FATAL: staging the references failed" >&2; exit 3; }
else
  "$REPO/brev/stage_t2_combined.sh" --from-briefs --map "$MAP" --mount "$MOUNT" || {
    echo "FATAL: staging the references failed" >&2; exit 3; }
fi

# The roles, one file per clip in the map, and a refusal if any clip is missing one: a clip whose
# roles file is absent would train with the stub derived from the raw contact flag, under this
# arm's name, which is the one failure the whole arm-flag style exists to prevent.
echo "== adding the contact roles from $ROLES"
mkdir -p "$OUT/contact_roles"
missing=""
while read -r ep _; do
  [ -n "$ep" ] || continue
  TAG="e$(printf '%02d' "$ep")"
  if [ -f "$ROLES/$TAG.npz" ]; then
    cp "$ROLES/$TAG.npz" "$OUT/contact_roles/$TAG.npz"
  else
    missing="$missing $TAG"
  fi
done < "$MAP"
if [ -n "$missing" ]; then
  echo "FATAL: no roles file for:$missing" >&2
  echo "       run: v2d contact-roles --clips $(echo "$missing" | tr ' ' ',' | sed 's/^,//')" >&2
  exit 3
fi
echo "   $(find "$OUT/contact_roles" -name '*.npz' | wc -l) roles files"

# One manifest over everything, generated LAST, exactly as stage_t2_combined.sh generates its own.
( cd "$OUT" && find . -type f | sort | xargs md5sum ) > "$OUT.md5"
FILES=$(find "$OUT" -type f | wc -l)
LINES=$(grep -c '[^[:space:]]' "$OUT.md5")
if [ "$FILES" != "$LINES" ]; then
  echo "FATAL: manifest has $LINES lines for $FILES files; do not launch from this tree" >&2
  exit 3
fi
echo "== $OUT: $FILES files, manifest $LINES lines, $(du -sh "$OUT" | cut -f1)"
echo "   roles in the container: /v2d/assets/$MOUNT/contact_roles"
