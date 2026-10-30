#!/usr/bin/env bash
# Pack the Tier 1 entry from whichever rollout each clip ships, with the per-clip frame reference
# that clip was actually driven by.
#
# The entry is mixed, and that is the whole reason this is a script rather than one packer call.
# A clip the fine-tune did not improve ships its wave-2 rollout, which was driven by the
# bridge-cleaned Tier 2 reference; a clip the fine-tune did improve ships a rollout driven by the
# released CLEAN Tier 1 trajectory. The kit packer fits one rigid transform per episode from frame
# 0 of the evaluator's recorded reference to frame 0 of whatever `--frame-references` names, so
# pointing the whole entry at one tree would silently mis-convert half of it.
#
# `v2d frame-refs --assemble` builds that mixed tree from a per-clip map and refuses an incomplete
# one. Adoption is decided by tools/t1/adopt.py under the no-regression rule, which may read the
# kit score HERE and only here: Tier 1's own input_note names the released trajectory as the
# configured scoring reference, so on this tier the permitted input is what we are scored against.
set -euo pipefail
S=${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}
REPO=/home/jetson3/projects/v2d-challenge
PY="$REPO/.venv311/bin/python"; V2D="$REPO/.venv311/bin/v2d"
KIT=$S/kit_extract/v2d_submission_kit; DATA=$S/data
SHIPPED_ROLLOUTS=$S/t2_entry_wave2/rollouts
SHIPPED_REFS=$S/t2_entry_wave2/frame_refs
CLEAN_REFS=$S/refs_tier1                      # the released Tier 1 trajectory as npz
COMMIT="${COMMIT:-https://github.com/luaiabuelsamen/v2d-track2/commit/e8ffe8f}"
OUT=${1:-$S/t1_entry_v2}
ADOPTION=$REPO/results/t1_finetune_adoption.json

rm -rf "$OUT"; mkdir -p "$OUT/results"
cp "$SHIPPED_ROLLOUTS"/episode_*.parquet "$SHIPPED_ROLLOUTS"/episode_*.json "$OUT/results/"
ADOPTED=$("$PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['adopted']))" "$ADOPTION")
echo "adopted from the fine-tune: ${ADOPTED:-none}"

# Each adopted clip's rollout comes from the box that produced it.
#
# The generation-2 chain is held OUT of this search and consulted only for clips its own adoption
# record names. Directory order is not evidence: when ws_chain was simply appended to the list it
# became the last match and therefore the winner, and the entry silently shipped gen-2 e00 (0.8078)
# and e12 (0.8013) in place of the gen-1 rollouts that scored 0.8654 and 0.8839 -- a rollout that
# had LOST its own comparison, chosen because its directory sorted later.
CHAIN_ADOPTION=$REPO/results/t1_chain_adoption.json
XFER_ADOPTION=${XFER_ADOPTION:-$REPO/results/t1xfer_adoption.json}
XFER_ADOPTED=""
if [ -s "$XFER_ADOPTION" ]; then
  XFER_ADOPTED=$("$PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['adopted']))" "$XFER_ADOPTION")
  echo "transfer-arm adoptions: ${XFER_ADOPTED:-none}"
fi
CHAIN_ADOPTED=""
if [ -s "$CHAIN_ADOPTION" ]; then
  CHAIN_ADOPTED=$("$PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['adopted']))" "$CHAIN_ADOPTION")
fi
echo "adopted from the generation-2 chain: ${CHAIN_ADOPTED:-none}"
# Iterate over the UNION of every adoption source. The loop used to walk $ADOPTED alone -- the
# fine-tune list -- so a clip adopted ONLY by the transfer arm was never visited and its rollout
# was never substituted: the packer printed "transfer-arm adoptions: e19" and shipped the old
# rollout anyway, leaving the entry score bit-identical. Third time this packer has silently
# shipped the wrong rollout; each time the tell was a score that did not move.
ALL_ADOPTED=$(printf '%s\n' $ADOPTED $XFER_ADOPTED | sort -u | tr '\n' ' ')
for TAG in $ALL_ADOPTED; do
  EP=$((10#${TAG#e})); F=$(printf 'episode_%06d' "$EP"); SRC=""
  for d in ws_tier1 t1f_1 t1f_2c t1f_3c ws_s4a ws_s4b; do
    [ -s "$S/$d/results/rollouts/$F.parquet" ] && SRC=$S/$d/results/rollouts
  done
  case " $CHAIN_ADOPTED " in
    *" $TAG "*) [ -s "$S/ws_chain/results/rollouts/$F.parquet" ] && SRC=$S/ws_chain/results/rollouts ;;
  esac
  # T1XFER (donor-actor transfer) wins over every earlier tree for the clips it adopts, and like
  # the chain it is named explicitly rather than appended to the search list. Appending is what
  # made this packer ship two LOSING rollouts chosen by directory order
  # (results/t1_chain_adoption.json); a named source cannot be re-chosen by filesystem order.
  case " $XFER_ADOPTED " in
    *" $TAG "*) [ -s "$S/xfer_results/rollouts/$F.parquet" ] && SRC=$S/xfer_results/rollouts ;;
  esac
  [ -n "$SRC" ] || { echo "FATAL: $TAG adopted but no Tier 1 rollout found" >&2; exit 3; }
  cp "$SRC/$F.parquet" "$SRC/$F.json" "$OUT/results/"
  echo "   $TAG <- ${SRC#$S/}"
done

# The per-clip reference map: a clip's frame reference must be the one its rollout was DRIVEN by,
# or the packer's rigid-transform check fails. Clean Tier 1 for every clip whose shipped rollout
# came from a Tier 1-driven arm -- the fine-tune AND the donor transfer -- and bridge_v1 for the
# rest. Passing only $ADOPTION here is what made e19 fail at 0.0665 m / 0.4891 rad: its transfer
# rollout was driven by the clean reference while its frame reference was still the Tier 2 one.
"$PY" - "$ADOPTION" "$CLEAN_REFS" "$SHIPPED_REFS" "$OUT/refmap.json" "$XFER_ADOPTED" <<'PY'
import json, sys
adoption, clean, shipped, out = sys.argv[1:5]
extra = set((sys.argv[5] if len(sys.argv) > 5 else "").split())
adopted = set(json.load(open(adoption))["adopted"]) | extra
json.dump({f"e{i:02d}": (clean if f"e{i:02d}" in adopted else shipped) for i in range(30)},
          open(out, "w"), indent=1)
PY
"$V2D" frame-refs --dataset-root "$DATA" --assemble "$OUT/refmap.json" --out "$OUT/frame_refs" \
  -o "$OUT/frame_refs.json"

COUNT=$(ls "$OUT"/results/episode_*.parquet | wc -l)
[ "$COUNT" = 30 ] || { echo "FATAL: $COUNT episodes, expected 30" >&2; exit 3; }

"$PY" -I "$KIT/tools/pack_submission.py" --track track_2_tier1 \
  --results "$OUT/results" --dataset-root "$DATA" --frame-references "$OUT/frame_refs" \
  --code-commit "$COMMIT" -o "$OUT/track2_tier1.csv" 2>&1 | tail -6
"$PY" -I "$KIT/tools/preflight.py" --track track_2_tier1 \
  --results "$OUT/results" --dataset-root "$DATA" --skip-kaggle 2>&1 | tail -6
"$V2D" score --kit-root "$KIT" --dataset-root "$DATA" --tier tier1 \
  --pred "csv:$OUT/track2_tier1.csv" -o "$OUT/track2_tier1.score.json" | tail -8
echo
echo "entry: $OUT/track2_tier1.csv"
