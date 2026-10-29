#!/usr/bin/env bash
# Pack the Tier 3 entry from whichever rollout each clip ships, and prove it rather than assert it.
#
# Twenty-five clips ship their pass-1 rollout and five ship the warm-start fine-tune's, chosen by
# the input-only rule in `tools/t3/adopt.py` -- no ground truth is read to decide. Both sets were
# driven by the same reference (our reconstruction's retargeted motion), verified bit-identical
# over every frame, so one `--frame-references` tree covers the whole entry.
#
# Tier 3 is the one tier with `input_path: null`: there is no released input to fit the packer's
# rigid transform against, so the reference is our own npz, which is in released_track2_world by
# construction (frame-0 anchor, 0.0000 cm residual on all 30).
set -euo pipefail
S=${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}
REPO=/home/jetson3/projects/v2d-challenge
PY="$REPO/.venv311/bin/python"
KIT=$S/kit_extract/v2d_submission_kit
DATA=$S/data
REFS=$S/t3/npz_world
COMMIT="${COMMIT:-https://github.com/luaiabuelsamen/v2d-track2/commit/e8ffe8f}"
OUT=${1:-$S/t3_entry_v2}
ADOPTION=$REPO/results/t3_finetune.json

rm -rf "$OUT"; mkdir -p "$OUT/results"
# 1. pass 1 for every clip ...
cp "$S"/t3_eval/results/rollouts/episode_*.parquet "$S"/t3_eval/results/rollouts/episode_*.json "$OUT/results/"
# 2. ... then overwrite exactly the adopted ones, and say which
ADOPTED=$("$PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['adopted']))" "$ADOPTION")
echo "adopted from the fine-tune: ${ADOPTED:-none}"
for TAG in $ADOPTED; do
  EP=$((10#${TAG#e})); F=$(printf 'episode_%06d' "$EP")
  SRC=""
  for d in ws_t3a ws_t3b; do [ -s "$S/$d/results/rollouts/$F.parquet" ] && SRC=$S/$d/results/rollouts; done
  [ -n "$SRC" ] || { echo "FATAL: $TAG adopted but no fine-tuned rollout found" >&2; exit 3; }
  cp "$SRC/$F.parquet" "$SRC/$F.json" "$OUT/results/"
  echo "   $TAG <- $SRC"
done
COUNT=$(ls "$OUT"/results/episode_*.parquet | wc -l)
[ "$COUNT" = 30 ] || { echo "FATAL: $COUNT episodes, expected 30" >&2; exit 3; }

# 3. the organisers' packer, not a reimplementation
"$PY" -I "$KIT/tools/pack_submission.py" --track track_2_tier3 \
  --results "$OUT/results" --dataset-root "$DATA" --frame-references "$REFS" \
  --code-commit "$COMMIT" -o "$OUT/track2_tier3.csv"

# 4. the organisers' preflight (the kit's own gate; Kaggle counts a rejected upload against the
#    quota, so this runs before any submission), then our own scorer
"$PY" -I "$KIT/tools/preflight.py" --track track_2_tier3 \
  --results "$OUT/results" --dataset-root "$DATA" --skip-kaggle 2>&1 | tail -8
"$REPO/.venv311/bin/v2d" score --kit-root "$KIT" --dataset-root "$DATA" \
  --pred "csv:$OUT/track2_tier3.csv" -o "$OUT/track2_tier3.score.json" | tail -8
echo
echo "entry: $OUT/track2_tier3.csv"
