#!/usr/bin/env bash
# Runs ON the Brev instance. Invoked by brev/gate_launch.sh as:
#   bash -s -- <phase> <steps>
#
# Measures what the 11-day plan depends on: FlashCHORD G1 throughput, peak memory, and
# hours per clip, then (phase=full) trains one clip end to end, evaluates it, and leaves a
# trajectory parquet ready for packing. Writes everything to ~/gate_results/.
#
# It does not submit, post or push anything.
set -uo pipefail

PHASE="${1:-probe}"
STEPS="${2:-10000896}"
# phase=full must train on OUR retargeted clip. Pass its path on the instance as $3; the
# launcher uploads it with --g1-parquet. phase=probe only measures throughput, which does not
# depend on which clip it is, so it falls back to the example sequence shipped in the repo.
G1_PARQUET_ARG="${3:-}"
# B1: after the probe's short training run, evaluate the checkpoint it produced and export a
# trajectory, so the evaluate -> pack -> score path is exercised on real evaluator output. This
# is a *10M-step* floor, not a zero-training floor: evaluate_policy.py takes a checkpoint path
# and has no random-policy mode, so there is no cheaper honest number to get here.
EVAL_EP="${4:-}"
EVAL_WORLDS="${5:-64}"
COMMIT=8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef   # the evaluator commit pinned by KIT_MANIFEST
EASY_EP=24    # watering_can, 748 frames
HARD_EP=5     # big_black_box, 1003 frames
OUT="$HOME/gate_results"
WORK="$HOME/v2d"
HF=https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main

mkdir -p "$OUT"
exec > >(tee -a "$OUT/on_instance.log") 2>&1
# Signal completion on EVERY exit path, and set it here rather than at the end: a trap installed
# near the bottom only fires if execution gets there, which is exactly the case that fails.
trap 'touch "$OUT/DONE"' EXIT
say() { printf '\n== %s (%s)\n' "$*" "$(date -u +%H:%M:%S)"; }
json_escape() { python3 -c 'import json,sys;print(json.dumps(sys.stdin.read().strip()))'; }

say "Host check"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv || {
  echo "FATAL: no GPU visible"; exit 1
}
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "driver=$DRIVER free=${FREE_MIB}MiB gpu=$GPU_NAME"
# FlashCHORD's stated floor. Record rather than abort, so a near miss is still measured.
[ "${FREE_MIB%.*}" -lt 46000 ] && echo "WARNING: under FlashCHORD's 46,000 MiB floor; full training may OOM."
case "$DRIVER" in 5[89]*|6*|7*) ;; *) echo "WARNING: driver $DRIVER is below the required 580." ;; esac

say "Docker check"
docker info >/dev/null 2>&1 || { echo "FATAL: docker is not usable without sudo on this instance"; exit 1; }

say "Clone the repo at the pinned commit"
if [ ! -d "$WORK/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/nvidia-isaac/video_to_data.git "$WORK"
  git -C "$WORK" sparse-checkout init --cone
  git -C "$WORK" sparse-checkout set robotic_grounding
  git -C "$WORK" checkout "$COMMIT"
fi
git -C "$WORK" lfs install >/dev/null 2>&1 || true
git -C "$WORK" lfs pull --include="robotic_grounding/**" || echo "WARNING: git lfs pull incomplete"
git -C "$WORK" rev-parse HEAD

say "Fetch the two gate clips and their assets"
ASSETS="$HOME/v2d_assets"
mkdir -p "$ASSETS"
python3 - "$ASSETS" "$HF" "$EASY_EP" "$HARD_EP" <<'PY'
import json, sys, urllib.request, pathlib
assets, hf, *eps = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
tier = "track_2/tier_1_multiview_caption"
meta_url = f"{hf}/{tier}/meta/episodes_metadata.jsonl"
rows = [json.loads(l) for l in urllib.request.urlopen(meta_url).read().decode().splitlines() if l.strip()]
wanted = {r["episode_index"]: r for r in rows if r["episode_index"] in eps}
paths = [f"{tier}/meta/episodes_metadata.jsonl", f"{tier}/meta/info.json", f"{tier}/meta/episodes.jsonl"]
for ep, row in wanted.items():
    paths += [
        f"{tier}/data/chunk-000/episode_{ep:06d}.parquet",
        f"{tier}/{row['mesh']}",
        f"{tier}/{row['ground_plane']}",
        f"{tier}/{row['support_surface']}",
    ]
for rel in paths:
    out = pathlib.Path(assets) / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size:
        continue
    try:
        urllib.request.urlretrieve(f"{hf}/{rel}", out)
        print("ok  ", rel)
    except Exception as error:
        print("FAIL", rel, error)
print("objects:", {ep: r["object"] for ep, r in wanted.items()})
PY

say "Build the FlashCHORD image (first build is slow; it is cached afterwards)"
cd "$WORK/robotic_grounding/flash_chord"
BUILD_START=$(date +%s)
docker/run.sh build || { echo "FATAL: image build failed"; exit 1; }
BUILD_SECONDS=$(( $(date +%s) - BUILD_START ))
echo "image build took ${BUILD_SECONDS}s"

# The G1 recipe trains against a RETARGETED G1 reference parquet (robot_name=g1/data.parquet),
# not the released SOMA parquet. soma_to_g1.py produces it.
EXAMPLE_G1=/v2d/assets/human_motion_data/whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/data.parquet
if [ -n "$G1_PARQUET_ARG" ]; then
  TRAIN_PARQUET="$G1_PARQUET_ARG"
  echo "training reference: $TRAIN_PARQUET (supplied)"
elif [ "$PHASE" = full ]; then
  echo "FATAL: phase=full needs a retargeted G1 parquet for the gate clip."
  echo "       Produce it with soma_to_g1.py, then re-run with --g1-parquet <path>."
  echo "       Training the repo's example clip instead would measure the wrong thing."
  exit 2
else
  TRAIN_PARQUET="$EXAMPLE_G1"
  echo "training reference: $TRAIN_PARQUET (repo example; throughput probe only)"
fi

# The repo's own wrapper runs `docker exec -it`, and -t fails with "the input device is not a
# TTY" whenever stdin is not a terminal -- which it never is here, because this script arrives
# over `ssh 'bash -s'`. The second probe run reached the GPU, built the image in 229 s, and then
# lost every single container call to this. Talk to docker directly with -i and no -t.
CONTAINER=""
container_id() {
  [ -n "$CONTAINER" ] && { echo "$CONTAINER"; return 0; }
  CONTAINER="$(docker ps -q --filter name=flash_chord | head -1)"
  [ -n "$CONTAINER" ] || return 1
  echo "$CONTAINER"
}
run_in_container() {
  local cid
  cid="$(container_id)" || { echo "no flash_chord container is running" >&2; return 1; }
  docker exec -i "$cid" bash -lc "$1"
}

say "Start the container"
V2D_ASSETS_DIR="$ASSETS" docker/run.sh start latest 0 </dev/null >/dev/null 2>&1 &
for _ in $(seq 1 20); do
  sleep 5
  container_id >/dev/null 2>&1 && break
done
if ! container_id >/dev/null 2>&1; then
  echo "FATAL: no flash_chord container after 100s. docker ps:" >&2
  docker ps -a | head -5 >&2
  exit 1
fi
echo "container $(container_id) is up"
run_in_container 'python -c "import flash_chord, jax; print(\"flash_chord ok\", jax.devices())"' \
  || echo "WARNING: could not import flash_chord/jax in the container"

say "Resolve the composed config (free, no training)"
run_in_container "cd /workspace && python scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac --cfg job --resolve" \
  > "$OUT/composed_config.txt" 2>&1 || echo "WARNING: config resolve failed; see composed_config.txt"
grep -E "total_environment_steps|world_count" "$OUT/composed_config.txt" | head -5 || true

say "Train: phase=$PHASE, ${STEPS} environment steps"
TRAIN_START=$(date +%s)
( while true; do
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> "$OUT/gpu_samples.csv"
    sleep 20
  done ) & SAMPLER=$!

run_in_container "cd /workspace && python scripts/train_flash_sac.py \
  experiment=g1_recon_body_flash_sac \
  \"task.parquet='${TRAIN_PARQUET}'\" \
  training.total_environment_steps=${STEPS} \
  training.checkpoint.save_interval_environment_steps=${STEPS} \
  reset.seed=42 logging.mode=disabled \
  output_dir=outputs/gate_${PHASE}" 2>&1 | tail -80
TRAIN_RC=$?
kill $SAMPLER 2>/dev/null
TRAIN_SECONDS=$(( $(date +%s) - TRAIN_START ))
PEAK_MIB=$(awk -F, 'BEGIN{m=0}{gsub(/ /,"",$1); if ($1+0>m) m=$1+0}END{print m}' "$OUT/gpu_samples.csv" 2>/dev/null || echo 0)
echo "train rc=$TRAIN_RC seconds=$TRAIN_SECONDS peak=${PEAK_MIB}MiB"

CHECKPOINT="$(run_in_container "ls -1 /workspace/outputs/gate_${PHASE}/policy_*.safetensors 2>/dev/null | tail -1" | tr -d '\r')"
echo "checkpoint=$CHECKPOINT"

if [ "$PHASE" = full ] && [ -n "$CHECKPOINT" ]; then
  say "Evaluate the checkpoint (world_count=4096)"
  run_in_container "cd /workspace && python scripts/evaluate_policy.py \
    \"evaluation.checkpoint=${CHECKPOINT}\" evaluation.world_count=4096 \
    \"evaluation.metrics_output=${CHECKPOINT}.metrics.json\"" 2>&1 | tail -40
  run_in_container "cat ${CHECKPOINT}.metrics.json" > "$OUT/eval_metrics.json" 2>/dev/null || true
  run_in_container "ls -1 /workspace/outputs/gate_full" > "$OUT/full_outputs.txt" 2>/dev/null || true
  docker cp "$(docker ps -q --filter name=flash_chord | head -1)":/workspace/outputs/gate_full "$OUT/" 2>/dev/null || true
elif [ -n "$EVAL_EP" ] && [ -n "$CHECKPOINT" ]; then
  say "B1 floor: evaluate the probe checkpoint on e${EVAL_EP} (world_count=${EVAL_WORLDS})"
  # WANDB_MODE=disabled on purpose: evaluate_policy.py calls wandb.init itself, and this
  # instance has no W&B credential and is not getting one -- a rented VM is not a place to put a
  # token. The run is logged from the host instead, from the results copied back.
  run_in_container "cd /workspace && WANDB_MODE=disabled python scripts/evaluate_policy.py \
    \"evaluation.checkpoint=${CHECKPOINT}\" evaluation.world_count=${EVAL_WORLDS} \
    \"evaluation.metrics_output=${CHECKPOINT}.b1.metrics.json\"" 2>&1 | tail -40
  run_in_container "cat ${CHECKPOINT}.b1.metrics.json" > "$OUT/b1_eval_metrics.json" 2>/dev/null || true
  CID="$(docker ps -q --filter name=flash_chord | head -1)"
  run_in_container "ls -1R /workspace/outputs/gate_${PHASE}" > "$OUT/b1_outputs.txt" 2>/dev/null || true
  docker cp "$CID":/workspace/outputs/gate_${PHASE} "$OUT/b1_outputs" 2>/dev/null || true
fi

# ---------------------------------------------------------------- the numbers that matter
say "Extrapolation"
python3 - "$OUT" "$PHASE" "$STEPS" "$TRAIN_SECONDS" "$PEAK_MIB" "$GPU_NAME" "$DRIVER" "$FREE_MIB" "$BUILD_SECONDS" "$TRAIN_RC" <<'PY'
import json, sys
out, phase, steps, seconds, peak, gpu, driver, free, build, rc = sys.argv[1:]
steps, seconds, peak, build, rc = int(steps), int(seconds), int(peak), int(build), int(rc)
full_steps = 349999104
sps = steps / seconds if seconds else 0
hours_per_clip = (full_steps / sps / 3600) if sps else None
result = {
    "phase": phase, "train_return_code": rc, "gpu": gpu, "driver": driver,
    "free_mib_at_start": free, "image_build_seconds": build,
    "steps_run": steps, "train_seconds": seconds,
    "env_steps_per_second": round(sps, 1),
    "peak_gpu_mib": peak,
    "projected_hours_per_clip_350M": round(hours_per_clip, 2) if hours_per_clip else None,
}
if hours_per_clip:
    for label, rate in (("a6000_0.60", 0.60), ("l40s_1.06", 1.06)):
        result[f"usd_per_clip_{label}"] = round(hours_per_clip * rate, 2)
        result[f"usd_30_clips_{label}"] = round(hours_per_clip * rate * 30, 2)
        result[f"gpu_hours_30_clips"] = round(hours_per_clip * 30, 1)
    result["gate_cost_pass"] = bool(hours_per_clip * 30 <= 120)
print(json.dumps(result, indent=2))
json.dump(result, open(f"{out}/gate_{phase}.json", "w"), indent=2)
PY

say "Stopping the container"
docker/run.sh stop latest 0 >/dev/null 2>&1 || true
echo "Results in $OUT:"; ls -la "$OUT"
