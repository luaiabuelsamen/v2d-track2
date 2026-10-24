#!/usr/bin/env bash
# Build the CPU retargeting environment on the Jetson (Orin, aarch64, JetPack R36.4.3) with
# no root and no system packages. Idempotent: safe to re-run.
#
# Why each step is the way it is — every one of these was a dead end first:
#
#  * Python 3.11, not the system 3.10. `py-soma-x` imports `pxr` (OpenUSD), and `usd-core`
#    publishes aarch64 wheels only for cp311 and newer. uv installs a
#    python-build-standalone 3.11 into ~/.local without touching the system.
#  * `py-soma-x` is installed with --no-deps. It declares `cholespy`, which has no aarch64
#    wheel and fails its CMake build, but nothing on the retargeting path imports it.
#  * torch comes from the CPU index. The default aarch64 wheel drags in ~3.6 GB of CUDA 13
#    libraries that this box cannot use anyway: the Jetson driver reports CUDA 12.6, so Warp
#    falls back to the CPU regardless.
#  * pink needs a QP solver from qpsolvers; none ships by default.
#  * G1 meshes come from GitHub's /raw/<sha>/ endpoint, which resolves Git LFS content.
#    `git lfs pull` on this sparse checkout left every STL as a pointer.
#  * **Always run these tools with `python -I`.** The machine's global PYTHONPATH includes a
#    PyTorch *source tree* (~/projects/clean_env/pytorch), which shadows the venv's torch and
#    fails with "Failed to load PyTorch C extensions".
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT/.venv311"
SCRATCH="${1:-}"
COMMIT=8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef
export PATH="$HOME/.local/bin:$PATH"

if [ -z "$SCRATCH" ]; then
  echo "usage: $0 <scratch-dir>   # holds the sparse repo checkout and staged assets" >&2
  exit 2
fi
mkdir -p "$SCRATCH"
REPO="$SCRATCH/repo_src"
say() { printf '\n== %s\n' "$*"; }

say "uv (user space)"
command -v uv >/dev/null || python3 -m pip install --user --no-input uv
uv --version

say "CPython 3.11 (python-build-standalone, no root)"
uv python install 3.11
[ -d "$VENV" ] || uv venv -p 3.11 "$VENV"
"$VENV/bin/python" -V

say "Core packages"
uv pip install --python "$VENV/bin/python" \
  usd-core pin "pin-pink==4.2.0" trimesh numpy scipy pyarrow tqdm viser \
  warp-lang rtree huggingface_hub osqp quadprog daqp

say "CPU-only torch"
if ! "$VENV/bin/python" -I -c "import torch,sys;sys.exit(0 if '+cpu' in torch.__version__ else 1)" 2>/dev/null; then
  NV="$(uv pip list --python "$VENV/bin/python" 2>/dev/null | awk '/^nvidia-/{print $1}' | tr '\n' ' ')"
  # shellcheck disable=SC2086
  [ -n "$NV" ] && uv pip uninstall --python "$VENV/bin/python" torch $NV || true
  uv pip install --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cpu torch
fi

say "SOMA-X (no deps: cholespy does not build on aarch64 and is not needed)"
uv pip install --python "$VENV/bin/python" --no-deps "py-soma-x==0.2.1"

say "Sparse checkout of the retargeting code at $COMMIT"
if [ ! -d "$REPO/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone -q --filter=blob:none --no-checkout \
    https://github.com/nvidia-isaac/video_to_data.git "$REPO"
  git -C "$REPO" sparse-checkout init --cone
  git -C "$REPO" sparse-checkout set robotic_grounding/scripts/retarget robotic_grounding/source/robotic_grounding
  GIT_LFS_SKIP_SMUDGE=1 git -C "$REPO" checkout -q "$COMMIT"
fi
git -C "$REPO" rev-parse HEAD

say "Put robotic_grounding on the venv path"
echo "$REPO/robotic_grounding/source/robotic_grounding" \
  > "$VENV/lib/python3.11/site-packages/_robotic_grounding.pth"

say "G1 meshes and URDFs (raw endpoint resolves LFS; git lfs pull does not here)"
ASSETS=robotic_grounding/source/robotic_grounding/robotic_grounding/assets
python3 -c "
import json, urllib.request, pathlib, sys
url='https://api.github.com/repos/nvidia-isaac/video_to_data/git/trees/$COMMIT?recursive=1'
tree=json.load(urllib.request.urlopen(url))['tree']
pre='$ASSETS/'
want=[e['path'] for e in tree if e['type']=='blob' and e['path'].startswith(pre)
      and ('/meshes/g1/' in e['path'] or '/urdfs/g1/' in e['path'])]
pathlib.Path('$SCRATCH/g1_mesh_list.txt').write_text('\n'.join(want) + '\n')
print(len(want), 'mesh/urdf files')
"
fetched=0
while read -r p; do
  [ -z "$p" ] && continue
  out="$REPO/$p"
  mkdir -p "$(dirname "$out")"
  if [ -s "$out" ] && ! head -c 60 "$out" | grep -q "git-lfs.github.com"; then continue; fi
  curl -sSL --retry 3 -o "$out" "https://github.com/nvidia-isaac/video_to_data/raw/$COMMIT/$p"
  fetched=$((fetched+1))
done < "$SCRATCH/g1_mesh_list.txt"
echo "fetched $fetched file(s)"

say "Stage the pinned SOMA-X body model"
"$VENV/bin/python" -I "$REPO/robotic_grounding/scripts/setup_soma_assets.py" | tail -3

say "Verify"
"$VENV/bin/python" -I - <<'PY'
import numpy as np
mods = {}
for name in ("pxr", "soma", "pinocchio", "pink", "torch", "trimesh", "viser", "pyarrow", "scipy"):
    mods[name] = __import__(name)
print("torch", mods["torch"].__version__)
import qpsolvers
print("qp solvers", qpsolvers.available_solvers)
from robotic_grounding.retarget import G1_URDF_DIR, BODY_MODELS_DIR
print("G1 urdf dir exists:", (G1_URDF_DIR / "main.urdf").is_file())
neutral = BODY_MODELS_DIR / "soma" / "SOMA_neutral.npz"
with np.load(neutral, allow_pickle=True) as archive:
    print("SOMA rig joints:", len(archive["joint_names"]))
print("OK")
PY

cat <<EOF

Ready. Retarget everything with:

  $VENV/bin/python -I scripts/retarget_all.py \\
    --dataset-root <challenge data> --repo $REPO \\
    --soma-root $REPO/$ASSETS/body_models/soma \\
    --seq-root $SCRATCH/seq --out-root $SCRATCH/motion \\
    --log-dir $SCRATCH/logs --validation-dir $SCRATCH/validation \\
    --tiers tier1 tier2 --workers 3 -o results/retarget_all.json
EOF
