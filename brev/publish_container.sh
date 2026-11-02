#!/usr/bin/env bash
# Publish the container the submission form asks for: the image our policies were trained in.
#
# The form wants a *link*, so a Dockerfile in the repo is not enough -- the image has to be
# somewhere a judge can pull it from. It goes to GitHub Container Registry under the account that
# already owns the code, which means no second account and one credential.
#
# What ships is the flash_chord image built from NVIDIA's pinned commit, with OUR patch scripts
# in /opt/v2d/brev and the Tier 2 shipping recipe already applied, committed from the running
# container rather than rebuilt -- so it is literally the thing that trained, not a reconstruction
# of it. /opt/v2d/RECIPE.txt records which flags each entry used, so a judge can reproduce any arm
# by applying the other patches from inside the image.
#
#   brev/publish_container.sh --box v2d-container [--tag ghcr.io/<owner>/<name>:<version>] --go
#
# Without --go it prints what it would do and touches nothing.
set -uo pipefail

BOX=""
TAG="ghcr.io/luaiabuelsamen/v2d-track2:submission"
GO=0
# The Tier 2 entry's code patches. Finger targets and clean references are DATA, staged per clip,
# so they are not in the image and are named in RECIPE.txt instead of silently missing from it.
SHIP_FLAGS="start_curriculum,orientation_reward"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --box) BOX="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --flags) SHIP_FLAGS="$2"; shift 2 ;;
    --go) GO=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$BOX" ] || { echo "--box is required" >&2; exit 2; }

# ghcr rejects any uppercase in a repository name, and the error it returns says only
# "invalid reference format", which names neither the case nor the field.
case "$TAG" in *[A-Z]*) echo "FATAL: ghcr tags must be lowercase: $TAG" >&2; exit 2 ;; esac

TOKEN="$(python3 - <<'PY'
import pathlib, re
hosts = pathlib.Path.home() / ".config/gh/hosts.yml"
m = re.search(r"oauth_token:\s*(\S+)", hosts.read_text()) if hosts.exists() else None
print(m.group(1) if m else "")
PY
)"
[ -n "$TOKEN" ] || { echo "FATAL: no GitHub token in ~/.config/gh/hosts.yml" >&2; exit 1; }

# A push with a token that lacks write:packages fails only after the layers have uploaded, which
# on a multi-gigabyte image is twenty minutes of billing to learn a one-word answer. Ask first.
SCOPES="$(curl -sS -D - -o /dev/null --http1.1 -H "Authorization: token $TOKEN" \
  https://api.github.com/user | tr -d '\r' | sed -n 's/^[Xx]-[Oo][Aa]uth-[Ss]copes: //p')"
echo "token scopes: ${SCOPES:-<none reported>}"
case "$SCOPES" in
  *write:packages*) ;;
  *) echo "FATAL: the token cannot write packages. Run:" >&2
     echo "         gh auth refresh -h github.com -s write:packages" >&2
     exit 1 ;;
esac

COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "box    $BOX"
echo "tag    $TAG"
echo "flags  $SHIP_FLAGS"
echo "commit $COMMIT"
[ "$GO" = 1 ] || { echo "DRY RUN. Nothing pushed. Re-run with --go."; exit 0; }

say() { printf '\n== %s (%s)\n' "$*" "$(date -u +%H:%M:%S)"; }

say "Shipping our patch scripts to the box"
ssh -o StrictHostKeyChecking=accept-new "$BOX" "mkdir -p ~/v2d_brev" || exit 1
scp -q "$REPO_ROOT"/brev/patch_*.py "$REPO_ROOT"/brev/arm_flags.sh "$BOX":~/v2d_brev/ || exit 1

say "Baking the image on the box"
ssh "$BOX" "bash -s -- '$SHIP_FLAGS' '$TAG' '$COMMIT' '$TOKEN'" <<'REMOTE'
set -uo pipefail
FLAGS="$1"; TAG="$2"; COMMIT="$3"; TOKEN="$4"
CID="$(docker ps -q --filter name=flash_chord | head -1)"
[ -n "$CID" ] || { echo "FATAL: no flash_chord container running on this box" >&2; exit 1; }

# flash_chord is installed from /workspace, and /workspace is a BIND MOUNT. So the patches we
# apply through `docker exec` land on the host tree, not on any image layer, and a `docker commit`
# would publish a stock image while reporting success. The patched tree has to be COPYd into a new
# layer instead -- which also makes the published image self-contained, since a judge running it
# without our mount still gets the code that trained.
HOSTWS="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}' "$CID")"
[ -n "$HOSTWS" ] && [ -d "$HOSTWS" ] || { echo "FATAL: cannot find the host side of /workspace" >&2; exit 1; }
echo "host workspace: $HOSTWS"

docker exec -u 0 -i "$CID" mkdir -p /opt/v2d/brev < /dev/null || exit 1
mkdir -p "$HOSTWS/.v2d/brev"
cp ~/v2d_brev/* "$HOSTWS/.v2d/brev/" || exit 1

# Apply the shipping patches. Every patch script is fatal on an upstream it does not recognise,
# so a refusal here means the tree is not the one we trained in and must not be published.
IFS=',' read -ra FLAG_LIST <<< "$FLAGS"
for flag in "${FLAG_LIST[@]}"; do
  case "$flag" in
    start_curriculum)    script=patch_start_curriculum.py ;;
    orientation_reward)  script=patch_object_orientation.py ;;
    object_position)     script=patch_object_position.py ;;
    body_joint_position) script=patch_body_joint_position.py ;;
    contact_role)        script=patch_contact_role.py ;;
    voc_curriculum)      script=patch_voc_curriculum.py ;;
    warm_start)          script=patch_warm_start.py ;;
    *) echo "FATAL: unknown flag $flag" >&2; exit 1 ;;
  esac
  # The patch scripts take the flash_chord root, and the parameterised ones take their knob.
  # Both are passed explicitly, exactly as arm_flags_apply does, so the image records the arm
  # rather than a default that could change under it.
  extra=""
  [ "$flag" = orientation_reward ] && extra="--std 0.3"
  echo "-- applying $flag ($script $extra)"
  docker exec -i "$CID" python /workspace/.v2d/brev/"$script" /workspace/src/flash_chord $extra \
    < /dev/null || {
      echo "FATAL: $script refused to apply; the tree is not what we trained in" >&2; exit 1; }
done

cat > "$HOSTWS/.v2d/RECIPE.txt" <<RECIPE
V2D Challenge Track 2 -- Luai Abuelsamen, UC Berkeley
code: https://github.com/luaiabuelsamen/v2d-track2 at $COMMIT

NVIDIA's flash_chord image, built from the commit pinned by KIT_MANIFEST, with the flash_chord
source tree baked in at /workspace and these patches already applied:  $FLAGS

Per-entry recipes (the rest are DATA, staged per clip, and are not in the image):
  Tier 2  start_curriculum + orientation_reward(std 0.3) + finger targets + bridge-cleaned refs
  Tier 1  the same, plus warm_start fine-tuning on the clean multiview trajectory
  Tier 3  the same, on our own reconstruction as the reference

To reproduce another arm, apply its patch inside a running container, e.g.
  python /workspace/.v2d/brev/patch_object_position.py
Each script is atomic, idempotent, and fatal on an upstream it does not recognise.
RECIPE

cat > "$HOSTWS/Dockerfile.v2d" <<'DOCKER'
# syntax=docker/dockerfile:1
FROM flash_chord:latest
USER root
COPY . /workspace
COPY .v2d/brev /opt/v2d/brev
COPY .v2d/RECIPE.txt /opt/v2d/RECIPE.txt
USER 1001
DOCKER

echo "-- building $TAG"
# --provenance=false: buildx otherwise attaches an attestation manifest and pushes an index,
# which ghcr rejects with a bare "denied" AFTER every layer has uploaded -- twenty minutes to
# learn nothing. A single-platform manifest is also what a judge wants to pull.
docker build --provenance=false --sbom=false -f "$HOSTWS/Dockerfile.v2d" -t "$TAG" "$HOSTWS" || exit 1

# Prove the patch is in the IMAGE, not only on the host tree: a fresh container, no mounts.
echo "-- verifying the patch survived into the image"
docker run --rm --entrypoint bash "$TAG" -lc \
  'grep -c object_orientation /workspace/src/flash_chord/objectives/recon_body.py' || exit 1

echo "$TOKEN" | docker login ghcr.io -u luaiabuelsamen --password-stdin || exit 1
docker push "$TAG" || exit 1
docker logout ghcr.io >/dev/null 2>&1
echo "PUSHED $TAG"
REMOTE
