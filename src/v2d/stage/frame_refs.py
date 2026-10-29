"""Assemble a per-episode `--frame-references` tree for a mixed-arm entry.

The kit packer converts each rollout into the released world frame by fitting one rigid transform
from frame 0 of the evaluator's recorded reference to frame 0 of a *known* trajectory. Which
trajectory depends on what drove that rollout, and in a mixed entry that differs per clip:

* a wave-2 rollout was driven by the bridge-cleaned Tier 2 reference -> its own npz
* a Tier 3 rollout was driven by our video reconstruction          -> its own npz
* a clean-Tier-1 rollout was driven by our retarget of the released Tier 1 trajectory, so the
  reference *is* the released input, and there is no npz lying around for it

This builds the missing third case straight from the dataset, and then stitches a complete
directory from a per-episode source map. `v2d.submit.kit.npz_poses` requires an npz for every
scored episode, so "leave it out and let the packer fall back" is not an option for our own audit
even though the kit packer itself tolerates it.

    v2d frame-refs --dataset-root DIR --tier tier1 --out DIR            # released trajectory
    v2d frame-refs --assemble a.json --out DIR                          # mixed, from a map

The map is `{"e00": "/path/to/dir", ...}`: for each clip, the directory holding the npz that drove
its chosen rollout. Every clip in the dataset must appear, which is deliberate -- a silently
missing clip would be packed against the wrong reference and the frame check would only catch it if
the two references happened to differ at frame 0.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ..dataset import episodes, hold_last_visible, object_pose

FRAME = "released_track2_world"


def write_npz(out_dir: Path, episode: int, pose_xyzw: np.ndarray) -> Path:
    """One episode, in the shape all three consumers agree on: (frames, 1, 7) plus the indices."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_{episode:06d}.npz"
    np.savez_compressed(
        path,
        pose_xyzw=pose_xyzw.reshape(len(pose_xyzw), 1, 7),
        coordinate_frame=FRAME,
        episode_index=np.int64(episode),
        frame_index=np.arange(len(pose_xyzw), dtype=np.int64),
    )
    return path


def released(dataset_root: Path, tier: str, out_dir: Path) -> dict:
    """Emit the released trajectory of `tier` as frame references, one npz per episode."""
    written = []
    for row in episodes(dataset_root):
        pose, visible = object_pose(dataset_root, tier, row.index)
        # An invisible frame stores a zero-filled quaternion, which every downstream reader
        # rejects; carrying the last visible pose across it is what the scorer itself does.
        pose, filled = hold_last_visible(pose, visible)
        write_npz(out_dir, row.index, pose)
        written.append({"episode": row.index, "frames": len(pose), "invisible_filled": filled})
    return {
        "what": f"the released {tier} object trajectory as kit frame references",
        "tier": tier,
        "episodes": len(written),
        "per_episode": written,
    }


def assemble(mapping: dict[str, str], dataset_root: Path, out_dir: Path) -> dict:
    """Copy each clip's npz out of the directory named for it, refusing an incomplete map."""
    wanted = {f"e{row.index:02d}": row.index for row in episodes(dataset_root)}
    missing = sorted(set(wanted) - set(mapping))
    if missing:
        raise SystemExit(f"FATAL: the map leaves out {missing}; every scored clip needs a source")
    unknown = sorted(set(mapping) - set(wanted))
    if unknown:
        raise SystemExit(f"FATAL: the map names clips that are not scored: {unknown}")

    out_dir.mkdir(parents=True, exist_ok=True)
    used: dict[str, int] = {}
    for tag, episode in sorted(wanted.items()):
        source = Path(mapping[tag]) / f"episode_{episode:06d}.npz"
        if not source.is_file():
            raise SystemExit(f"FATAL: {tag} -> {source} is not there")
        with np.load(source, allow_pickle=False) as data:
            frame = str(data["coordinate_frame"].item())
            if frame != FRAME:
                raise SystemExit(f"FATAL: {tag} -> {source} is in {frame!r}, not {FRAME!r}")
        shutil.copy2(source, out_dir / source.name)
        used[str(Path(mapping[tag]))] = used.get(str(Path(mapping[tag])), 0) + 1
    return {
        "what": "a mixed per-episode frame-reference tree",
        "episodes": len(wanted),
        "sources": used,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="holds track_2/tier_*/")
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tier", choices=["tier1", "tier2"], help="emit this released trajectory")
    mode.add_argument("--assemble", type=Path, help='JSON map {"e00": "dir", ...}')
    parser.add_argument("-o", "--record", type=Path, help="write the record here")
    args = parser.parse_args(argv)

    if args.tier:
        record = released(args.dataset_root, args.tier, args.out)
    else:
        record = assemble(json.loads(args.assemble.read_text()), args.dataset_root, args.out)

    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "per_episode"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
