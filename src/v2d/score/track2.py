"""Score a Track 2 prediction locally with the host's own metric code.

The four scorers in the submission kit (`metric_code/track_2/{AUC,SP-SR,MP-SR,MPPE}.py`)
are the definition of record. This script does not reimplement them: it hash-checks each
file against `KIT_MANIFEST.json`, imports it, builds the host-shaped `solution` and
`submission` frames from the released dataset, and calls its `score()`. Local numbers are
therefore the leaderboard numbers by construction, with one caveat:

  AUC needs a per-object set of object-frame points, which the host carries in the
  solution column `auc_vertex_samples_b64` and does not publish. We rebuild it from the
  released `.glb` with the kit's own sampler (`v2dlb/vertices.py`: 500 vertices, seed 0),
  which is the documented convention. The kit warns the evaluator samples the *simulation*
  mesh instead, so AUC can differ in about the fourth decimal. SP-SR, MP-SR and MPPE do
  not depend on the mesh and are exact.

Predictions (`--pred`):
  reference        the clean Tier 1 trajectory (perfect-tracking sanity check)
  tier2-input      the released Tier 2 object input (the Tier 2 input ceiling)
  zeros            identity pose on every frame (what the kit's sample submission is)
  offset:DX,DY,DZ  reference translated by a constant, metres (analytic check)
  rotz:DEG         reference yawed by a constant angle (analytic check)
  csv:PATH         a packed submission CSV
  npz:DIR          per-episode episode_NNNNNN.npz holding pose_xyzw [T,1,7]

Usage:
  python scripts/score_track2.py --kit-root KIT --dataset-root DATA --pred reference
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import TIER_DIRS
from ..dataset import episodes, hold_last_visible, object_pose

METRIC_FILES = {
    "add_auc": "AUC.py",
    "spider_sr": "SP-SR.py",
    "maniptrans_sr": "MP-SR.py",
    "mppe_cm": "MPPE.py",
}
POSE_COLUMNS = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
REF_COLUMNS = ["ref_" + c for c in POSE_COLUMNS]
PROVENANCE = {
    "eval_script_sha256": "0" * 64,
    "eval_code_sha256": "0" * 64,
    "checkpoint_sha256": "0" * 64,
    "packer_script_sha256": "0" * 64,
    "uploader_script_sha256": "0" * 64,
    "code_commit_url": "https://github.com/local/scorer/commit/" + "0" * 40,
}


# --------------------------------------------------------------------------- kit loading
def load_metric_modules(kit_root: Path) -> dict:
    """Import the four kit scorers after checking each against KIT_MANIFEST.json."""
    manifest = json.loads((kit_root / "KIT_MANIFEST.json").read_text())["files"]
    modules = {}
    for metric, filename in METRIC_FILES.items():
        rel = f"metric_code/track_2/{filename}"
        path = kit_root / rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = manifest.get(rel)
        if expected and digest != expected:
            raise SystemExit(f"{rel}: sha256 {digest} != manifest {expected}")
        spec = importlib.util.spec_from_file_location(f"v2d_metric_{metric}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if metric != module._METRIC:
            raise SystemExit(f"{rel}: declares _METRIC={module._METRIC!r}, expected {metric!r}")
        modules[metric] = module
    return modules


def kit_sampler(kit_root: Path):
    """The kit's own vertex sampler, so AUC points match the documented convention."""
    if str(kit_root) not in sys.path:
        sys.path.insert(0, str(kit_root))
    from v2dlb.vertices import build_blob

    return build_blob


# --------------------------------------------------------------------------- dataset
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def build_prediction(
    spec: str, reference: np.ndarray, dataset_root: Path, episode: int, frames: int
) -> np.ndarray:
    """(T,7) predicted pose_xyzw for one episode."""
    if spec == "reference":
        return reference.copy()
    if spec == "zeros":
        out = np.zeros((frames, 7))
        out[:, 6] = 1.0
        return out
    if spec in ("tier2-input", "tier2-pos-only", "tier2-rot-only"):
        pose, visible = object_pose(dataset_root, "tier2", episode)
        pose, _ = hold_last_visible(pose, visible)
        if spec == "tier2-pos-only":  # Tier 2 position, clean orientation
            pose = np.concatenate([pose[:, :3], reference[:, 3:]], axis=1)
        elif spec == "tier2-rot-only":  # clean position, Tier 2 orientation
            pose = np.concatenate([reference[:, :3], pose[:, 3:]], axis=1)
        return pose
    if spec.startswith("offset:"):
        delta = np.array([float(v) for v in spec.split(":", 1)[1].split(",")], dtype=np.float64)
        out = reference.copy()
        out[:, :3] += delta
        return out
    if spec.startswith("rotz:"):
        half = np.radians(float(spec.split(":", 1)[1])) / 2.0
        spin = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
        out = reference.copy()
        out[:, 3:] = quat_mul(np.broadcast_to(spin, out[:, 3:].shape), out[:, 3:])
        return out
    if spec.startswith("npz:"):
        path = Path(spec.split(":", 1)[1]) / f"episode_{episode:06d}.npz"
        with np.load(path) as archive:
            pose = np.asarray(archive["pose_xyzw"], dtype=np.float64)
        return pose.reshape(pose.shape[0], -1)[:, :7]
    raise SystemExit(f"unknown --pred {spec!r}")


# --------------------------------------------------------------------------- frames
def scored_grid(kit_root: Path, grid: str) -> set[str] | None:
    """The row_ids the host actually scores, or None for every released frame.

    The released data has 22,990 frames but the scored grid has 22,989: episode 11's last
    frame (f000900) is absent from the kit's sample submission, and the uploader checks a
    submission's rows against that sample. So the sample is the authority on what is scored.
    """
    if grid == "all":
        return None
    path = Path(grid) if grid != "sample" else kit_root / "data" / "track_2_sample_submission.csv"
    return set(pd.read_csv(path, usecols=["row_id"], dtype=str)["row_id"])


def build_frames(
    dataset_root: Path, kit_root: Path, pred: str, tier: str, grid: str = "sample"
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Host-shaped (solution, submission) frames plus a few diagnostics."""
    metadata = episodes(dataset_root)
    allowed = scored_grid(kit_root, grid)
    build_blob = kit_sampler(kit_root)
    mesh_root = dataset_root / "track_2" / TIER_DIRS["tier1"] / "mesh"
    names = sorted({row.object_name for row in metadata})
    blob, _ = build_blob(mesh_root, names)

    pred_csv = None
    if pred.startswith("csv:"):
        pred_csv = pd.read_csv(Path(pred.split(":", 1)[1]), dtype={"row_id": str})
        pred_csv = pred_csv.set_index("row_id")

    sol_rows, sub_rows, invisible, dropped = [], [], 0, 0
    for row in metadata:
        episode = row.index
        reference, visible = object_pose(dataset_root, "tier1", episode)
        reference, filled = hold_last_visible(reference, visible)
        invisible += filled
        released = len(reference)

        all_ids = [f"track_2/e{episode:06d}/w000/f{frame:06d}/o0" for frame in range(released)]
        keep = (
            np.ones(released, dtype=bool)
            if allowed is None
            else np.array([rid in allowed for rid in all_ids])
        )
        dropped += int((~keep).sum())
        row_ids = [rid for rid, k in zip(all_ids, keep, strict=True) if k]
        frame_index = np.arange(released)[keep]
        frames = len(row_ids)
        solution = pd.DataFrame(
            {
                "row_id": row_ids,
                "episode_index": episode,
                "world_index": 0,
                "frame_index": frame_index,
                "object_slot": 0,
                "object_name": row.object_name,
            }
        )
        solution[REF_COLUMNS] = reference[keep]
        solution["auc_vertex_samples_b64"] = ""
        sol_rows.append(solution)

        if pred_csv is not None:
            block = pred_csv.reindex(row_ids)
            missing = block["pos_x"].isna().sum()
            if missing:
                raise SystemExit(f"episode {episode}: CSV is missing {missing} of {frames} rows")
            prediction = block[POSE_COLUMNS].to_numpy(dtype=np.float64)
        else:
            prediction = build_prediction(pred, reference, dataset_root, episode, released)[keep]
        if prediction.shape != (frames, 7):
            raise SystemExit(
                f"episode {episode}: prediction is {prediction.shape}, expected {(frames, 7)}"
            )

        submission = pd.DataFrame({"row_id": row_ids})
        submission[POSE_COLUMNS] = prediction
        sub_rows.append(submission)

    solution = pd.concat(sol_rows, ignore_index=True)
    submission = pd.concat(sub_rows, ignore_index=True)
    # Exactly one row carries the AUC payload, as _load_vertices requires.
    solution.loc[0, "auc_vertex_samples_b64"] = blob
    for column, value in PROVENANCE.items():
        submission[column] = value
    return (
        solution,
        submission,
        {
            "frames": len(solution),
            "invisible_frames_filled": invisible,
            "frames_outside_scored_grid": dropped,
        },
    )


# --------------------------------------------------------------------------- scoring
def score_all(modules: dict, solution: pd.DataFrame, submission: pd.DataFrame) -> dict:
    return {
        metric: float(module.score(solution.copy(), submission.copy(), "row_id"))
        for metric, module in modules.items()
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="v2d score",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kit-root", type=Path, required=True, help="extracted v2d_submission_kit/"
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="holds track_2/tier_*/")
    parser.add_argument(
        "--pred", default="reference", help="prediction source; see the module docstring"
    )
    parser.add_argument(
        "--tier",
        default="tier2",
        choices=sorted(TIER_DIRS),
        help="tier label recorded in the output",
    )
    parser.add_argument(
        "--grid",
        default="sample",
        help="scored rows: 'sample' (the kit's sample submission, the host's grid), 'all' "
        "(every released frame), or a path to another sample CSV",
    )
    parser.add_argument(
        "--per-episode", action="store_true", help="also score each episode on its own"
    )
    parser.add_argument("-o", "--out", type=Path, help="write the result as JSON")
    args = parser.parse_args(argv)

    modules = load_metric_modules(args.kit_root)
    solution, submission, diagnostics = build_frames(
        args.dataset_root, args.kit_root, args.pred, args.tier, args.grid
    )
    overall = score_all(modules, solution, submission)

    result = {
        "pred": args.pred,
        "tier": args.tier,
        "scorer": "submission kit metric_code/track_2 (sha256-checked against KIT_MANIFEST.json)",
        "auc_note": "AUC vertices rebuilt from released .glb with v2dlb/vertices.py "
        "(500 pts, seed 0); the host samples the simulation mesh, so AUC can differ in "
        "~the 4th decimal",
        "overall": overall,
        **diagnostics,
    }

    if args.per_episode:
        per_episode = {}
        for episode, group in solution.groupby("episode_index", sort=True):
            block = submission[submission["row_id"].isin(set(group["row_id"]))]
            one = group.copy()
            one["auc_vertex_samples_b64"] = ""
            one.loc[one.index[0], "auc_vertex_samples_b64"] = solution.loc[
                0, "auc_vertex_samples_b64"
            ]
            per_episode[int(episode)] = score_all(modules, one, block)
        result["per_episode"] = per_episode

    result["grid"] = args.grid
    print(json.dumps(result["overall"], indent=2))
    print(
        f"frames={diagnostics['frames']} invisible_filled={diagnostics['invisible_frames_filled']} "
        f"outside_grid={diagnostics['frames_outside_scored_grid']}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
