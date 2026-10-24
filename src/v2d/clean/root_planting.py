"""A deployable foot-planting cleaner for Tier 2's human root perturbation.

`docs/EXP_CLEANUP_CONTACT.md` showed that a *perfect* object-pose cleaner still leaves 38 % of
the hand-object contact loss on the table (0.0777 of 0.2043), because Tier 2 also perturbs the
human. That perturbation is narrow and completely characterised
(`scripts/probe_support_root.py`): the root joint's rotation by 5.17 deg on average and the root
translation by 1.09 cm, with **all 76 non-root joints bit-identical**. So the body's shape and
articulation are untouched and the noise is a rigid, time-varying displacement of the whole body.
Only 38.9 % of it is a constant per-clip offset, so a per-frame estimator is needed.

The argument this rests on is the object cleaner's, applied to the body: **a planted foot does
not move.** During a stance segment the stance foot's world pose should be constant, so whatever
rigid correction holds it still is an estimate of the perturbation. Two properties make this the
right constraint rather than a convenient one:

- The body above a planted foot moves through the other joints, which Tier 2 leaves alone. So
  pinning the stance foot removes displacement without suppressing real motion.
- It needs no ground plane. The released plane is expressed in the object frame, and the SOMA
  body comes out in its own frame (its feet sit ~270 cm from that plane), so a floor-referenced
  constraint would need the transform the release does not publish. "Stationary" is
  frame-independent.

Per episode:

1. **Foot vertex sets.** The body model ships a `segment_feet` vertex list; each vertex is
   assigned to the nearer of the two toe-end joints at frame 0, splitting it by side.
2. **Stance detection**, per foot, on a `--smooth-frames` moving average of the foot centroid,
   with the same reasoning as the object cleaner: differentiating a noisy position at 30 fps
   turns jitter into metres per second. The threshold is **each clip's own
   `--stance-quantile` of that foot's speed**, not an absolute speed, because an absolute one
   provably cannot work here. Measured on paired clips (`results/probe_root_stance.json`): the
   median slower-foot speed is **0.0053 m/s in Tier 1 and 0.0330 in Tier 2, a 6.2x noise
   floor**. At 0.02 m/s Tier 1 calls 85 % of frames planted and Tier 2 only 22 %; at 0.05 m/s
   Tier 2 reaches 77 % but Tier 1 is at 97 %; by 0.10 m/s both exceed 99 % and the test is
   vacuous. No fixed speed recovers Tier 1's stance set from Tier 2's data. A per-clip quantile
   sidesteps the noise floor entirely by asking only which frames are *quietest*, which the
   perturbation does not reorder much.

   The default quantile is 0.4: Tier 1 says stance is the majority state in these clips (85 % of
   frames at 0.02 m/s, since the subject mostly stands and manipulates), so taking the quietest
   40 % per foot is conservative by more than a factor of two. Contiguous runs of at least
   `--min-stance-frames` count.

   A cheaper idea was tried first and **rejected by measurement**: since the perturbation is a
   rigid transform of the whole body, it ought to move both feet alike, so the *difference*
   between the two feet's speeds should reject it as common mode. The left-right speed
   correlation is **-0.098, +0.019 and +0.035** across the probed clips, i.e. nothing is shared.
   A rotation about the root displaces two feet on opposite sides of it in opposite directions,
   so there is no common mode to reject.
3. **Per-frame rigid correction.** For each frame, take whichever feet are in a stance segment,
   and Kabsch-fit their current vertices onto that segment's median vertex positions. The result
   is the rigid transform that puts the planted foot back where it stayed. Both feet are used
   when both are planted, which conditions the fit better than one foot alone.
4. **Sanity cap.** A correction whose rotation exceeds `--max-root-deg` is discarded rather than
   applied: Tier 2's root error never exceeds 8 deg, so anything far above that is a stance
   mis-detection, not a perturbation estimate.
5. **Propagation.** Slerp the rotation and lerp the translation between corrected frames, held
   flat outside the first and last, exactly as the object cleaner propagates tilt.
6. **Mechanism check.** Whether the correction actually holds the stance foot still, as the
   stance-foot centroid's 3D scatter within its own segment before and after. The
   pre-registration asked for the *vertical* std; there is no gravity axis in this frame to
   project onto, so 3D scatter stands in and is the harder test.
7. **Write-back.** A rigid world transform (R_d, t_d) is re-expressed in the two fields Tier 2
   actually perturbs. With `V_world = R_root (V_posed - J0) + J0 + transl`, applying
   `V -> R_d V + t_d` means `R_root' = R_d R_root` and `transl' = R_d (J0 + transl) + t_d - J0`.
   `J0` is the root joint's rest position, read from a zero-pose forward pass. `--self-check`
   verifies the identity numerically before anything is written.

Output is one `episode_%06d.npz` per episode holding `root_rotvec (T,3)` and `translation (T,3)`,
which `make_soma_sequences.py --human-npz DIR` substitutes for the released values. Nothing else
in the episode changes.

Usage:
  python scripts/clean_root_planting.py --seq-root DIR --repo DIR --soma-root DIR \
      --dataset-root DATA --npz-out DIR [--self-check] [-o results/clean_root_planting.json]

  The sequence tree comes from:
  python scripts/make_soma_sequences.py --dataset-root DATA --tier tier2 --out DIR
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..geometry import minimal_rotation

TIER1 = "tier_1_multiview_caption"


def smooth(position: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge padding, so differentiating does not amplify the noise.

    Duplicated from scripts/clean_support_contact.py rather than imported: these tools run under
    `python -I`, which implies `-P`, so the script's own directory is not on sys.path and
    scripts/ cannot import from itself.
    """
    if window <= 1:
        return position
    pad = window // 2
    padded = np.pad(position, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack(
        [
            np.convolve(padded[:, axis], kernel, mode="valid")[: len(position)]
            for axis in range(position.shape[1])
        ],
        axis=1,
    )


def slow_segments(
    position: np.ndarray, fps: float, quantile: float, min_frames: int, window: int
) -> tuple[list[tuple[int, int]], np.ndarray, float]:
    """Contiguous [start, stop) runs among this foot's own quietest `quantile` of frames.

    Returns the segments, the per-frame speed, and the speed the quantile resolved to, so the
    threshold a clip actually used is recorded rather than assumed.
    """
    speed = np.linalg.norm(np.gradient(smooth(position, window), axis=0), axis=-1) * fps
    limit = float(np.quantile(speed, quantile))
    slow = speed <= limit
    segments, start = [], None
    for index, value in enumerate(slow):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_frames:
                segments.append((start, index))
            start = None
    if start is not None and len(slow) - start >= min_frames:
        segments.append((start, len(slow)))
    return segments, speed, limit


#: The retargeter consumes poses.npy as einsum("ij,tjk->tik", diag(1,-1,-1,1), poses), so the
#: released world and the SOMA body frame differ by this. It is a proper rotation (180 deg about
#: x, det = +1) and an involution, so applying it to the body or to the plane is equivalent.
#: Reading the resulting ~270 cm body-to-plane distance as an unreachable frame rather than as an
#: unapplied flip is what sent the first root cleaner down a dead end; see
#: docs/EXP_SUPPORT_AND_ROOT.md.
CV_TO_SOMA_FLIP = np.diag([1.0, -1.0, -1.0])


def sole_plane(
    points: np.ndarray, normal: np.ndarray, offset: float, band_m: float = 0.02
) -> np.ndarray:
    """The unit normal of the sole, from the foot vertices nearest the ground.

    Fitting a plane through the *whole* foot does not give the sole: a foot is a 3D shape with
    toes, heel, instep and ankle, so the smallest singular direction of the full cloud follows
    the foot's overall form. Measured, that produced 12 degree corrections and 16 cm body shifts
    against a true root error of 5.17 degrees and 1.09 cm. Only the vertices within `band_m` of
    the lowest one are on the sole, and those are what a floor can speak to.
    """
    height = points @ normal + offset
    near_ground = points[height <= height.min() + band_m]
    if len(near_ground) < 8:  # degenerate contact patch; fall back to the whole foot
        near_ground = points
    centred = near_ground - near_ground.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[-1] / np.linalg.norm(vt[-1])


def floor_correction(
    foot: np.ndarray, normal: np.ndarray, offset: float, max_deg: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """The rigid correction putting a stance foot's sole flat on, and touching, the floor.

    A floor pins three of six degrees of freedom: the height, and the two tilt angles. Yaw and
    horizontal position are left alone, because a plane carries no information about them --
    which is an honest accounting of what this reference can do, not a shortfall.
    """
    sole = sole_plane(foot, normal, offset)
    # Point the sole normal downward, so the correction levels the foot rather than flipping it.
    if float(np.dot(sole, normal)) > 0:
        sole = -sole
    rotation = minimal_rotation(sole, -normal)
    if float(np.degrees(rotation.magnitude())) > max_deg:
        return None
    pivot = foot.mean(axis=0)
    matrix = rotation.as_matrix()
    # Rotate about the foot centroid, then drop the foot until its lowest vertex touches.
    rotated = (foot - pivot) @ matrix.T + pivot
    translation = pivot - matrix @ pivot
    lowest = float((rotated @ normal + offset).min())
    return matrix, translation - lowest * normal


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rigid (R, t) minimising |R source + t - target|, with a proper rotation."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    return rotation, target_mean - rotation @ source_mean


def clean_episode(
    vertices: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    args,
    plane: tuple[np.ndarray, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return per-frame (rotation (T,3,3), translation (T,3)) corrections and diagnostics.

    With `plane` and `--anchor floor`, each stance frame is corrected against the released ground
    plane: an external reference. Without it the correction is fitted to the stance segment's own
    median, which is a statistic of the noisy data and was measured to make the root error worse
    (5.17 -> 5.58 deg).
    """
    frames = len(vertices)
    # One stance segment list per foot, and the median stance pose each one should hold.
    planted: list[list[tuple[int, int, np.ndarray]]] = []
    limits = []
    for side in (left, right):
        centroid = vertices[:, side].mean(axis=1)
        segments, _, limit = slow_segments(
            centroid, args.fps, args.stance_quantile, args.min_stance_frames, args.smooth_frames
        )
        limits.append(round(limit, 5))
        planted.append(
            [
                (start, stop, np.median(vertices[start:stop][:, side], axis=0))
                for start, stop in segments
            ]
        )

    corrections: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    applied_deg, rejected_deg = [], []
    if getattr(args, "anchor", "planting") == "floor":
        if plane is None:
            raise ValueError("--anchor floor needs the released ground plane")
        normal, offset = plane
        for frame in range(frames):
            # The lowest-footed stance foot at this frame is the one on the ground.
            feet_here = [
                side
                for side, segments in zip((left, right), planted, strict=True)
                if any(start <= frame < stop for start, stop, _ in segments)
            ]
            if not feet_here:
                continue
            chosen = min(
                feet_here, key=lambda side: float((vertices[frame][side] @ normal + offset).min())
            )
            found = floor_correction(vertices[frame][chosen], normal, offset, args.max_root_deg)
            if found is None:
                rejected_deg.append(0.0)
                continue
            rotation, translation = found
            corrections[frame] = (rotation, translation)
            applied_deg.append(float(np.degrees(Rotation.from_matrix(rotation).magnitude())))
    else:
        for frame in range(frames):
            source, target = [], []
            for side, segments in zip((left, right), planted, strict=True):
                for start, stop, reference in segments:
                    if start <= frame < stop:
                        source.append(vertices[frame][side])
                        target.append(reference)
                        break
            if not source:
                continue
            rotation, translation = kabsch(np.concatenate(source), np.concatenate(target))
            angle = float(np.degrees(Rotation.from_matrix(rotation).magnitude()))
            # Tier 2's root error never exceeds 8 deg; far above that is a stance mis-detection.
            if angle > args.max_root_deg:
                rejected_deg.append(round(angle, 3))
                continue
            corrections[frame] = (rotation, translation)
            applied_deg.append(angle)

    diagnostics = {
        "frames": frames,
        "stance_segments": [len(s) for s in planted],
        "stance_frames": [sum(stop - start for start, stop, _ in s) for s in planted],
        "stance_speed_limit_m_per_s": limits,
        "corrected_frames": len(corrections),
        "rejected_frames": len(rejected_deg),
        "correction_deg_mean": round(float(np.mean(applied_deg)), 3) if applied_deg else 0.0,
        "correction_deg_max": round(float(np.max(applied_deg)), 3) if applied_deg else 0.0,
    }
    if not corrections:
        diagnostics["status"] = "no stance segments; left unchanged"
        return (np.tile(np.eye(3), (frames, 1, 1)), np.zeros((frames, 3)), diagnostics)

    keys = np.array(sorted(corrections))
    key_rotations = Rotation.from_matrix(np.stack([corrections[int(k)][0] for k in keys]))
    key_translations = np.stack([corrections[int(k)][1] for k in keys])
    index = np.clip(np.arange(frames), keys[0], keys[-1])
    if len(keys) == 1:
        rotations = Rotation.concatenate([key_rotations[0]] * frames).as_matrix()
        translations = np.tile(key_translations[0], (frames, 1))
    else:
        rotations = Slerp(keys, key_rotations)(index).as_matrix()
        translations = np.stack(
            [np.interp(index, keys, key_translations[:, axis]) for axis in range(3)], axis=1
        )
    # The mechanism check: does the correction actually hold the stance foot still? Measured as
    # the stance-foot centroid's scatter within its own segment, before and after, in 3D.
    #
    # The pre-registration asked for the *vertical* std. That is not measurable here: the
    # released ground plane lives in the object frame while the body comes out in SOMA's own,
    # with no published transform between them, so there is no gravity axis to project onto.
    # 3D scatter is the honest substitute and is strictly the harder test, since it contains the
    # vertical component plus two more.
    before, after = [], []
    for side, segments in zip((left, right), planted, strict=True):
        for start, stop, _ in segments:
            raw = vertices[start:stop][:, side].mean(axis=1)
            fixed = np.einsum("tij,tj->ti", rotations[start:stop], raw) + translations[start:stop]
            before.append(float(np.linalg.norm(raw - raw.mean(axis=0), axis=-1).mean()))
            after.append(float(np.linalg.norm(fixed - fixed.mean(axis=0), axis=-1).mean()))
    if before:
        diagnostics["stance_scatter_cm_before"] = round(float(np.mean(before)) * 100, 4)
        diagnostics["stance_scatter_cm_after"] = round(float(np.mean(after)) * 100, 4)
        diagnostics["stance_scatter_drop"] = (
            round(1.0 - float(np.mean(after)) / float(np.mean(before)), 4)
            if np.mean(before) > 0
            else 0.0
        )
    diagnostics["status"] = "cleaned"
    return rotations, translations, diagnostics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seq-root",
        type=Path,
        required=True,
        help="tier2 sequence tree from make_soma_sequences.py",
    )
    parser.add_argument("--repo", type=Path, required=True, help="extracted challenge repo")
    parser.add_argument("--soma-root", type=Path, required=True, help="body_models/soma")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--npz-out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--anchor",
        choices=["planting", "floor"],
        default="floor",
        help="what the correction is measured against. 'floor' uses the released ground plane, "
        "an external reference. 'planting' fits the stance segment's own median, which is a "
        "statistic of the noisy data and was measured to make the root error worse.",
    )
    parser.add_argument(
        "--stance-quantile",
        type=float,
        default=0.4,
        help="a foot is planted in its own quietest fraction of frames; an "
        "absolute speed cannot work, see the module docstring",
    )
    parser.add_argument("--min-stance-frames", type=int, default=5)
    parser.add_argument("--smooth-frames", type=int, default=9)
    parser.add_argument(
        "--max-root-deg",
        type=float,
        default=15.0,
        help="discard a per-frame correction above this as a stance mis-detection",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="verify the write-back identity numerically on the first episode",
    )
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.repo / "robotic_grounding/source/robotic_grounding"))
    import torch
    from robotic_grounding.retarget.read_soma import SOMA

    reader = SOMA(data_root=args.soma_root, device=torch.device("cpu"))
    with np.load(args.soma_root / "SOMA_neutral.npz", allow_pickle=True) as archive:
        feet = np.asarray(archive["segment_feet"]).ravel()

    root = args.dataset_root / "track_2" / TIER1
    metadata = [
        json.loads(line)
        for line in (root / "meta" / "episodes_metadata.jsonl").read_text().splitlines()
        if line.strip()
    ]
    metadata.sort(key=lambda r: r["episode_index"])
    args.npz_out.mkdir(parents=True, exist_ok=True)

    episodes, checked = {}, False
    for row in metadata:
        episode = int(row["episode_index"])
        if args.episodes is not None and episode not in args.episodes:
            continue
        params_path = args.seq_root / f"sequence_id={row['sequence_id']}" / "soma_params.npz"
        motion = reader.load_motion(params_path)
        vertices = np.asarray(motion["vertices"], dtype=np.float64)
        names = [str(n) for n in motion["joint_names"]]
        joints = np.asarray(motion["joints"], dtype=np.float64)

        # Split the shipped foot vertex set by side: nearer toe-end joint at frame 0.
        toes = np.stack(
            [joints[0, names.index("LeftToeEnd")], joints[0, names.index("RightToeEnd")]]
        )
        side_of = np.argmin(
            np.linalg.norm(vertices[0, feet][:, None, :] - toes[None], axis=-1), axis=1
        )
        left, right = feet[side_of == 0], feet[side_of == 1]

        plane = None
        if args.anchor == "floor":
            # The plane ships in the released world frame; flip it into the body's frame. Doing
            # it to the plane rather than to 18k vertices per frame is the cheap direction.
            raw = np.asarray(
                json.loads((root / row["ground_plane"]).read_text())["plane"], dtype=np.float64
            )
            normal = CV_TO_SOMA_FLIP @ (raw[:3] / np.linalg.norm(raw[:3]))
            offset = float(raw[3] / np.linalg.norm(raw[:3]))
            # Orient it so the body sits on the positive side, from the noisy body alone.
            if float(np.mean(vertices.reshape(-1, 3) @ normal + offset)) < 0:
                normal, offset = -normal, -offset
            plane = (normal, offset)
        rotations, translations, diagnostics = clean_episode(vertices, left, right, args, plane)
        diagnostics["anchor"] = args.anchor

        with np.load(params_path, allow_pickle=True) as archive:
            poses = np.asarray(archive["poses"], dtype=np.float64)
            transl = np.asarray(archive["transl"], dtype=np.float64)
            identity = np.asarray(archive["identity_coeffs"], dtype=np.float64)
            scale = np.asarray(archive["scale_params"], dtype=np.float64)

        # J0: the root joint's rest position, from a zero-pose forward pass with this identity.
        with torch.no_grad():
            rest = reader.layer(
                torch.zeros((1, *poses.shape[1:]), dtype=torch.float32),
                torch.from_numpy(identity[:1]).float(),
                scale_params=torch.from_numpy(scale[:1]).float(),
                transl=torch.zeros((1, 3), dtype=torch.float32),
                apply_correctives=False,
            )
            j0 = rest["joints"].detach().cpu().numpy()[0, 0].astype(np.float64)

        root_rotation = Rotation.from_rotvec(poses[:, 0])
        new_root = Rotation.from_matrix(rotations) * root_rotation
        new_transl = np.einsum("tij,tj->ti", rotations, j0 + transl) + translations - j0

        if args.self_check and not checked:
            checked = True
            probe = np.array([0, len(poses) // 2, len(poses) - 1])
            edited = poses.copy()
            edited[:, 0] = new_root.as_rotvec()
            with torch.no_grad():
                out = reader.layer(
                    torch.from_numpy(edited[probe]).float(),
                    torch.from_numpy(np.repeat(identity[:1], len(probe), axis=0)).float(),
                    scale_params=torch.from_numpy(np.repeat(scale[:1], len(probe), axis=0)).float(),
                    transl=torch.from_numpy(new_transl[probe]).float(),
                    apply_correctives=False,
                )
            got = out["vertices"].detach().cpu().numpy().astype(np.float64)
            want = (
                np.einsum("tij,tvj->tvi", rotations[probe], vertices[probe])
                + translations[probe, None, :]
            )
            error = float(np.abs(got - want).max())
            print(
                f"self-check on e{episode:02d}: max vertex error {error * 1000:.4f} mm "
                f"(J0={np.round(j0, 4)})"
            )
            if error > 1e-4:
                raise SystemExit(f"write-back identity is wrong: {error * 1000:.4f} mm > 0.1 mm")

        np.savez(
            args.npz_out / f"episode_{episode:06d}.npz",
            episode_index=episode,
            frame_index=np.arange(len(poses)),
            root_rotvec=new_root.as_rotvec(),
            translation=new_transl,
        )
        # How much the body actually moved, which is the thing the retargeter will see.
        shift = np.einsum("tij,tvj->tvi", rotations, vertices) + translations[:, None, :] - vertices
        diagnostics["body_shift_cm_mean"] = round(
            float(np.linalg.norm(shift, axis=-1).mean() * 100), 4
        )
        diagnostics["body_shift_cm_max"] = round(
            float(np.linalg.norm(shift, axis=-1).max() * 100), 4
        )
        diagnostics["object"] = row["object"]
        episodes[episode] = diagnostics
        # 18k vertices x ~900 frames per clip, and torch does not hand the memory back between
        # forward passes: without this the loop is killed around the fourth episode on this box.
        del motion, vertices, joints, rotations, translations, shift
        gc.collect()
        print(
            f"e{episode:02d} {row['object'][:22]:22s} {diagnostics['status']:30s} "
            f"stance={diagnostics['stance_frames']}/{diagnostics['frames']} "
            f"corrected={diagnostics['corrected_frames']:4d} "
            f"rot={diagnostics['correction_deg_mean']:5.2f} deg  "
            f"body shift {diagnostics['body_shift_cm_mean']:5.2f} cm"
        )

    summary = {
        "episodes": len(episodes),
        "cleaned": sum(1 for v in episodes.values() if v["status"] == "cleaned"),
        "unchanged": sum(1 for v in episodes.values() if v["status"] != "cleaned"),
        "anchor": args.anchor,
        "stance_quantile": args.stance_quantile,
        "smooth_frames": args.smooth_frames,
        "max_root_deg": args.max_root_deg,
        "mean_stance_fraction": round(
            float(
                np.mean([sum(v["stance_frames"]) / (2 * v["frames"]) for v in episodes.values()])
            ),
            4,
        ),
        "mean_corrected_fraction": round(
            float(np.mean([v["corrected_frames"] / v["frames"] for v in episodes.values()])), 4
        ),
        "mean_correction_deg": round(
            float(np.mean([v["correction_deg_mean"] for v in episodes.values()])), 3
        ),
        "mean_body_shift_cm": round(
            float(np.mean([v["body_shift_cm_mean"] for v in episodes.values()])), 4
        ),
        "mean_stance_scatter_drop": round(
            float(
                np.mean(
                    [
                        v["stance_scatter_drop"]
                        for v in episodes.values()
                        if "stance_scatter_drop" in v
                    ]
                )
            ),
            4,
        )
        if any("stance_scatter_drop" in v for v in episodes.values())
        else None,
        #: R3 as the pre-registration phrased it asked for a *vertical* std, which is not
        #: measurable in SOMA's frame (no published gravity axis); this is the 3D substitute.
        "stance_scatter_drop_30pct_clips": sum(
            1 for v in episodes.values() if v.get("stance_scatter_drop", 0) >= 0.30
        ),
    }
    print("\n" + json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "what": "deployable foot-planting cleaner for the Tier 2 human root; "
                    "no clean reference",
                    "summary": summary,
                    "per_episode": episodes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
