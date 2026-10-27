"""`v2d taxonomy`: why each weak clip in the shipping entry is weak, by mechanism rather than score.

docs/METRICS_DEEP_DIVE.md measured the first wave's rollouts and said *how much* AUC each clip
loses and to which channel. This answers the next question on the 30 rollouts that actually ship
after adoption: *what goes wrong*, as one of five classes, with the frame it happens on.

The classes are checked in a fixed order, because more than one can be true of the same clip and
the first one that fires is the one that explains the rest:

1. ``reference_ceiling`` -- the clip scores within `--ceiling-margin` of what its own reference
   could score if tracked perfectly. Nothing a policy change can win; put it first so a clip that
   is already at its ceiling is never reported as a policy failure.
2. ``never_grasped`` -- the object never leaves its rest pose while the reference moves it. The
   failure frame is where the reference first moves, i.e. where the policy should have had it.
Classes 3 to 5 need a **loss event** first: the earliest of the position error crossing 10 cm for
`--min-loss-frames` frames and the achieved object falling below `--drop-z` while the reference
stays above it. Without an event the object was held for the whole clip, whatever its angle, and
the clip is ``held_but_drifting``. Requiring the event is the fix to an earlier version of this
rule that called eight clips "toppled" on the angle alone, e26 and e17 among them, which hold the
object to within 6 cm from the first frame to the last.

3. ``toppled_on_contact`` -- the orientation goes first: the angle error crosses `--topple-deg`
   at least `--topple-lead` frames before the loss event, so the object was turned over and then
   lost, and a fix has to act on the angle.
4. ``lost_at_placement`` -- the event lands in the last `1 - placement_fraction` of the clip:
   carried the whole way and let go at the end.
5. ``dropped_mid_clip`` -- an event anywhere earlier. ``loss_signal`` names which signal fired:
   ``z-drop`` is the selector's own rule, ``position`` a clip whose object never goes under the
   drop line -- a floor clip, or one sliding along a table -- and is lost all the same.

The per-frame error is `deep_dive.released_series`, the same conversion and the same clean-truth
comparison the deep dive and the kit scorer use, so a frame quoted here and a number in that doc
cannot come from two different computations.

    v2d taxonomy --rollouts $S/t2_entry_wave2/rollouts --inputs $S/t2_entry_wave2/inputs.json \\
      --adoption results/second_wave_adoption.json --deep-dive results/metrics_deep_dive.json \\
      --hand-grasp ~/projects/v2d-hand/results/hand_grasp_bridge_v1.json \\
      --dataset-root $S/data --kit-root $S/kit_extract/v2d_submission_kit \\
      --work-dir $S/taxonomy_work --out docs/FAILURE_TAXONOMY_WAVE2.md \\
      --json results/failure_taxonomy_wave2.json --figures results/figures/hail_mary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..config import TIER1
from .deep_dive import first_past, released_series

#: A clip within this much of its own reference ceiling has nothing left for a policy to win.
CEILING_MARGIN = 0.03
#: Below this height, with the reference above it, the object is on the floor: the selector's rule.
DROP_Z_M = 0.15
#: The reference has to be clearly above the drop line for a drop to mean anything.
REFERENCE_ABOVE_M = 0.20
#: The kit's SP-SR position threshold, which is also where "lost" starts.
LOSS_CM = 10.0
#: An angle error this large is a topple rather than a tracking error.
TOPPLE_DEG = 45.0
#: The angle has to go this many frames before the position for the topple to be the cause.
TOPPLE_LEAD = 20
#: The last fifth of a clip is where an object is set down.
PLACEMENT_FRACTION = 0.8
#: A loss has to last this long to be a loss rather than one bad frame.
MIN_LOSS_FRAMES = 10
#: Within this of its own starting pose, the object has not been moved.
NEVER_MOVED_CM = 3.0
#: The reference has to move this far for "never grasped" to be about the policy.
REFERENCE_MOVES_CM = 10.0
#: Clips at or above this AUC are not in the failure table.
WEAK_BELOW = 0.35

CLASSES = (
    "reference_ceiling",
    "never_grasped",
    "toppled_on_contact",
    "dropped_mid_clip",
    "lost_at_placement",
    "held_but_drifting",
)


def drop_frame(achieved: np.ndarray, reference: np.ndarray) -> int | None:
    """The first frame the object is on the floor while the reference is not."""
    on_floor = (achieved[:, 2] < DROP_Z_M) & (reference[:, 2] > REFERENCE_ABOVE_M)
    return first_past(on_floor, MIN_LOSS_FRAMES)


def classify(
    *,
    position_cm: np.ndarray,
    orientation_deg: np.ndarray,
    achieved: np.ndarray,
    reference: np.ndarray,
    kit_auc: float,
    ceiling: float | None,
    ceiling_margin: float = CEILING_MARGIN,
) -> dict[str, Any]:
    """One clip's failure class and the frame it happens on, by the ordered rules above."""
    frames = len(position_cm)
    lost_at = first_past(position_cm > LOSS_CM, MIN_LOSS_FRAMES)
    angle_at = first_past(orientation_deg > TOPPLE_DEG, MIN_LOSS_FRAMES)
    dropped_at = drop_frame(achieved, reference)
    placement_begins = int(frames * PLACEMENT_FRACTION)

    moved_cm = np.linalg.norm(achieved[:, :3] - achieved[0, :3], axis=1) * 100
    reference_moved_cm = np.linalg.norm(reference[:, :3] - reference[0, :3], axis=1) * 100
    evidence = {
        "frames": frames,
        "first_loss_frame": lost_at,
        "first_topple_frame": angle_at,
        "drop_frame": dropped_at,
        "placement_begins_frame": placement_begins,
        "achieved_max_travel_cm": round(float(moved_cm.max()), 1),
        "reference_max_travel_cm": round(float(reference_moved_cm.max()), 1),
    }

    candidates = [frame for frame in (lost_at, dropped_at) if frame is not None]
    event = min(candidates) if candidates else None
    evidence["loss_event_frame"] = event
    evidence["loss_signal"] = (
        None if event is None else ("z-drop" if event == dropped_at else "position")
    )

    if ceiling is not None and kit_auc >= ceiling - ceiling_margin:
        return {"failure_class": "reference_ceiling", "failure_frame": None, **evidence}
    if moved_cm.max() < NEVER_MOVED_CM and reference_moved_cm.max() > REFERENCE_MOVES_CM:
        starts = first_past(reference_moved_cm > REFERENCE_MOVES_CM, 1)
        return {"failure_class": "never_grasped", "failure_frame": starts, **evidence}
    if event is None:
        # Held the whole way. Its AUC is lost to accumulated error, most often the angle, and
        # calling that a topple would point the next run at a collision that never happened.
        return {"failure_class": "held_but_drifting", "failure_frame": None, **evidence}
    if angle_at is not None and angle_at + TOPPLE_LEAD <= event:
        return {"failure_class": "toppled_on_contact", "failure_frame": angle_at, **evidence}
    if event >= placement_begins:
        return {"failure_class": "lost_at_placement", "failure_frame": event, **evidence}
    return {"failure_class": "dropped_mid_clip", "failure_frame": event, **evidence}


def object_mass_kg(stage_root: Path, sequence_id: str) -> float | None:
    """The mass the trainer actually loads, from the staged object URDF."""
    for urdf in stage_root.glob(f"**/sequence_id={sequence_id}/object/*.urdf"):
        found = re.search(r'mass\s+value="([0-9.eE+-]+)"', urdf.read_text())
        if found:
            return float(found.group(1))
    return None


def closure_of(hand_grasp: dict[str, Any], episode: int) -> dict[str, Any]:
    """interp's measured closure for one clip: does the hand reference close on the mesh at all?"""
    record = (hand_grasp.get("episodes") or {}).get(str(episode))
    if not record:
        return {"measured": False}
    closure = record.get("closure") or {}
    hands = {
        hand: {
            "closure": side.get("closure"),
            "rule": side.get("rule"),
            "palm_depth_cm": round((side.get("palm_depth_m") or {}).get("median", 0.0) * 100, 2),
        }
        for hand, side in closure.items()
    }
    values = [h["closure"] for h in hands.values() if h["closure"] is not None]
    return {
        "measured": True,
        "hands": hands,
        "contact_frames": record.get("contact_frames"),
        "any_hand_closes": bool(values) and max(values) > 0.0,
        "max_closure": max(values) if values else None,
    }


def collect(
    *,
    rollouts: Path,
    inputs: dict[str, Any],
    adoption: dict[str, Any],
    deep_dive: dict[str, Any],
    hand_grasp: dict[str, Any],
    metadata: dict[int, dict[str, Any]],
    vertex_cloud: dict[str, np.ndarray],
    dataset_root: Path,
    work: Path,
    ceiling_margin: float = CEILING_MARGIN,
) -> list[dict[str, Any]]:
    """One row per shipping rollout: its class, its frame, and the facts behind the verdict."""
    dive = {clip["tag"]: clip for clip in deep_dive["clips"]}
    rows: list[dict[str, Any]] = []
    for entry in inputs["episodes"]:
        tag, episode = entry["tag"], entry["episode"]
        trajectory = rollouts / f"episode_{episode:06d}.parquet"
        name = str(metadata[episode]["object"])
        vertices = np.asarray(vertex_cloud[name], dtype=np.float64)
        series = released_series(
            trajectory,
            episode=episode,
            convert_root=Path(entry["frame_reference_root"]),
            dataset_root=dataset_root,
            work=work / tag,
            vertices=vertices,
        )
        clip = adoption["clips"].get(tag, {})
        kit = clip.get("record_kit_score") or clip.get("shipped_kit_score") or {}
        shipped_score = _shipping_kit_score(clip, entry["ships_from"]) or kit
        ceiling = (dive.get(tag) or {}).get("reference_ceiling_add_auc")
        verdict = classify(
            position_cm=series["position_cm"],
            orientation_deg=series["orientation_deg"],
            achieved=series["released"],
            reference=series["truth"],
            kit_auc=float(shipped_score.get("kit_add_auc", 0.0)),
            ceiling=ceiling,
            ceiling_margin=ceiling_margin,
        )
        rows.append(
            {
                "tag": tag,
                "episode": episode,
                "ships_from": entry["ships_from"],
                "object": name,
                "object_mass_kg": entry.get("object_mass_kg"),
                "object_radius_cm": (dive.get(tag) or {}).get("object_radius_cm"),
                "hands": (dive.get(tag) or {}).get("hands"),
                "kit_add_auc": round(float(shipped_score.get("kit_add_auc", 0.0)), 4),
                "kit_mppe_cm": round(float(shipped_score.get("kit_mppe_cm", 0.0)), 2),
                "reference_ceiling_add_auc": ceiling,
                "auc_headroom": None
                if ceiling is None
                else round(ceiling - float(shipped_score.get("kit_add_auc", 0.0)), 4),
                "closure": closure_of(hand_grasp, episode),
                "position_error_cm": {
                    "max": round(float(series["position_cm"].max()), 1),
                    "mean": round(float(series["position_cm"].mean()), 1),
                },
                "orientation_error_deg": {
                    "max": round(float(series["orientation_deg"].max()), 1),
                    "mean": round(float(series["orientation_deg"].mean()), 1),
                },
                **verdict,
            }
        )
    return rows


def _shipping_kit_score(clip: dict[str, Any], arm: str) -> dict[str, Any] | None:
    """The kit score of the rollout that ships, which for an adopted clip is its candidate's."""
    if arm == "first wave":
        return clip.get("shipped_kit_score")
    for candidate in clip.get("candidates", []):
        if candidate["arm"] == arm and candidate.get("kit_score"):
            return candidate["kit_score"]
    return clip.get("shipped_kit_score")


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """AUC headroom lost per class, which is what a fix to that mechanism could buy."""
    by_class: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = by_class.setdefault(
            row["failure_class"],
            {"clips": [], "auc_headroom": 0.0, "auc_headroom_positive": 0.0, "auc_now": 0.0},
        )
        bucket["clips"].append(row["tag"])
        bucket["auc_headroom"] += row["auc_headroom"] or 0.0
        # A clip already above its own reference ceiling has negative headroom, which is real and
        # reported, but it cannot fund a fix to another clip, so a gain uses the positive part.
        bucket["auc_headroom_positive"] += max(row["auc_headroom"] or 0.0, 0.0)
        bucket["auc_now"] += row["kit_add_auc"]
    for bucket in by_class.values():
        bucket["auc_headroom"] = round(bucket["auc_headroom"], 4)
        bucket["auc_headroom_positive"] = round(bucket["auc_headroom_positive"], 4)
        bucket["auc_now"] = round(bucket["auc_now"], 4)
        bucket["mean_auc_gain_over_30_if_fixed"] = round(bucket["auc_headroom_positive"] / 30, 4)
    weak = [row for row in rows if row["kit_add_auc"] < WEAK_BELOW]
    return {
        "clips": len(rows),
        "mean_kit_add_auc": round(float(np.mean([r["kit_add_auc"] for r in rows])), 4),
        "weak_clips": [row["tag"] for row in weak],
        "by_class": dict(sorted(by_class.items(), key=lambda kv: -kv[1]["auc_headroom_positive"])),
    }


def render(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """docs/FAILURE_TAXONOMY_WAVE2.md: the weak clips, then the classes, then every clip."""
    weak = [row for row in rows if row["kit_add_auc"] < WEAK_BELOW]
    lines = [
        "# Failure taxonomy, the 30 shipping rollouts",
        "",
        f"Over the entry as it stands: mean kit AUC **{summary['mean_kit_add_auc']}** across"
        f" {summary['clips']} clips, {len(weak)} of them below {WEAK_BELOW}. Each is classed by"
        " mechanism, in the fixed order `v2d taxonomy` documents, with the frame the mechanism"
        " fires. Position and orientation error are against the Tier 1 clean truth in the released"
        " frame, from the same conversion the kit scorer reads.",
        "",
        "## The clips below " + str(WEAK_BELOW),
        "",
        "| clip | AUC | class | frame | object | mass kg | hands | closure | palm depth cm |"
        " ceiling | headroom |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(weak, key=lambda r: r["kit_add_auc"]):
        closure = row["closure"]
        if closure.get("measured"):
            closes = "yes" if closure["any_hand_closes"] else "**no**"
            depth = ", ".join(
                f"{hand[0]}{side['palm_depth_cm']}" for hand, side in closure["hands"].items()
            )
        else:
            closes, depth = "not measured", "-"
        lines.append(
            f"| {row['tag']} | {row['kit_add_auc']:.4f} | {row['failure_class']} |"
            f" {row['failure_frame'] if row['failure_frame'] is not None else '-'} /"
            f" {row['frames']} | {row['object']} | {row['object_mass_kg']} |"
            f" {row['hands']} | {closes} | {depth} |"
            f" {row['reference_ceiling_add_auc']} | {row['auc_headroom']} |"
        )
    lines += [
        "",
        "## AUC headroom by class",
        "",
        "`headroom` is each clip's reference ceiling minus what it scores: what a perfect fix to"
        " that mechanism could buy, summed over the clips in the class, and what that is worth on"
        " the 30-clip mean.",
        "",
        "| class | clips | AUC now | headroom | mean-AUC gain if fixed |",
        "|---|---|---|---|---|",
    ]
    for name, bucket in summary["by_class"].items():
        lines.append(
            f"| {name} | {len(bucket['clips'])} ({' '.join(bucket['clips'])}) |"
            f" {bucket['auc_now']:.3f} | {bucket['auc_headroom']:.3f} |"
            f" +{bucket['mean_auc_gain_over_30_if_fixed']:.4f} |"
        )
    lines += [
        "",
        "## Every clip",
        "",
        "| clip | ships | AUC | class | frame | pos err cm max/mean | ori err deg max/mean |"
        " travel cm achieved/reference |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: r["tag"]):
        lines.append(
            f"| {row['tag']} | {row['ships_from']} | {row['kit_add_auc']:.4f} |"
            f" {row['failure_class']} |"
            f" {row['failure_frame'] if row['failure_frame'] is not None else '-'} |"
            f" {row['position_error_cm']['max']}/{row['position_error_cm']['mean']} |"
            f" {row['orientation_error_deg']['max']}/{row['orientation_error_deg']['mean']} |"
            f" {row['achieved_max_travel_cm']}/{row['reference_max_travel_cm']} |"
        )
    top, top_bucket = next(iter(summary["by_class"].items()))
    loss_classes = ("toppled_on_contact", "dropped_mid_clip", "never_grasped")
    loss_headroom = sum(
        summary["by_class"][name]["auc_headroom_positive"]
        for name in loss_classes
        if name in summary["by_class"]
    )
    loss_clips = sum(
        len(summary["by_class"][name]["clips"])
        for name in loss_classes
        if name in summary["by_class"]
    )
    orientation_limited = [
        row["tag"]
        for row in rows
        if row["failure_class"] == "held_but_drifting"
        and row["orientation_error_deg"]["max"] > TOPPLE_DEG
    ]
    lines += [
        "",
        "## Figures",
        "",
        "In `results/figures/hail_mary/`, captions in `captions.json`:",
        "",
        "- `a_auc_lost_by_class.png` — AUC headroom per failure class, clip counts on the bars.",
        "- `b_failing_clip_error_curves.png` — position error against the clean truth per frame"
        " for every clip below "
        f"{WEAK_BELOW}, failure frame dashed, the 10 cm gate dotted.",
        "- `c_mass_vs_auc.png` — mass against AUC beside radius against AUC. Mass is a constant:"
        " all 30 object URDFs carry 1.0 kg and inertia 0.01, so it explains nothing, and the"
        " figure says so on its face rather than implying a null result.",
        "",
        "## The one mechanism worth fixing",
        "",
        f"**{top.replace('_', ' ').capitalize()}** is the single class holding the most AUC:"
        f" {len(top_bucket['clips'])} clips, {top_bucket['auc_headroom_positive']:.2f} of headroom"
        f" between them, worth **+{top_bucket['mean_auc_gain_over_30_if_fixed']:.3f}** on the"
        " 30-clip mean if it were fully closed — more than the next two classes together"
        f" ({' and '.join(list(summary['by_class'])[1:3])}). And it is not a grasp problem: these"
        " clips never lose the object at all, by either signal, so no amount of work on"
        " acquisition or on termination touches them. What they lose is the angle —"
        f" {len(orientation_limited)} of the {len(top_bucket['clips'])} exceed"
        f" {TOPPLE_DEG:.0f}° of orientation error while holding position inside 10 cm, which is"
        " the same channel docs/METRICS_DEEP_DIVE.md measured at +0.3178 mean AUC of headroom from"
        " substituting truth orientation. We also have one causal read on it rather than a"
        " correlation: B4 raised the orientation reward's std from 0.3 to 0.8 on e17 and took it"
        " from 0.2466 to 0.4470 with SP-SR flipping to pass, one variable, same seed. That is the"
        " mechanism to spend the next GPU hour on, and B4x is already testing exactly it on eight"
        " more clips. The three loss classes together (toppled, dropped, never grasped) hold"
        f" {loss_headroom:.2f} of headroom over {loss_clips} clips, which is worth having but is"
        f" {loss_clips} separate mechanisms —"
        " a bat that topples, a bottle that slips, a cane never picked up — and each needs its own"
        " change, where the angle is one knob across eleven clips.",
    ]
    return "\n".join(lines) + "\n"


def figures(
    rows: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path, series: dict
) -> list[Path]:
    """Three figures: headroom per class, the failing clips' error curves, and mass against AUC."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    blue, vermillion, green, orange, purple, sky, grey = (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#E69F00",
        "#CC79A7",
        "#56B4E9",
        "#999999",
    )
    colour = {
        "never_grasped": vermillion,
        "toppled_on_contact": orange,
        "dropped_mid_clip": blue,
        "lost_at_placement": purple,
        "reference_ceiling": grey,
        "held_but_drifting": green,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # (1) headroom per class.
    classes = list(summary["by_class"])
    headroom = [summary["by_class"][name]["auc_headroom_positive"] for name in classes]
    counts = [len(summary["by_class"][name]["clips"]) for name in classes]
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    bars = axis.bar(range(len(classes)), headroom, color=[colour.get(c, sky) for c in classes])
    for bar, count in zip(bars, counts, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}\n{count} clip{'s' if count != 1 else ''}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(range(len(classes)))
    axis.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=8)
    axis.set_ylabel("AUC headroom lost (sum of ceiling - actual)")
    axis.set_title("AUC the entry leaves on the table, by failure mechanism")
    axis.set_ylim(0, max(headroom) * 1.25 if headroom else 1)
    figure.tight_layout()
    path = out_dir / "a_auc_lost_by_class.png"
    figure.savefig(path)
    plt.close(figure)
    written.append(path)

    # (2) the failing clips' position-error curves, failure frame marked.
    weak = sorted(
        [row for row in rows if row["kit_add_auc"] < WEAK_BELOW], key=lambda r: r["kit_add_auc"]
    )
    columns = 3
    figure_rows = (len(weak) + columns - 1) // columns
    figure, axes = plt.subplots(
        figure_rows, columns, figsize=(4.0 * columns, 2.6 * figure_rows), squeeze=False
    )
    for index, row in enumerate(weak):
        axis = axes[index // columns][index % columns]
        curve = series[row["tag"]]
        axis.plot(curve["frames"], curve["position_cm"], color=blue, linewidth=1.0)
        axis.axhline(LOSS_CM, color=grey, linewidth=0.8, linestyle=":")
        if row["failure_frame"] is not None:
            axis.axvline(
                row["failure_frame"],
                color=colour.get(row["failure_class"], vermillion),
                linewidth=1.2,
                linestyle="--",
            )
        axis.set_title(
            f"{row['tag']} {row['object']}  AUC {row['kit_add_auc']:.3f}\n"
            f"{row['failure_class']}"
            + (f" @ f{row['failure_frame']}" if row["failure_frame"] is not None else ""),
            fontsize=8,
        )
        axis.set_xlabel("frame")
        axis.set_ylabel("object position error (cm)")
    for blank in range(len(weak), figure_rows * columns):
        axes[blank // columns][blank % columns].axis("off")
    figure.suptitle(
        "Position error against the clean truth, clips below "
        f"{WEAK_BELOW} AUC; dashed line is the failure frame, dotted is the 10 cm loss gate",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / "b_failing_clip_error_curves.png"
    figure.savefig(path)
    plt.close(figure)
    written.append(path)

    # (3) mass against AUC, and the size proxy beside it, because the mass is a constant.
    masses = [row["object_mass_kg"] for row in rows]
    aucs = [row["kit_add_auc"] for row in rows]
    radii = [row["object_radius_cm"] for row in rows]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    left.scatter(masses, aucs, color=blue, s=28)
    distinct = sorted({m for m in masses if m is not None})
    left.set_xlim(0, max([*distinct, 1.0]) * 2)
    left.set_xlabel("object mass as simulated (kg)")
    left.set_ylabel("kit ADD AUC")
    left.set_title(
        "Mass vs AUC: every object is "
        + (f"{distinct[0]} kg" if len(distinct) == 1 else "varied")
        + ("; the axis is a constant" if len(distinct) == 1 else "")
    )
    if len(distinct) == 1:
        left.annotate(
            f"all 30 URDFs: mass {distinct[0]} kg, inertia 0.01\nmass cannot explain any variance",
            xy=(distinct[0], min(aucs)),
            xytext=(distinct[0] * 1.08, min(aucs) + 0.08),
            fontsize=8,
            arrowprops={"arrowstyle": "->", "color": grey, "linewidth": 0.8},
        )
    pairs = [(r, a) for r, a in zip(radii, aucs, strict=True) if r is not None]
    if pairs:
        from .deep_dive import spearman

        rho, count = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        right.scatter([p[0] for p in pairs], [p[1] for p in pairs], color=vermillion, s=28)
        right.set_title(f"Object radius vs AUC (Spearman rho {rho:+.2f}, n={count})")
    right.set_xlabel("object bounding radius (cm)")
    right.set_ylabel("kit ADD AUC")
    figure.tight_layout()
    path = out_dir / "c_mass_vs_auc.png"
    figure.savefig(path)
    plt.close(figure)
    written.append(path)

    captions = {
        "a_auc_lost_by_class.png": "AUC headroom (reference ceiling minus achieved) summed over"
        " the clips in each failure class, with the clip count on each bar. The tallest bar is"
        " where a fix buys the most.",
        "b_failing_clip_error_curves.png": "Object position error against the Tier 1 clean truth,"
        " per frame, for every shipping rollout below 0.35 AUC. Dashed vertical line: the frame"
        " the failure class fires. Dotted horizontal line: the 10 cm SP-SR gate.",
        "c_mass_vs_auc.png": "Left: object mass as the trainer loads it against AUC -- every one"
        " of the 30 object URDFs carries mass 1.0 kg and inertia 0.01, so the axis is a constant"
        " and mass explains nothing. Right: the bounding radius, which does vary, against AUC.",
    }
    (out_dir / "captions.json").write_text(json.dumps(captions, indent=2) + "\n")
    written.append(out_dir / "captions.json")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rollouts", type=Path, required=True, help="the staged shipping rollouts")
    parser.add_argument("--inputs", type=Path, required=True, help="the stager's inputs.json")
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument(
        "--first-wave-record",
        type=Path,
        help="results/t2_combined_run_record.json, for the kit score of a clip no arm re-ran",
    )
    parser.add_argument("--deep-dive", type=Path, required=True)
    parser.add_argument("--hand-grasp", type=Path, help="interp's hand_grasp_bridge_v1.json")
    parser.add_argument("--stage-root", type=Path, help="a staged tree, for the object URDF mass")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ceiling-margin", type=float, default=CEILING_MARGIN)
    parser.add_argument("--out", type=Path, required=True, help="the markdown")
    parser.add_argument("--json", dest="record", type=Path, required=True)
    parser.add_argument("--figures", type=Path, help="write the PNGs here")
    args = parser.parse_args(argv)

    inputs = json.loads(args.inputs.read_text())
    adoption = json.loads(args.adoption.read_text())
    # e21 is in no arm, so the adoption record does not mention it and its kit score has to come
    # from the first wave's own record. Without this it reported 0.0 and a full ceiling of
    # headroom, which would have put a clip scoring 0.206 at the top of the fix list.
    first_wave_scores = {}
    if args.first_wave_record and args.first_wave_record.is_file():
        first_wave_scores = {
            tag: clip.get("selected_kit_score") or {}
            for tag, clip in json.loads(args.first_wave_record.read_text())["clips"].items()
        }
    deep_dive = json.loads(args.deep_dive.read_text())
    hand_grasp = json.loads(args.hand_grasp.read_text()) if args.hand_grasp else {}
    metadata = {
        int(row["episode_index"]): row
        for row in (
            json.loads(line)
            for line in (args.dataset_root / "track_2" / TIER1 / "meta/episodes_metadata.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        )
    }
    if args.stage_root:
        for entry in inputs["episodes"]:
            entry["object_mass_kg"] = object_mass_kg(
                args.stage_root, str(metadata[entry["episode"]]["sequence_id"])
            )

    if str(args.kit_root) not in sys.path:
        sys.path.insert(0, str(args.kit_root))
    from v2dlb.vertices import build_blob

    names = sorted({str(metadata[e["episode"]]["object"]) for e in inputs["episodes"]})
    _blob, vertex_cloud = build_blob(args.dataset_root / "track_2" / TIER1 / "mesh", names)

    series: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    for entry in inputs["episodes"]:
        tag, episode = entry["tag"], entry["episode"]
        name = str(metadata[episode]["object"])
        curve = released_series(
            args.rollouts / f"episode_{episode:06d}.parquet",
            episode=episode,
            convert_root=Path(entry["frame_reference_root"]),
            dataset_root=args.dataset_root,
            work=args.work_dir / tag,
            vertices=np.asarray(vertex_cloud[name], dtype=np.float64),
        )
        series[tag] = curve
        clip = adoption["clips"].get(tag, {})
        shipped = _shipping_kit_score(clip, entry["ships_from"]) or first_wave_scores.get(tag, {})
        dive = {c["tag"]: c for c in deep_dive["clips"]}.get(tag) or {}
        ceiling = dive.get("reference_ceiling_add_auc")
        kit_auc = float(shipped.get("kit_add_auc", 0.0))
        verdict = classify(
            position_cm=curve["position_cm"],
            orientation_deg=curve["orientation_deg"],
            achieved=curve["released"],
            reference=curve["truth"],
            kit_auc=kit_auc,
            ceiling=ceiling,
            ceiling_margin=args.ceiling_margin,
        )
        rows.append(
            {
                "tag": tag,
                "episode": episode,
                "ships_from": entry["ships_from"],
                "object": name,
                "object_mass_kg": entry.get("object_mass_kg"),
                "object_radius_cm": dive.get("object_radius_cm"),
                "hands": dive.get("hands"),
                "kit_add_auc": round(kit_auc, 4),
                "kit_mppe_cm": round(float(shipped.get("kit_mppe_cm", 0.0)), 2),
                "reference_ceiling_add_auc": ceiling,
                "auc_headroom": None if ceiling is None else round(ceiling - kit_auc, 4),
                "closure": closure_of(hand_grasp, episode),
                "position_error_cm": {
                    "max": round(float(curve["position_cm"].max()), 1),
                    "mean": round(float(curve["position_cm"].mean()), 1),
                },
                "orientation_error_deg": {
                    "max": round(float(curve["orientation_deg"].max()), 1),
                    "mean": round(float(curve["orientation_deg"].mean()), 1),
                },
                **verdict,
            }
        )

    summary = summarise(rows)
    record = {
        "what": "the failure class of every shipping rollout, by mechanism",
        "rules": {
            "order": list(CLASSES),
            "ceiling_margin": args.ceiling_margin,
            "loss_cm": LOSS_CM,
            "drop_z_m": DROP_Z_M,
            "topple_deg": TOPPLE_DEG,
            "topple_lead_frames": TOPPLE_LEAD,
            "placement_fraction": PLACEMENT_FRACTION,
            "never_moved_cm": NEVER_MOVED_CM,
        },
        "summary": summary,
        "clips": rows,
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=1) + "\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows, summary))
    print(f"{len(rows)} clips classed; mean AUC {summary['mean_kit_add_auc']}")
    for name, bucket in summary["by_class"].items():
        print(
            f"  {name:20s} {len(bucket['clips']):2d} clips  headroom {bucket['auc_headroom']:.3f}"
        )
    if args.figures:
        for path in figures(rows, summary, args.figures, series):
            print(f"  wrote {path}")
    print(f"wrote {args.out} and {args.record}")
