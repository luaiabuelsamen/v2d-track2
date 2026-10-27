"""`v2d deep-dive-figures`: the figures behind docs/METRICS_DEEP_DIVE.md.

One idea per figure, every axis labelled with its unit, and the Okabe-Ito palette throughout so
the panels stay readable to a colourblind reader and in greyscale. Captions go to `captions.json`
beside the PNGs so the doc and the figures cannot drift apart.

The per-frame curves are recomputed through `deep_dive.released_series`, the same function the
metrics table uses, so a plotted curve and a tabulated number cannot disagree.

    v2d deep-dive-figures --record results/metrics_deep_dive.json --results RUN/results \\
        --dataset-root DATA --kit-root KIT --provenance results/t2_combined_provenance.json \\
        --work-dir SCRATCH --out-dir results/figures/deep_dive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..config import TIER1
from .deep_dive import (
    MANIPTRANS_ORIENTATION_DEG,
    MANIPTRANS_POSITION_M,
    PALM_MOVING_M_PER_S,
    SPIDER_ORIENTATION_RAD,
    SPIDER_POSITION_M,
    read_rollout,
    released_series,
    staged_hands,
    surface_distance,
)

#: Okabe-Ito, which stays distinguishable for every common form of colour blindness.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
GREY = "#999999"

#: The stock-recipe cells for the four comparison clips, and the root each was rolled out against.
#: Stock trained on the released Tier 2 input, the combined run on the bridge-cleaned tree, so each
#: must be converted against its own reference or the overlay compares two different frames.
STOCK_CELLS = {
    "e03": ("ab350/raw_e03", 99999744),
    "e09": ("ab350/raw_e09", 349999104),
    "e18": ("ab350_step1_broken/raw_e18", 349999104),
    "e24": ("full350", 299999232),
}
COMPARISON = ("e03", "e09", "e18", "e24")
#: The z-drop-selected stock kit AUCs on the record (docs/AB4_DESIGN.md); mean 0.36635 is the bar.
STOCK_KIT_AUC = {"e03": 0.3556, "e09": 0.2409, "e18": 0.3826, "e24": 0.4863}


def palette(figure_count: int = 0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    return plt


def save(figure, path: Path, captions: dict[str, str], caption: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    captions[path.name] = caption
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path


def weak_set(record: dict[str, Any]) -> set[str]:
    return {clip["tag"] for clip in record["clips"] if clip["bucket"] != "strong"}


def figure_a(plt, record, series, out: Path, captions) -> Path:
    """Per-clip error in time, 5x6, with the first loss marked."""
    rows = sorted(record["clips"], key=lambda c: c["tag"])
    figure, axes = plt.subplots(5, 6, figsize=(18, 11), sharex=False)
    weak = weak_set(record)
    for axis, clip in zip(axes.ravel(), rows, strict=True):
        tag = clip["tag"]
        data = series[tag]
        frame = np.arange(len(data["position_cm"]))
        axis.plot(frame, data["position_cm"], color=BLUE, lw=0.8, label="position (cm)")
        axis.set_ylim(0, max(25.0, float(np.percentile(data["position_cm"], 99)) * 1.1))
        twin = axis.twinx()
        twin.plot(frame, data["orientation_deg"], color=VERMILLION, lw=0.7, alpha=0.85,
                  label="orientation (deg)")
        twin.set_ylim(0, 185)
        twin.grid(False)
        twin.tick_params(labelsize=6)
        # The reference's own speed, faint: it says whether the clip was moving when it failed.
        speed = np.zeros(len(frame))
        reference = data["truth"][:, :3]
        speed[1:] = np.linalg.norm(np.diff(reference, axis=0), axis=1) * 30.0 * 100
        axis.fill_between(frame, 0, speed, color=GREY, alpha=0.18, lw=0, label="ref speed (cm/s)")
        if clip["first_loss_frame"] is not None:
            axis.axvline(clip["first_loss_frame"], color=GREEN, lw=1.0, ls="--")
        axis.axhline(10.0, color=GREY, lw=0.6, ls=":")
        colour = VERMILLION if tag in weak else "black"
        axis.set_title(f"{tag} {clip['object'][:16]}  AUC {clip['kit_add_auc']:.2f}",
                       color=colour)
        axis.tick_params(labelsize=6)
    for axis in axes[-1]:
        axis.set_xlabel("frame")
    for axis in axes[:, 0]:
        axis.set_ylabel("position error (cm)")
    handles = [
        plt.Line2D([], [], color=BLUE, label="object position error (cm, left)"),
        plt.Line2D([], [], color=VERMILLION, label="object orientation error (deg, right)"),
        plt.Line2D([], [], color=GREEN, ls="--", label="first loss (10 cm, 10 frames)"),
        plt.Line2D([], [], color=GREY, ls=":", label="10 cm gate"),
        plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.18, label="reference speed (cm/s)"),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.015))
    figure.suptitle("Object error against clean truth, per clip, selected checkpoint", y=1.035)
    figure.tight_layout()
    return save(
        figure, out / "a_per_clip_error_timeseries.png", captions,
        "Object position (blue, left axis) and orientation (vermillion, right axis) error against "
        "the clean Tier 1 truth over each clip's selected checkpoint; green dashed line is the "
        "first sustained loss past 10 cm, grey fill the reference's own speed. Weak clips titled "
        "in vermillion.",
    )


def figure_b(plt, record, series, stock, out: Path, captions) -> Path:
    """The four comparison clips, combined recipe against stock."""
    figure, axes = plt.subplots(2, 4, figsize=(15, 6))
    by_tag = {clip["tag"]: clip for clip in record["clips"]}
    for index, tag in enumerate(COMPARISON):
        for row, (key, unit, limit) in enumerate(
            (("position_cm", "position error (cm)", 10.0),
             ("orientation_deg", "orientation error (deg)", 30.0))
        ):
            axis = axes[row, index]
            combined = series[tag][key]
            axis.plot(np.arange(len(combined)), combined, color=BLUE, lw=0.9, label="combined")
            if tag in stock:
                raw = stock[tag][key]
                axis.plot(np.arange(len(raw)), raw, color=ORANGE, lw=0.9, alpha=0.9, label="stock")
            axis.axhline(limit, color=GREY, lw=0.6, ls=":")
            if by_tag[tag]["first_loss_frame"] is not None and row == 0:
                axis.axvline(by_tag[tag]["first_loss_frame"], color=GREEN, lw=1.0, ls="--")
            axis.set_ylabel(unit if index == 0 else "")
            axis.set_xlabel("frame" if row == 1 else "")
            if row == 0:
                axis.set_title(
                    f"{tag}  combined {by_tag[tag]['kit_add_auc']:.3f}"
                    + (f"  stock {stock[tag]['auc']:.3f}" if tag in stock else "")
                )
            if index == 0 and row == 0:
                axis.legend(loc="upper left", fontsize=7)
    figure.suptitle("Combined recipe against the stock recipe, the four clips with a stock record")
    figure.tight_layout()
    return save(
        figure, out / "b_comparison_clips_vs_stock.png", captions,
        "The four clips that have a stock-recipe record: combined recipe (blue) against stock "
        "(orange), position error above and orientation below, both against clean truth. Dotted "
        "lines are the 10 cm and 30 deg gates.",
    )


def figure_c(plt, record, out: Path, captions) -> Path:
    """Checkpoint curves and the variance they imply."""
    import matplotlib.pyplot as _plt  # noqa: F401

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    weak = weak_set(record)
    late_std = []
    for clip in sorted(record["clips"], key=lambda c: c["tag"]):
        curve = clip["checkpoint_curve"]
        steps = curve["steps_millions"]
        aucs = curve["kit_add_auc"]
        is_weak = clip["tag"] in weak
        left.plot(steps, aucs, color=VERMILLION if is_weak else SKY,
                  lw=1.4 if is_weak else 0.9, alpha=0.95 if is_weak else 0.55,
                  marker="o", ms=2.5, zorder=3 if is_weak else 2)
        selected = clip["selected_environment_steps"] // 1_000_000
        left.plot([selected], [clip["kit_add_auc"]], marker="*", ms=9,
                  color="black", zorder=4, lw=0)
        if len(aucs) >= 5:
            late_std.append(float(np.std(aucs[-5:])))
    left.set_xlabel("environment steps (millions)")
    left.set_ylabel("kit ADD-AUC")
    left.set_title("AUC against checkpoint, all 30 clips (weak in vermillion, "
                   "star = registered pick)")
    left.set_ylim(0, 1)

    box = right.boxplot([late_std], widths=0.5, patch_artist=True, showmeans=True)
    box["boxes"][0].set_facecolor(SKY)
    box["boxes"][0].set_alpha(0.6)
    right.scatter(np.random.default_rng(0).normal(1, 0.03, len(late_std)), late_std,
                  s=14, color=BLUE, alpha=0.7, zorder=3)
    right.set_xticks([1], ["30 clips"])
    right.set_ylabel("std of kit AUC over a clip's last 5 checkpoints")
    right.set_title(f"Within-run spread\nmean {np.mean(late_std):.4f}, max {max(late_std):.4f}")
    figure.tight_layout()
    return save(
        figure, out / "c_checkpoint_curves_and_variance.png", captions,
        "Left: kit AUC against training steps for every clip, weak clips highlighted, black star "
        "on the checkpoint the registered z-drop rule picked. Right: the standard deviation of "
        "AUC across each clip's last five checkpoints -- the noise floor any second-wave A/B has "
        "to beat.",
    )


def figure_d(plt, record, out: Path, captions) -> Path:
    """What each clip's AUC would be with one channel made perfect."""
    rows = sorted(record["clips"], key=lambda c: c["kit_add_auc"])
    tags = [clip["tag"] for clip in rows]
    actual = [clip["kit_add_auc"] for clip in rows]
    position = [clip["position_only"]["kit_add_auc"] for clip in rows]
    orientation = [clip["orientation_only"]["kit_add_auc"] for clip in rows]
    index = np.arange(len(tags))
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.bar(index - 0.27, actual, 0.27, color=GREY, label="as submitted")
    axis.bar(index, position, 0.27, color=BLUE,
             label="orientation replaced by truth (position-only loss)")
    axis.bar(index + 0.27, orientation, 0.27, color=VERMILLION,
             label="position replaced by truth (orientation-only loss)")
    axis.set_xticks(index, tags, rotation=90)
    axis.set_ylabel("kit ADD-AUC")
    axis.set_xlabel("clip, ordered by submitted AUC")
    axis.set_ylim(0, 1)
    axis.legend(loc="upper left")
    axis.set_title("AUC decomposition: which channel the score is lost to, per clip")
    figure.tight_layout()
    return save(
        figure, out / "d_auc_decomposition.png", captions,
        "Per clip: the submitted AUC (grey) against the AUC the same rollout earns with its "
        "orientation replaced by truth (blue, so only position error remains) and with its "
        "position replaced by truth (vermillion, only orientation remains). The taller the gap "
        "over grey, the more that channel is costing.",
    )


def figure_e(plt, record, series, hands, out: Path, captions) -> Path:
    """Pantomime: the hand does the motion, the object does not come along."""
    rows = [clip for clip in sorted(record["clips"], key=lambda c: -(
        (c["hand_object"] or {}).get("pantomime_fraction") or 0.0))
        if clip["hand_object"] and clip["hand_object"]["frames_reference_says_held"] > 0][:12]
    figure, axes = plt.subplots(3, 4, figsize=(16, 8))
    for axis, clip in zip(axes.ravel(), rows, strict=False):
        tag = clip["tag"]
        trace = hands.get(tag)
        if trace is None:
            axis.set_visible(False)
            continue
        frame = np.arange(len(trace["achieved_cm"]))
        axis.plot(frame, trace["achieved_cm"], color=VERMILLION, lw=0.9,
                  label="palm to achieved object")
        axis.plot(frame, trace["reference_cm"], color=BLUE, lw=0.9, alpha=0.85,
                  label="palm to reference object")
        axis.fill_between(frame, 0, np.where(trace["held"], axis.get_ylim()[1], 0),
                          color=GREEN, alpha=0.10, lw=0)
        twin = axis.twinx()
        twin.plot(frame, trace["speed_cm_s"], color=GREY, lw=0.6, alpha=0.7)
        twin.set_ylim(0, max(10.0, float(np.percentile(trace["speed_cm_s"], 99)) * 1.2))
        twin.grid(False)
        twin.tick_params(labelsize=6)
        axis.set_title(
            f"{tag} {clip['object'][:14]}  pantomime "
            f"{clip['hand_object']['pantomime_fraction']:.2f}",
            color=VERMILLION if clip["bucket"] != "strong" else "black",
        )
        axis.tick_params(labelsize=6)
        axis.set_ylabel("palm to object surface (cm)")
        axis.set_xlabel("frame")
    handles = [
        plt.Line2D([], [], color=VERMILLION, label="palm to ACHIEVED object surface (cm)"),
        plt.Line2D([], [], color=BLUE, label="palm to REFERENCE object surface (cm)"),
        plt.Line2D([], [], color=GREY, label="palm speed (cm/s, right axis)"),
        plt.Rectangle((0, 0), 1, 1, color=GREEN, alpha=0.10,
                      label="reference reports the object in hand"),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    figure.suptitle(
        "Pantomime: reference palm to object surface. Where vermillion leaves blue while the "
        "green band holds and the palm keeps moving, the hand is performing without the object.",
        y=1.075,
    )
    figure.tight_layout()
    return save(
        figure, out / "e_hand_to_object_distance.png", captions,
        "The twelve clips with the most pantomime. Distance from the reference palm to the nearest "
        "point of the object's AUC vertex cloud, for the achieved object (vermillion) and the "
        "reference object (blue); grey is palm speed, the green band is where the retargeter "
        "reports the object in hand. Reference palms, because the achieved body was never "
        "exported.",
    )


def figure_f(plt, record, out: Path, captions) -> Path:
    """What correlates with the score across clips, and what does not."""
    clips = record["clips"]
    auc = [clip["kit_add_auc"] for clip in clips]
    weak = weak_set(record)
    colours = [VERMILLION if clip["tag"] in weak else BLUE for clip in clips]
    panels = [
        ("object_radius_cm", [c["object_radius_cm"] for c in clips], "object radius (cm)"),
        ("first_loss_frame", [c["first_loss_frame"] for c in clips], "first loss (frame)"),
        ("bridge_anchor_segments", [c["bridge_anchor_segments"] for c in clips],
         "bridge anchor segments"),
        ("reference_ceiling_add_auc", [c["reference_ceiling_add_auc"] for c in clips],
         "reference ceiling (kit AUC)"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, (key, values, label) in zip(axes, panels, strict=True):
        pairs = [(v, a, c, t["tag"]) for v, a, c, t in zip(values, auc, colours, clips, strict=True)
                 if v is not None]
        axis.scatter([p[0] for p in pairs], [p[1] for p in pairs],
                     c=[p[2] for p in pairs], s=26, alpha=0.85)
        if key == "reference_ceiling_add_auc" and pairs:
            lo = min(min(p[0] for p in pairs), min(p[1] for p in pairs))
            hi = max(max(p[0] for p in pairs), max(p[1] for p in pairs))
            axis.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, ls="--", label="y = x")
            axis.legend(fontsize=7, loc="lower right")
        rho = record["correlations"]["pairs"][key]["spearman_rho"]
        n = record["correlations"]["pairs"][key]["n"]
        axis.set_xlabel(label)
        axis.set_ylabel("kit ADD-AUC")
        axis.set_title(f"rho = {rho}  (n = {n})")
    figure.suptitle("What the score tracks across the 30 clips (weak clips in vermillion)")
    figure.tight_layout()
    return save(
        figure, out / "f_correlations.png", captions,
        "Kit AUC against object radius, the frame of first loss, the bridge smoother's anchor "
        "count and the reference's own ceiling, with the Spearman rank correlation in each title. "
        "The dashed line in the last panel is y = x: points above it are clips that beat the "
        "reference they tracked.",
    )


def figure_g(plt, record, out: Path, captions) -> Path:
    """How far each clip is from the two success thresholds."""
    rows = sorted(record["clips"], key=lambda c: -c["success"]["spider_position_margin_cm"])
    tags = [clip["tag"] for clip in rows]
    index = np.arange(len(tags))
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for axis, (name, position_key, orientation_key, position_limit, orientation_limit) in zip(
        (top, bottom),
        (
            ("SP-SR", "spider_position_margin_cm", "spider_orientation_margin_deg",
             SPIDER_POSITION_M * 100, np.degrees(SPIDER_ORIENTATION_RAD)),
            ("MP-SR", "maniptrans_position_margin_cm", "maniptrans_orientation_margin_deg",
             MANIPTRANS_POSITION_M * 100, MANIPTRANS_ORIENTATION_DEG),
        ),
        strict=True,
    ):
        position = [clip["success"][position_key] for clip in rows]
        orientation = [clip["success"][orientation_key] for clip in rows]
        axis.bar(index - 0.21, position, 0.42, color=BLUE,
                 label=f"position margin (cm; threshold {position_limit:.0f} cm)")
        axis.bar(index + 0.21, orientation, 0.42, color=VERMILLION,
                 label=f"orientation margin (deg; threshold {orientation_limit:.0f} deg)")
        axis.axhline(0, color="black", lw=0.8)
        axis.set_ylabel(f"{name} margin\n(positive = passes)")
        axis.legend(loc="lower left", fontsize=7)
        passes = sum(1 for clip in rows if clip["success"][f"{name.split('-')[0].lower()}"
                                                           .replace("sp", "spider")
                                                           .replace("mp", "maniptrans")
                                                           + "_pass"])
        axis.set_title(f"{name}: {passes} of {len(rows)} clips pass "
                       f"(both margins must be positive)")
        axis.set_yscale("symlog", linthresh=5)
    bottom.set_xticks(index, tags, rotation=90)
    bottom.set_xlabel("clip, ordered by SP-SR position margin")
    figure.tight_layout()
    return save(
        figure, out / "g_threshold_margins.png", captions,
        "Distance from each success threshold, per clip, on a symmetric log scale. A clip passes "
        "only when both of its bars are positive. SP-SR above (10 cm / 0.5 rad), MP-SR below "
        "(3 cm / 30 deg); both thresholds apply to the time-averaged error.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True, help="results/metrics_deep_dive.json")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--stock-root", type=Path, help="holds the ab350/full350 stock cells")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    record = json.loads(args.record.read_text())
    provenance = json.loads(args.provenance.read_text())
    by_clip = provenance["ref_dataset_by_clip"]
    clips = provenance["clips"]
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
    if str(args.kit_root) not in sys.path:
        sys.path.insert(0, str(args.kit_root))
    from v2dlb.vertices import build_blob

    names = sorted({str(row["object"]) for row in metadata.values()})
    _blob, cloud = build_blob(args.dataset_root / "track_2" / TIER1 / "mesh", names)

    plt = palette()
    series: dict[str, dict[str, np.ndarray]] = {}
    hands: dict[str, dict[str, np.ndarray]] = {}
    for clip in record["clips"]:
        tag, episode = clip["tag"], clip["episode"]
        vertices = np.asarray(cloud[str(metadata[episode]["object"])], dtype=np.float64)
        series[tag] = released_series(
            args.results / tag / f"policy_{clip['selected_environment_steps']}/objects.parquet",
            episode=episode,
            convert_root=Path(by_clip[tag]) if tag in by_clip else args.dataset_root,
            dataset_root=args.dataset_root,
            work=args.work_dir / f"fig_{tag}",
            vertices=vertices,
        )
        staged = staged_hands(Path(clips[tag]["source_parquet"]))
        if staged is None:
            continue
        # Simulator frame, as in deep_dive.clip_metrics: the palms and the rollout's own object
        # tracks share a frame, the released-frame poses do not share it with them.
        achieved_sim, reference_sim = read_rollout(
            args.results / tag / f"policy_{clip['selected_environment_steps']}/objects.parquet"
        )
        usable = min(len(achieved_sim), staged["palms"].shape[0])
        achieved_sim, reference_sim = achieved_sim[:usable], reference_sim[:usable]
        palms = staged["palms"][:usable]
        contact = staged["contact"][:, :usable]
        achieved = np.stack(
            [surface_distance(palms[:, hand, :3], achieved_sim, vertices)
             for hand in range(palms.shape[1])], axis=0
        ).min(axis=0)
        reference = np.stack(
            [surface_distance(palms[:, hand, :3], reference_sim, vertices)
             for hand in range(palms.shape[1])], axis=0
        ).min(axis=0)
        speed = np.zeros(usable)
        speed[1:] = np.linalg.norm(np.diff(palms[:, :, :3], axis=0), axis=-1).max(axis=-1) * (
            staged["fps"] * 100
        )
        hands[tag] = {
            "achieved_cm": achieved * 100,
            "reference_cm": reference * 100,
            "speed_cm_s": speed,
            "held": contact.max(axis=0) > 0.5,
            "moving": speed > PALM_MOVING_M_PER_S * 100,
        }

    stock: dict[str, dict[str, Any]] = {}
    root = args.stock_root
    if root is not None:
        for tag, (relative, steps) in STOCK_CELLS.items():
            trajectory = root / relative / f"policy_{steps}/objects.parquet"
            if not trajectory.is_file():
                continue
            episode = int(tag[1:])
            # Stock trained on the RELEASED Tier 2 input, so it converts against the release.
            stock[tag] = released_series(
                trajectory,
                episode=episode,
                convert_root=args.dataset_root,
                dataset_root=args.dataset_root,
                work=args.work_dir / f"stock_{tag}",
                vertices=np.asarray(cloud[str(metadata[episode]["object"])], dtype=np.float64),
            )
            stock[tag]["auc"] = STOCK_KIT_AUC[tag]

    captions: dict[str, str] = {}
    written = [
        figure_a(plt, record, series, args.out_dir, captions),
        figure_c(plt, record, args.out_dir, captions),
        figure_d(plt, record, args.out_dir, captions),
        figure_e(plt, record, series, hands, args.out_dir, captions),
        figure_f(plt, record, args.out_dir, captions),
        figure_g(plt, record, args.out_dir, captions),
    ]
    if stock:
        written.insert(1, figure_b(plt, record, series, stock, args.out_dir, captions))
    else:
        print("no stock cells found; skipping figure b (pass --stock-root)")

    (args.out_dir / "captions.json").write_text(
        json.dumps(
            {
                "what": "one-line caption per figure behind docs/METRICS_DEEP_DIVE.md",
                "record": str(args.record),
                "palette": "Okabe-Ito; blue=position, vermillion=orientation or weak, "
                "green=event, grey=context",
                "dpi": 150,
                "captions": dict(sorted(captions.items())),
            },
            indent=2,
        )
        + "\n"
    )
    for path in written:
        print(f"{path}  {path.stat().st_size // 1024} KiB")
    print(f"{args.out_dir / 'captions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
