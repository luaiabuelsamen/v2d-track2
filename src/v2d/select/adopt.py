"""Which second-wave clips replace their first-wave rollout, decided on the input alone.

The second wave re-trains 28 clips, one variable per arm, against rollouts that are already on
the Kaggle boards. Deciding per clip which rollout to submit is a selection problem, and it has
the same trap `v2d pick-checkpoint` has: the kit score is Tier 1 truth, so choosing the better
score would be choosing on the answer and nothing about the resulting entry would be honest.

So the rule registered in docs/SECOND_WAVE.md reads the same thing `pick-checkpoint` reads and
nothing else: **take the new rollout only if its z-drop-selected checkpoint sits closer to the
noisy input reference than the old selected checkpoint does.** Ties go to the first wave. The kit
scores are computed and reported here, because the pre-registered bars are stated on them, but
they are not an input to the verdict -- `verdict_reads_ground_truth` is False and the test suite
pins it.

One caveat the rule needs, and it is the reason this is a command rather than a `sort -k2`: the
comparison assumes both rollouts tracked the same input. That holds for every arm except **B5**,
whose whole point is that e01's reference changed. For such a clip `--reference-from-old` scores
the new rollout against the reference the *shipped* entry used, so the two numbers are
commensurable and the new arm is not flattered by being measured against its own easier target.

Every other clip is checked rather than assumed: the two rollouts' reference columns must agree to
within `--tolerance`, and a clip whose references disagree without `--reference-from-old` is
reported `undecidable` instead of being quietly decided on two different yardsticks.

    v2d adopt-check --old $S/t2_combined_run/results \
      --group A=$S/second_wave/run_A/results --group B5=$S/second_wave/run_B5/results \
      --record results/t2_combined_run_record.json --reference-from-old e01 \
      -o results/second_wave_adoption.json --markdown results/second_wave_results.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

# Two rollouts count as tracking the same input if their reference columns agree to 1 cm. The
# references we expect to be identical are identical (a finger solve copies the object columns
# through untouched); the tolerance is here so that a float round-trip cannot be mistaken for a
# changed reference, not to wave through a different trajectory.
REFERENCE_TOLERANCE_M = 0.01

# The bars from docs/SECOND_WAVE.md, fixed before launch. Part A is a count over its 18 clips;
# Part B is per clip, and these are the first-wave kit AUCs rounded as they were registered.
PART_A_CLIPS = (
    "e00",
    "e02",
    "e03",
    "e06",
    "e08",
    "e10",
    "e11",
    "e12",
    "e15",
    "e16",
    "e18",
    "e20",
    "e22",
    "e23",
    "e24",
    "e26",
    "e27",
    "e29",
)
PART_A_MANIPTRANS_BAR = 9
PART_A_MANIPTRANS_BEFORE = 5
PART_A_MAX_AUC_REGRESSION = 0.05
PART_B_BARS = {
    "e01": 0.200,
    "e04": 0.147,
    "e05": 0.266,
    "e07": 0.251,
    "e09": 0.245,
    "e14": 0.109,
    "e17": 0.247,
    "e19": 0.116,
    "e25": 0.134,
    "e28": 0.089,
}


def load_trajectory(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    """One checkpoint's achieved and reference object positions, filtered as the selector filters.

    `environment_valid` is dropped the same way `v2d.select.checkpoints.read_checkpoint` drops it,
    because this command's distances have to be the same distances the registered rule chose on.
    """
    table = pq.read_table(checkpoint / "objects.parquet")
    columns = {name: np.asarray(table[name]) for name in table.schema.names}
    valid = columns.get("environment_valid")
    keep = np.ones(table.num_rows, dtype=bool) if valid is None else valid.astype(bool)
    achieved = np.stack([columns[f"achieved_position_{a}"][keep] for a in "xyz"], axis=1)
    reference = np.stack([columns[f"reference_position_{a}"][keep] for a in "xyz"], axis=1)
    return achieved, reference


def mean_distance_cm(achieved: np.ndarray, reference: np.ndarray) -> float:
    frames = min(len(achieved), len(reference))
    distance = np.linalg.norm(achieved[:frames] - reference[:frames], axis=1)
    return round(float(distance.mean()) * 100, 4)


def selected_checkpoint(results: Path, tag: str) -> dict[str, Any] | None:
    """What the registered rule picked for one clip, or None if it has not been picked yet."""
    selection = results / tag / "selection.json"
    if not selection.is_file():
        return None
    episode = json.loads(selection.read_text())["episodes"][tag]
    steps = episode["selected_environment_steps"]
    directory = results / tag / f"policy_{steps}"
    if not (directory / "objects.parquet").is_file():
        return None
    row = next(item for item in episode["checkpoints"] if item["environment_steps"] == steps)
    return {
        "cell": str(results / tag),
        "checkpoint_dir": str(directory),
        "selected_environment_steps": steps,
        "selected_checkpoint_sha256": episode["selected_checkpoint_sha256"],
        "checkpoints_evaluated": len(episode["checkpoints"]),
        "drop_free_checkpoints": len(episode["drop_free_checkpoints"]),
        "selection_reason": episode["selection_reason"],
        "mean_distance_to_own_reference_cm": row["mean_distance_to_input_reference_cm"],
    }


def kit_score(checkpoint_dir: Path) -> dict[str, float] | None:
    """The kit numbers for a checkpoint, for the record. Never an input to the verdict."""
    score = checkpoint_dir / "score.json"
    if not score.is_file():
        return None
    overall = json.loads(score.read_text())["overall"]
    return {
        "kit_add_auc": overall["add_auc"],
        "kit_spider_sr": overall["spider_sr"],
        "kit_maniptrans_sr": overall["maniptrans_sr"],
        "kit_mppe_cm": overall["mppe_cm"],
    }


def decide(
    new: dict[str, Any],
    old: dict[str, Any],
    *,
    reference_from_old: bool,
    tolerance_m: float = REFERENCE_TOLERANCE_M,
) -> dict[str, Any]:
    """The rule on one clip: closer to the input reference wins, and a tie keeps the board."""
    new_achieved, new_reference = load_trajectory(Path(new["checkpoint_dir"]))
    old_achieved, old_reference = load_trajectory(Path(old["checkpoint_dir"]))

    frames = min(len(new_reference), len(old_reference))
    gap_m = float(np.abs(new_reference[:frames] - old_reference[:frames]).max())
    same_input = gap_m <= tolerance_m and len(new_reference) == len(old_reference)

    old_distance = mean_distance_cm(old_achieved, old_reference)
    if reference_from_old:
        # The B5 case: measure the new rollout against the reference the shipped entry used.
        new_distance = mean_distance_cm(new_achieved, old_reference)
        reference_used = "the shipped rollout's input reference"
    else:
        new_distance = mean_distance_cm(new_achieved, new_reference)
        reference_used = "each rollout's own input reference, checked to be the same one"

    verdict = "undecidable"
    if same_input or reference_from_old:
        verdict = "adopt" if new_distance < old_distance else "keep"

    return {
        "reference_used": reference_used,
        "reference_from_old": reference_from_old,
        "references_agree": same_input,
        "reference_max_gap_cm": round(gap_m * 100, 4),
        "frames_new": len(new_reference),
        "frames_old": len(old_reference),
        "new_mean_distance_to_input_cm": new_distance,
        "old_mean_distance_to_input_cm": old_distance,
        "verdict": verdict,
        "verdict_reads_ground_truth": False,
    }


def compare(
    groups: dict[str, Path],
    old_results: Path,
    *,
    reference_from_old: set[str],
    record: dict[str, Any] | None = None,
    tolerance_m: float = REFERENCE_TOLERANCE_M,
) -> dict[str, Any]:
    clips: dict[str, Any] = {}
    waiting: dict[str, list[str]] = {}
    for arm, results in sorted(groups.items()):
        if not results.is_dir():
            waiting[arm] = []
            continue
        for cell in sorted(results.glob("e[0-9][0-9]")):
            tag = cell.name
            new = selected_checkpoint(results, tag)
            if new is None:
                waiting.setdefault(arm, []).append(tag)
                continue
            old = selected_checkpoint(old_results, tag)
            if old is None:
                raise SystemExit(
                    f"FATAL: {tag} has a second-wave rollout but no first-wave selection under "
                    f"{old_results}. The rule compares against what is on the board; with nothing "
                    "to compare against there is no decision to make."
                )
            entry: dict[str, Any] = {
                "arm": arm,
                "episode": int(tag[1:]),
                "new": new,
                "old": old,
                "decision": decide(
                    new,
                    old,
                    reference_from_old=tag in reference_from_old,
                    tolerance_m=tolerance_m,
                ),
                "new_kit_score": kit_score(Path(new["checkpoint_dir"])),
                "old_kit_score": kit_score(Path(old["checkpoint_dir"])),
            }
            if record is not None:
                clip = record["clips"].get(tag)
                entry["record_kit_score"] = clip["selected_kit_score"] if clip else None
            clips[tag] = entry

    return {
        "what": "per clip, whether the second wave's rollout replaces the one on the board",
        "rule": {
            "source": "docs/SECOND_WAVE.md, 'The rule for the final entry — no truth reads'",
            "decided_on": "mean distance from the z-drop-selected checkpoint to the input",
            "tie_break": "a tie keeps the first-wave rollout",
            "reference_tolerance_m": tolerance_m,
            "reference_from_old": sorted(reference_from_old),
            "ground_truth_read": False,
        },
        "summary": summarise(clips),
        "bars": bars(clips),
        "clips_awaiting_results": waiting,
        "clips": clips,
    }


def _auc(score: dict[str, float] | None) -> float | None:
    return None if score is None else score["kit_add_auc"]


def summarise(clips: dict[str, Any]) -> dict[str, Any]:
    scored = {tag: clip for tag, clip in clips.items() if clip["new_kit_score"]}
    adopted = [tag for tag, clip in clips.items() if clip["decision"]["verdict"] == "adopt"]
    summary: dict[str, Any] = {
        "clips_landed": len(clips),
        "clips_scored": len(scored),
        "clips_adopted": len(adopted),
        "adopted": sorted(adopted),
        "undecidable": sorted(
            tag for tag, clip in clips.items() if clip["decision"]["verdict"] == "undecidable"
        ),
    }
    gains = [
        _auc(clip["new_kit_score"]) - clip["record_kit_score"]["kit_add_auc"]
        for clip in scored.values()
        if clip.get("record_kit_score")
    ]
    if gains:
        summary["mean_kit_auc_change_on_landed_clips"] = round(sum(gains) / len(gains), 6)
        summary["clips_improved_on_kit_auc"] = sum(1 for gain in gains if gain > 0)
    return summary


def bars(clips: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered bars, reported as they stand. Reporting, not deciding."""
    part_a = {tag: clip for tag, clip in clips.items() if tag in PART_A_CLIPS}
    scored_a = {tag: clip for tag, clip in part_a.items() if clip["new_kit_score"]}
    passes = sum(1 for clip in scored_a.values() if clip["new_kit_score"]["kit_maniptrans_sr"] > 0)
    regressions = sorted(
        tag
        for tag, clip in scored_a.items()
        if clip.get("record_kit_score")
        and clip["new_kit_score"]["kit_add_auc"]
        < clip["record_kit_score"]["kit_add_auc"] - PART_A_MAX_AUC_REGRESSION
    )
    part_b = {}
    for tag, bar in sorted(PART_B_BARS.items()):
        clip = clips.get(tag)
        if clip is None or not clip["new_kit_score"]:
            part_b[tag] = {"bar": bar, "new_kit_add_auc": None, "beats_bar": None}
            continue
        auc = clip["new_kit_score"]["kit_add_auc"]
        part_b[tag] = {"bar": bar, "new_kit_add_auc": round(auc, 6), "beats_bar": bool(auc > bar)}
    return {
        "part_a": {
            "clips": len(PART_A_CLIPS),
            "clips_scored": len(scored_a),
            "maniptrans_passes_before": PART_A_MANIPTRANS_BEFORE,
            "maniptrans_passes_now": passes,
            "maniptrans_bar": PART_A_MANIPTRANS_BAR,
            "clips_below_record_by_more_than": PART_A_MAX_AUC_REGRESSION,
            "clips_regressed": regressions,
            "met": bool(passes >= PART_A_MANIPTRANS_BAR and not regressions)
            if len(scored_a) == len(PART_A_CLIPS)
            else None,
        },
        "part_b": part_b,
    }


def _cell(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render(record: dict[str, Any]) -> str:
    """The whole table, rewritten from the JSON each pass so the file is never half a run old."""
    summary, bar = record["summary"], record["bars"]
    lines = [
        "# Second wave, as it lands",
        "",
        "Written by `brev/second_wave_puller.sh` as each clip finishes, from",
        "`v2d adopt-check` (`results/second_wave_adoption.json`). The **adopt** column is the",
        "registered rule and reads only the noisy input reference; the kit columns are Tier 1",
        "truth and are reported, never used to decide.",
        "",
        f"{summary['clips_landed']} of 28 clips landed, {summary['clips_scored']} kit-scored, "
        f"{summary['clips_adopted']} adopted.",
        "",
        "| clip | arm | record AUC | new AUC | Δ AUC | MP-SR rec → new | input dist cm, old → new "
        "| adopt |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag, clip in sorted(record["clips"].items()):
        new, old = clip["new_kit_score"], clip.get("record_kit_score")
        decision = clip["decision"]
        delta = _cell(new["kit_add_auc"] - old["kit_add_auc"], 4) if new and old else "—"
        mp = (
            f"{'pass' if old and old['kit_maniptrans_sr'] > 0 else 'fail'} → "
            f"{'pass' if new and new['kit_maniptrans_sr'] > 0 else 'fail'}"
            if new
            else "—"
        )
        lines.append(
            f"| {tag} | {clip['arm']} | {_cell(_auc(old))} | {_cell(_auc(new))} | {delta} | {mp} "
            f"| {decision['old_mean_distance_to_input_cm']:.2f} → "
            f"{decision['new_mean_distance_to_input_cm']:.2f} "
            f"| **{decision['verdict']}** |"
        )
    part_a = bar["part_a"]
    lines += [
        "",
        "## The pre-registered bars",
        "",
        f"- **Part A** ({part_a['clips_scored']}/{part_a['clips']} scored): MP-SR passes on "
        f"**{part_a['maniptrans_passes_now']}**, bar is {part_a['maniptrans_bar']} "
        f"(was {part_a['maniptrans_passes_before']}); "
        + (
            f"clips more than {part_a['clips_below_record_by_more_than']} AUC below their record: "
            + (", ".join(part_a["clips_regressed"]) if part_a["clips_regressed"] else "none")
        ),
        "- **Part B**, per clip against its own record:",
    ]
    for tag, item in bar["part_b"].items():
        verdict = {True: "beats", False: "below", None: "waiting"}[item["beats_bar"]]
        lines.append(
            f"  - {tag}: bar {item['bar']:.3f}, new {_cell(item['new_kit_add_auc'], 4)} ({verdict})"
        )
    if record["clips_awaiting_results"]:
        waiting = "; ".join(
            f"{arm}: {' '.join(tags) if tags else 'nothing pulled yet'}"
            for arm, tags in sorted(record["clips_awaiting_results"].items())
        )
        lines += ["", f"Still training or not yet selected — {waiting}."]
    if summary["undecidable"]:
        lines += [
            "",
            "**Undecidable** (the two rollouts tracked different references and the clip was not "
            "registered for the shipped-reference comparison): " + " ".join(summary["undecidable"]),
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="ARM=RESULTS_DIR",
        help="a second-wave arm and its results dir, repeatable",
    )
    parser.add_argument(
        "--old", type=Path, required=True, metavar="RESULTS_DIR", help="the first-wave results dir"
    )
    parser.add_argument(
        "--record", type=Path, help="results/t2_combined_run_record.json, for the record column"
    )
    parser.add_argument(
        "--reference-from-old",
        action="append",
        default=[],
        metavar="eXX",
        help="a clip whose reference changed: score the new rollout against the shipped reference",
    )
    parser.add_argument("--tolerance", type=float, default=REFERENCE_TOLERANCE_M, metavar="M")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, help="also write the human table here")
    args = parser.parse_args(argv)

    groups: dict[str, Path] = {}
    for item in args.group:
        if "=" not in item:
            raise SystemExit(f"--group wants ARM=RESULTS_DIR, got {item!r}")
        arm, _, directory = item.partition("=")
        groups[arm] = Path(directory)
    if not args.old.is_dir():
        raise SystemExit(f"FATAL: no first-wave results dir at {args.old}")

    record = None
    if args.record:
        record = json.loads(args.record.read_text())
        for tag, bar in PART_B_BARS.items():
            clip = record["clips"].get(tag)
            if clip and abs(clip["selected_kit_score"]["kit_add_auc"] - bar) > 0.0005:
                raise SystemExit(
                    f"FATAL: {args.record} has {tag} at "
                    f"{clip['selected_kit_score']['kit_add_auc']:.4f}, but the bar registered in "
                    f"docs/SECOND_WAVE.md is {bar}. The bars were fixed before launch; a record "
                    "that disagrees with them is a different run."
                )

    result = compare(
        groups,
        args.old,
        reference_from_old=set(args.reference_from_old),
        record=record,
        tolerance_m=args.tolerance,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render(result))

    for tag, clip in sorted(result["clips"].items()):
        decision = clip["decision"]
        new = clip["new_kit_score"]
        old = clip.get("record_kit_score")
        print(
            f"{tag} [{clip['arm']}] "
            f"record {_cell(_auc(old))} -> new {_cell(_auc(new))}  "
            f"input {decision['old_mean_distance_to_input_cm']:.2f} -> "
            f"{decision['new_mean_distance_to_input_cm']:.2f} cm  {decision['verdict']}"
        )
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {args.out}" + (f" and {args.markdown}" if args.markdown else ""))


if __name__ == "__main__":
    main()
