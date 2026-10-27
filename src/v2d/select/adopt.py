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

**More than one candidate per clip.** B4x re-runs six clips that Part A also re-ran and one that
B3 re-ran, so a clip can have two new rollouts. They are decided as a chain, in the order the
`--group` flags are given: the incumbent starts as the shipped rollout, each candidate is judged
against the incumbent *of that moment*, and a candidate that wins becomes the incumbent for the
next one. The alternative -- judging every candidate against the shipped rollout -- would let a
clip "adopt" twice and leave which one ships undefined. The earlier code keyed clips by tag alone,
so a second candidate silently replaced the first in the record; that is the bug this chain fixes.

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

# The follow-up arm, which is the one that can share a clip with another arm.
B4X_ARM = "B4x"

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
    """Decide every clip, chaining candidates in the order their groups were given."""
    candidates: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    waiting: dict[str, list[str]] = {}
    for arm, results in groups.items():
        if not results.is_dir():
            waiting[arm] = []
            continue
        for cell in sorted(results.glob("e[0-9][0-9]")):
            tag = cell.name
            found = selected_checkpoint(results, tag)
            if found is None:
                waiting.setdefault(arm, []).append(tag)
            else:
                candidates.setdefault(tag, []).append((arm, found))

    clips: dict[str, Any] = {}
    for tag, found in sorted(candidates.items()):
        shipped = selected_checkpoint(old_results, tag)
        if shipped is None:
            raise SystemExit(
                f"FATAL: {tag} has a second-wave rollout but no first-wave selection under "
                f"{old_results}. The rule compares against what is on the board; with nothing "
                "to compare against there is no decision to make."
            )
        entry: dict[str, Any] = {
            "episode": int(tag[1:]),
            "shipped": shipped,
            "shipped_kit_score": kit_score(Path(shipped["checkpoint_dir"])),
            "candidates": [],
        }
        if record is not None:
            clip = record["clips"].get(tag)
            entry["record_kit_score"] = clip["selected_kit_score"] if clip else None

        # The chain. Each candidate is judged against the incumbent of that moment, so a clip
        # that two arms re-ran cannot "adopt" twice and leave the winner undefined.
        incumbent_arm, incumbent = "first wave", shipped
        for arm, new in found:
            decision = decide(
                new,
                incumbent,
                reference_from_old=tag in reference_from_old,
                tolerance_m=tolerance_m,
            )
            decision["judged_against"] = incumbent_arm
            won = decision["verdict"] == "adopt"
            entry["candidates"].append(
                {
                    "arm": arm,
                    "cell": new,
                    "kit_score": kit_score(Path(new["checkpoint_dir"])),
                    "decision": decision,
                    "became_incumbent": won,
                }
            )
            if won:
                incumbent_arm, incumbent = arm, new
        entry["ships"] = {
            "arm": incumbent_arm,
            "selected_environment_steps": incumbent["selected_environment_steps"],
            "selected_checkpoint_sha256": incumbent["selected_checkpoint_sha256"],
            "checkpoint_dir": incumbent["checkpoint_dir"],
        }
        clips[tag] = entry

    return {
        "what": "per clip, which rollout ships: the board's, or an arm that beat it on the input",
        "rule": {
            "source": "docs/SECOND_WAVE.md, 'The rule for the final entry — no truth reads'",
            "decided_on": "mean distance from the z-drop-selected checkpoint to the input",
            "tie_break": "a tie keeps the incumbent",
            "chain": "candidates are judged in --group order against the incumbent of that moment",
            "group_order": list(groups),
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
    scored = [c for clip in clips.values() for c in clip["candidates"] if c["kit_score"]]
    replaced = {
        tag: clip["ships"]["arm"]
        for tag, clip in clips.items()
        if clip["ships"]["arm"] != "first wave"
    }
    by_arm: dict[str, int] = {}
    for arm in replaced.values():
        by_arm[arm] = by_arm.get(arm, 0) + 1
    summary: dict[str, Any] = {
        "clips_landed": len(clips),
        "candidates_landed": sum(len(clip["candidates"]) for clip in clips.values()),
        "candidates_scored": len(scored),
        "clips_replaced": len(replaced),
        "ships_by_arm": dict(sorted(by_arm.items())),
        "replaced": dict(sorted(replaced.items())),
        "undecidable": sorted(
            tag
            for tag, clip in clips.items()
            for c in clip["candidates"]
            if c["decision"]["verdict"] == "undecidable"
        ),
    }
    gains = [
        c["kit_score"]["kit_add_auc"] - clip["record_kit_score"]["kit_add_auc"]
        for clip in clips.values()
        for c in clip["candidates"]
        if c["kit_score"] and clip.get("record_kit_score")
    ]
    if gains:
        summary["mean_kit_auc_change_per_candidate"] = round(sum(gains) / len(gains), 6)
        summary["candidates_improving_on_kit_auc"] = sum(1 for gain in gains if gain > 0)
    # What the entry would score if it were repacked now: the shipping rollout of every clip we
    # hold, which is the number that actually matters.
    held = [(clip["ships"], clip) for clip in clips.values()]
    shipping_aucs = []
    for ships, clip in held:
        if ships["arm"] == "first wave":
            score = clip.get("record_kit_score") or clip["shipped_kit_score"]
        else:
            score = next(c["kit_score"] for c in clip["candidates"] if c["arm"] == ships["arm"])
        if score:
            shipping_aucs.append(score["kit_add_auc"])
    if shipping_aucs:
        summary["mean_kit_auc_of_shipping_rollouts_on_landed_clips"] = round(
            sum(shipping_aucs) / len(shipping_aucs), 6
        )
    return summary


def candidate_of(clip: dict[str, Any], arm: str) -> dict[str, Any] | None:
    """One clip's candidate from a named arm, or None if that arm has not landed it."""
    for candidate in clip["candidates"]:
        if candidate["arm"] == arm:
            return candidate
    return None


def best_kit_auc_held(clip: dict[str, Any], *, excluding: str) -> tuple[float | None, str]:
    """The best kit AUC we hold for a clip, and where it came from, ignoring one arm.

    This is B4x's bar, and it is the honest one: six of its clips are Part A clips, and Part A has
    already lifted e08 from 0.4431 to 0.6654. A B4x arm judged against the first-wave *record*
    could clear its bar at 0.50 and still be worse than the rollout we hold.
    """
    best, source = None, "nothing scored yet"
    record = clip.get("record_kit_score")
    if record:
        best, source = record["kit_add_auc"], "first wave"
    for candidate in clip["candidates"]:
        if candidate["arm"] == excluding or not candidate["kit_score"]:
            continue
        auc = candidate["kit_score"]["kit_add_auc"]
        if best is None or auc > best:
            best, source = auc, candidate["arm"]
    return best, source


def bars(clips: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered bars, reported as they stand. Reporting, not deciding."""
    part_a: dict[str, Any] = {}
    for tag in PART_A_CLIPS:
        clip = clips.get(tag)
        candidate = candidate_of(clip, "A") if clip else None
        if candidate and candidate["kit_score"]:
            part_a[tag] = (clip, candidate)
    passes = sum(
        1 for _, candidate in part_a.values() if candidate["kit_score"]["kit_maniptrans_sr"] > 0
    )
    regressions = sorted(
        tag
        for tag, (clip, candidate) in part_a.items()
        if clip.get("record_kit_score")
        and candidate["kit_score"]["kit_add_auc"]
        < clip["record_kit_score"]["kit_add_auc"] - PART_A_MAX_AUC_REGRESSION
    )
    part_b = {}
    for tag, bar in sorted(PART_B_BARS.items()):
        clip = clips.get(tag)
        # The registered wave's own arm for this clip, never B4x -- B4x has its own bar below.
        candidate = None
        if clip:
            for item in clip["candidates"]:
                if item["arm"] != B4X_ARM:
                    candidate = item
                    break
        if candidate is None or not candidate["kit_score"]:
            part_b[tag] = {"bar": bar, "new_kit_add_auc": None, "beats_bar": None}
            continue
        auc = candidate["kit_score"]["kit_add_auc"]
        part_b[tag] = {
            "arm": candidate["arm"],
            "bar": bar,
            "new_kit_add_auc": round(auc, 6),
            "beats_bar": bool(auc > bar),
        }
    # B4x: the bar is the best rollout held for that clip when it is judged, not the record.
    part_b4x = {}
    for tag, clip in sorted(clips.items()):
        candidate = candidate_of(clip, B4X_ARM)
        if candidate is None:
            continue
        bar, source = best_kit_auc_held(clip, excluding=B4X_ARM)
        auc = candidate["kit_score"]["kit_add_auc"] if candidate["kit_score"] else None
        part_b4x[tag] = {
            "bar": None if bar is None else round(bar, 6),
            "bar_from": source,
            "new_kit_add_auc": None if auc is None else round(auc, 6),
            "beats_bar": None if (auc is None or bar is None) else bool(auc > bar),
        }
    return {
        "part_a": {
            "clips": len(PART_A_CLIPS),
            "clips_scored": len(part_a),
            "maniptrans_passes_before": PART_A_MANIPTRANS_BEFORE,
            "maniptrans_passes_now": passes,
            "maniptrans_bar": PART_A_MANIPTRANS_BAR,
            "clips_below_record_by_more_than": PART_A_MAX_AUC_REGRESSION,
            "clips_regressed": regressions,
            "met": bool(passes >= PART_A_MANIPTRANS_BAR and not regressions)
            if len(part_a) == len(PART_A_CLIPS)
            else None,
        },
        "part_b": part_b,
        "part_b4x": part_b4x,
    }


def _cell(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def pending_arms(record: dict[str, Any]) -> dict[str, list[str]]:
    """Per clip, the arms that still owe it a rollout.

    The chain is recomputed from scratch every pass, so a clip whose B4x rollout lands before its
    Part A rollout is judged against the board and can read "B4x ships" -- and then flip when Part
    A arrives and B4x is re-judged against it. That is the rule working, not a bug, but a reader
    cannot tell a settled verdict from a provisional one without this.
    """
    waiting: dict[str, list[str]] = {}
    for arm, tags in record["clips_awaiting_results"].items():
        for tag in tags:
            waiting.setdefault(tag, []).append(arm)
    return {tag: sorted(arms) for tag, arms in waiting.items()}


def render(record: dict[str, Any]) -> str:
    """The whole table, rewritten from the JSON each pass so the file is never half a run old."""
    summary, bar = record["summary"], record["bars"]
    pending = pending_arms(record)
    expected = 36  # 28 in the registered wave plus B4x's 8
    lines = [
        "# Second wave, as it lands",
        "",
        "Written by `brev/second_wave_puller.sh` as each clip finishes, from",
        "`v2d adopt-check` (`results/second_wave_adoption.json`). The **verdict** column is the",
        "registered rule and reads only the noisy input reference; the kit columns are Tier 1",
        "truth and are reported, never used to decide. A clip two arms re-ran has one row per arm,",
        "each judged against the incumbent at that moment, and **ships** marks the winner.",
        "",
        f"{summary['clips_landed']} clips landed of {expected} expected, "
        f"{summary['candidates_landed']} candidate rollouts, "
        f"{summary['clips_replaced']} clips replaced "
        f"({', '.join(f'{arm} {n}' for arm, n in summary['ships_by_arm'].items()) or 'none yet'}).",
        "",
        "| clip | arm | record AUC | new AUC | Δ AUC | MP-SR | input dist cm, incumbent → new "
        "| verdict | ships |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tag, clip in sorted(record["clips"].items()):
        old_score = clip.get("record_kit_score")
        for candidate in clip["candidates"]:
            new_score, decision = candidate["kit_score"], candidate["decision"]
            delta = (
                _cell(new_score["kit_add_auc"] - old_score["kit_add_auc"], 4)
                if new_score and old_score
                else "—"
            )
            mp = ("pass" if new_score["kit_maniptrans_sr"] > 0 else "fail") if new_score else "—"
            ships = "**ships**" if clip["ships"]["arm"] == candidate["arm"] else ""
            if ships and tag in pending:
                # Provisional: another arm still owes this clip a rollout, and it is judged
                # against whatever is winning when it lands.
                ships = f"ships, pending {' '.join(pending[tag])}"
            lines.append(
                f"| {tag} | {candidate['arm']} | {_cell(_auc(old_score))} "
                f"| {_cell(_auc(new_score))} | {delta} | {mp} "
                f"| {decision['old_mean_distance_to_input_cm']:.2f} → "
                f"{decision['new_mean_distance_to_input_cm']:.2f} "
                f"| {decision['verdict']} | {ships} |"
            )
        if clip["ships"]["arm"] == "first wave":
            mark = "**ships (board)**"
            if tag in pending:
                mark = f"ships (board), pending {' '.join(pending[tag])}"
            lines.append(
                f"| {tag} | — | {_cell(_auc(old_score))} | — | — | — | — | — | {mark} |"
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
        arm = f" [{item['arm']}]" if item.get("arm") else ""
        lines.append(
            f"  - {tag}{arm}: bar {item['bar']:.3f}, new "
            f"{_cell(item['new_kit_add_auc'], 4)} ({verdict})"
        )
    if bar["part_b4x"]:
        lines += [
            "",
            "- **B4x**, per clip against the best rollout held when it was judged — not the "
            "first-wave record, because six of its clips are Part A clips:",
        ]
        for tag, item in bar["part_b4x"].items():
            verdict = {True: "beats", False: "below", None: "waiting"}[item["beats_bar"]]
            lines.append(
                f"  - {tag}: bar {_cell(item['bar'], 4)} (from {item['bar_from']}), new "
                f"{_cell(item['new_kit_add_auc'], 4)} ({verdict})"
            )
    if "mean_kit_auc_of_shipping_rollouts_on_landed_clips" in summary:
        lines += [
            "",
            "Mean kit AUC of the shipping rollout over the clips landed so far: "
            f"**{summary['mean_kit_auc_of_shipping_rollouts_on_landed_clips']:.4f}** "
            "(not comparable to the 30-clip 0.3908 until every clip is in).",
        ]
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

    # Insertion order IS the chain order, so the flags are read in the order they were given.
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
        old = clip.get("record_kit_score")
        for candidate in clip["candidates"]:
            decision = candidate["decision"]
            print(
                f"{tag} [{candidate['arm']}] "
                f"record {_cell(_auc(old))} -> new {_cell(_auc(candidate['kit_score']))}  "
                f"input {decision['old_mean_distance_to_input_cm']:.2f} -> "
                f"{decision['new_mean_distance_to_input_cm']:.2f} cm  {decision['verdict']}"
                f" (vs {decision['judged_against']})"
            )
        print(f"{tag} ships: {clip['ships']['arm']}")
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {args.out}" + (f" and {args.markdown}" if args.markdown else ""))


if __name__ == "__main__":
    main()
