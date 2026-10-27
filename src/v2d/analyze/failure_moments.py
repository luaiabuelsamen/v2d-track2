"""`v2d failure-moments`: what physically happens in the 30 frames before a clip is lost.

The object-physics sweep found that across the 30 clips, AUC falls with the object's peak speed
(Spearman -0.49), lift height (-0.44), peak acceleration (-0.40) and peak spin (-0.36), and has no
relationship with size or volume (+0.01) -- because `soma_to_g1.py` writes every object URDF as
1.0 kg with inertia diag 0.01, so nothing in the simulation is heavy. But only four clips fail at
a fast part of their clip. The rest fail while the object is nearly still, and a correlation with
speed cannot explain those, so this looks at the window itself.

Everything here is measured, and the measurements come from two different places, which is the one
thing to keep straight while reading it:

- The **achieved** object track is the policy's own, from the rollout the entry ships.
- The **hands and body are the reference's**, from the staged motion the policy was tracking.
  FlashCHORD's rollout export carries the object only, so there is no record of where the policy's
  own hands were. A hand that opens here is the hand the policy was *told* to open; whether its
  own hand followed is not in the data and is never claimed.

With that split, the window answers five questions per clip: does the reference hand open, does
the object slide against the palm it is held by, does it reach the support surface, does contact
move from one hand to the other, and is the body moving.

    v2d failure-moments --clips e05=602,e09=852,e17=399,e19=133,e28=118 \\
      --rollouts $S/t2_entry_wave2_dry/rollouts --motion-root ~/projects/v2d-hand/out/\\
finger_refs_bridge_v1/tier2/whole_body/soma --dataset-root $S/data \\
      --physics /tmp/obj_physics.csv --out docs/FAILURE_MOMENTS.md \\
      --json results/failure_moments.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ..config import TIER1
from ..dataset import episodes as dataset_episodes
from .deep_dive import read_rollout, surface_distance

#: The window the question is asked over: one second at 30 fps, before and including the frame.
WINDOW = 30
#: Finger joints this far from their open pose count as closed; the Dex3 closes to about 1.2 rad.
CLOSED_RAD = 0.3
#: A palm within this of the object surface is holding it rather than near it.
HOLDING_CM = 8.0
#: Movement under this is "still" at 30 fps.
STILL_M_PER_S = 0.05
#: The policy's object moving this many times faster than the reference asks is being flung.
FLUNG_RATIO = 3.0
#: ...and fast enough that the ratio is not two small numbers divided.
FLUNG_M_PER_S = 0.3


def motion_for(motion_root: Path, sequence_id: str) -> Path:
    """The staged reference parquet for one sequence."""
    found = sorted(motion_root.glob(f"sequence_id={sequence_id}*/robot_name=g1/data.parquet"))
    if not found:
        raise SystemExit(f"no reference motion for {sequence_id} under {motion_root}")
    return found[0]


#: The Dex3 finger joints, by name, as the retargeter writes them into `robot_joint_names`.
FINGER_JOINT = "_hand_"


def reference_state(parquet: Path) -> dict[str, np.ndarray]:
    """The hands, body and object of the motion the policy tracked, as per-frame arrays.

    Three shapes have to be taken from the file rather than assumed, and one column is empty:
    `hand_contact_active` is `(hand, frame)` and is transposed here; `ee_pose_w` is
    `(frame, hand, 7)` and is the only palm track in the tree, because `hand_frames_w` and
    `hand_finger_joints` ship empty; and the finger angles have to come out of
    `robot_joint_positions` by name, which is also the only place the policy's target closure is
    written down.
    """
    table = pq.read_table(parquet)
    row = {name: table[name][0].as_py() for name in table.schema.names}
    joint_names = [str(n) for n in row["robot_joint_names"]]
    joints = np.asarray(row["robot_joint_positions"], dtype=np.float64)
    sides = [str(s) for s in (row.get("hand_sides") or [])]
    finger_columns = {
        side: [i for i, name in enumerate(joint_names) if name.startswith(f"{side}{FINGER_JOINT}")]
        for side in sides
    }
    contact = np.asarray(row["hand_contact_active"], dtype=np.float64)
    return {
        "object_position": np.asarray(row["object_root_position"], dtype=np.float64),
        "body_position": np.asarray(row["robot_root_position"], dtype=np.float64),
        # (hand, frame) in the file; everything downstream wants (frame, hand).
        "contact_active": contact.T if contact.ndim == 2 else contact.reshape(-1, 1),
        "palms": np.asarray(row["ee_pose_w"], dtype=np.float64)[..., :3],
        "fingers": np.stack(
            [joints[:, finger_columns[side]].mean(axis=1) for side in sides], axis=1
        )
        if all(finger_columns.get(side) for side in sides)
        else np.zeros((len(joints), max(len(sides), 1))),
        "hand_sides": sides,
    }


def speed(track: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Per-frame speed in m/s, padded to the track's length."""
    if len(track) < 2:
        return np.zeros(len(track))
    step = np.linalg.norm(np.diff(track, axis=0), axis=1) * fps
    return np.concatenate([step[:1], step])


def palm_positions(state: dict[str, np.ndarray]) -> np.ndarray:
    """``(T, hands, 3)`` palm positions, from `ee_pose_w`."""
    array = np.asarray(state["palms"], dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise SystemExit(f"unexpected palm shape {array.shape}")
    return array


def window_of(frames: int, failure_frame: int, window: int = WINDOW) -> tuple[int, int]:
    """``(start, stop)`` of the window ending at the failure frame, clipped to the clip."""
    stop = min(failure_frame + 1, frames)
    return max(0, stop - window - 1), stop


def describe(
    *,
    tag: str,
    failure_frame: int,
    achieved: np.ndarray,
    reference_object: np.ndarray,
    state: dict[str, np.ndarray],
    vertices: np.ndarray,
    support_height_m: float | None,
    window: int = WINDOW,
) -> dict[str, Any]:
    """The measured facts of one clip's window, with no interpretation attached."""
    frames = min(len(achieved), len(state["object_position"]))
    start, stop = window_of(frames, failure_frame, window)
    span = slice(start, stop)

    palms = palm_positions(state)[:frames]
    fingers = state["fingers"][:frames]
    contact = state["contact_active"][:frames]
    body = state["body_position"][:frames]

    # Which palm is holding the object, by distance at the start of the window -- and that has to
    # be measured against the REFERENCE object. Measuring it against the achieved one picks the
    # nearest palm to wherever the policy's object has ended up, which on a clip that was already
    # lost is a hand on the other side of the room, and then every other number is about that hand.
    # Distance to the object's SURFACE, not to its origin: a cane's origin is 58 cm from the hand
    # that is holding its handle, and measuring to the origin called every one of these clips
    # "no palm on the object".
    to_palm_reference = np.stack(
        [
            surface_distance(palms[:frames, hand], reference_object[:frames], vertices) * 100
            for hand in range(palms.shape[1])
        ],
        axis=1,
    )
    to_palm = np.stack(
        [
            surface_distance(palms[:frames, hand], achieved[:frames], vertices) * 100
            for hand in range(palms.shape[1])
        ],
        axis=1,
    )
    holding = int(np.argmin(to_palm_reference[start])) if to_palm.shape[1] else 0
    sides = state["hand_sides"] or [str(i) for i in range(to_palm.shape[1])]
    off_reference = (
        np.linalg.norm(achieved[:frames, :3] - reference_object[:frames, :3], axis=1) * 100
    )

    closed = np.abs(fingers[:, holding]) if fingers.size and fingers.shape[1] > holding else None
    contact_per_hand = contact[:frames] if contact.size else np.zeros((frames, 0))

    record: dict[str, Any] = {
        "tag": tag,
        "failure_frame": failure_frame,
        "window": [start, stop - 1],
        "object_speed_m_s": {
            "in_window_max": round(float(speed(achieved[:frames, :3])[span].max()), 3),
            "clip_max": round(float(speed(achieved[:frames, :3]).max()), 3),
            "reference_in_window_max": round(
                float(speed(reference_object[:frames, :3])[span].max()), 3
            ),
        },
        "object_height_m": {
            "at_failure": round(float(achieved[min(failure_frame, frames - 1), 2]), 3),
            "window_start": round(float(achieved[start, 2]), 3),
            "support_height": support_height_m,
            # Coming down onto the support, not merely being below it: an object already on the
            # floor is under every support in the scene, and that is not an event.
            "reaches_support": None
            if support_height_m is None
            else bool(
                achieved[start, 2] > support_height_m + 0.05
                and abs(float(achieved[min(failure_frame, frames - 1), 2]) - support_height_m)
                <= 0.05
            ),
        },
        "palm_to_object_cm": {
            "hand": sides[holding] if holding < len(sides) else str(holding),
            "window_start": round(float(to_palm[start, holding]), 1),
            "at_failure": round(float(to_palm[min(failure_frame, frames - 1), holding]), 1),
            "grew_by": round(
                float(to_palm[min(failure_frame, frames - 1), holding] - to_palm[start, holding]), 1
            ),
            "was_holding": bool(to_palm[start, holding] <= HOLDING_CM),
        },
        "reference_palm_to_reference_object_cm": {
            "window_start": round(float(to_palm_reference[start, holding]), 1),
            "at_failure": round(
                float(to_palm_reference[min(failure_frame, frames - 1), holding]), 1
            ),
            "reference_is_holding": bool(to_palm_reference[start, holding] <= HOLDING_CM),
        },
        "achieved_off_reference_cm": {
            "window_start": round(float(off_reference[start]), 1),
            "at_failure": round(float(off_reference[min(failure_frame, frames - 1)]), 1),
            "clip_max": round(float(off_reference.max()), 1),
            "already_lost_when_the_window_opens": bool(off_reference[start] > 10.0),
        },
        "body_speed_m_s": {
            "in_window_mean": round(float(speed(body)[span].mean()), 3),
            "moving": bool(speed(body)[span].mean() > STILL_M_PER_S),
        },
    }
    if closed is not None:
        record["reference_hand"] = {
            "finger_closure_rad_window_start": round(float(closed[start]), 3),
            "finger_closure_rad_at_failure": round(
                float(closed[min(failure_frame, frames - 1)]), 3
            ),
            "opens_in_window": bool(
                closed[min(failure_frame, frames - 1)] < closed[start] - CLOSED_RAD
            ),
        }
    if contact_per_hand.shape[1]:
        transitions = []
        for hand in range(contact_per_hand.shape[1]):
            column = contact_per_hand[span, hand] > 0.5
            if column.size and column.any() != column.all():
                changes = int(np.count_nonzero(np.diff(column.astype(int))))
                transitions.append(
                    {
                        "hand": sides[hand] if hand < len(sides) else str(hand),
                        "changes_in_window": changes,
                        "active_at_window_start": bool(column[0]),
                        "active_at_failure": bool(column[-1]),
                    }
                )
        record["reference_contact"] = {
            "transitions": transitions,
            "any_transition": bool(transitions),
            "hands_active_at_failure": [
                sides[hand] if hand < len(sides) else str(hand)
                for hand in range(contact_per_hand.shape[1])
                if contact_per_hand[min(failure_frame, frames - 1), hand] > 0.5
            ],
        }
    return record


def mechanism(record: dict[str, Any]) -> str:
    """One name for what the window shows, from the measurements alone.

    The first question is whether anything happens in this window at all: a clip whose object is
    already off its reference when the window opens went wrong earlier, and naming an event here
    would be naming the consequence.
    """
    palm = record["palm_to_object_cm"]
    reference_palm = record["reference_palm_to_reference_object_cm"]
    off = record["achieved_off_reference_cm"]
    hand = record.get("reference_hand") or {}
    contact = record.get("reference_contact") or {}
    if off["already_lost_when_the_window_opens"]:
        return f"already lost before this window ({off['window_start']:.0f} cm off when it opens)"
    if off["at_failure"] < 10.0:
        return "the shipping rollout does not fail here (still on its reference)"
    if hand.get("opens_in_window"):
        return "the reference hand opens"
    speeds = record["object_speed_m_s"]
    flung = speeds["in_window_max"] >= FLUNG_M_PER_S and speeds["in_window_max"] >= (
        FLUNG_RATIO * max(speeds["reference_in_window_max"], 1e-6)
    )
    if flung:
        return "the policy's object is flung far faster than the reference moves it"
    if not reference_palm["reference_is_holding"] and reference_palm["at_failure"] <= HOLDING_CM:
        return "lost as the hands arrive, before the grip closes"
    if contact.get("any_transition"):
        return "contact moves between hands"
    if palm["was_holding"] and palm["grew_by"] > 5:
        return "the object slides out of the grip"
    if record["object_height_m"].get("reaches_support"):
        return "the object reaches the support surface"
    if record["body_speed_m_s"]["moving"] and record["object_speed_m_s"]["in_window_max"] < 0.2:
        return "the body moves on without the object"
    return "no single event in the window"


def support_heights(dataset_root: Path) -> dict[int, float | None]:
    """The highest horizontal support top per clip, where the clip ships a primitive."""
    from ..supports import candidate_heights

    heights: dict[int, float | None] = {}
    for episode in dataset_episodes(dataset_root):
        path = dataset_root / "track_2" / TIER1 / episode.support_surface
        tops = candidate_heights(path if path.is_file() else None)
        heights[episode.index] = max(tops) if len(tops) > 1 else None
    return heights


def inertia_note(physics: Path | None) -> list[dict[str, Any]]:
    """How far each object's real shape inertia at 1 kg is from the 0.01 the URDF writes."""
    if physics is None or not physics.is_file():
        return []
    rows = []
    with physics.open() as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "tag": row["tag"],
                        "object": row["obj"],
                        "auc": float(row["auc"]),
                        "longest_m": float(row["longest_m"]),
                        "inertia_max_kg_m2": float(row["Imax"]),
                        "inertia_min_kg_m2": float(row["Imin"]),
                        "i_ratio_sim": float(row["I_ratio_sim"]),
                        "times_too_easy_to_spin": round(1.0 / float(row["I_ratio_sim"]), 1)
                        if float(row["I_ratio_sim"])
                        else None,
                    }
                )
            except (KeyError, ValueError):
                continue
    return sorted(rows, key=lambda r: r["i_ratio_sim"])


def paragraph(record: dict[str, Any], object_name: str, auc: float) -> str:
    """One clip's window in plain sentences, every number from `record`."""
    palm = record["palm_to_object_cm"]
    height = record["object_height_m"]
    speeds = record["object_speed_m_s"]
    hand = record.get("reference_hand") or {}
    contact = record.get("reference_contact") or {}
    start, stop = record["window"]

    sentences = [
        f"**{record['tag']} {object_name}** (AUC {auc:.3f}) is lost at frame"
        f" {record['failure_frame']}; the window is frames {start} to {stop}."
        f" The object is moving at {speeds['in_window_max']:.2f} m/s at most in that second,"
        f" against {speeds['clip_max']:.2f} m/s over the clip, and the reference moves it at"
        f" {speeds['reference_in_window_max']:.2f} m/s."
    ]
    reference_palm = record["reference_palm_to_reference_object_cm"]
    off = record["achieved_off_reference_cm"]
    sentences.append(
        f" Its object is already {off['window_start']:.0f} cm off the reference when the window"
        f" opens and {off['at_failure']:.0f} cm at the failure frame"
        f" (worst over the clip {off['clip_max']:.0f} cm)."
        if off["already_lost_when_the_window_opens"]
        else f" Its object is within {off['window_start']:.1f} cm of the reference when the window"
        f" opens and {off['at_failure']:.1f} cm at the failure frame."
    )
    sentences.append(
        f" In the reference, the {palm['hand']} palm is"
        f" {reference_palm['window_start']:.1f} cm from the surface of the object it is meant to"
        " be holding"
        + (
            ", so the reference has it in hand"
            if reference_palm["reference_is_holding"]
            else ", so the reference is not holding it here either"
        )
        + f"; the same palm is {palm['window_start']:.1f} cm from the policy's own object at the"
        f" window start and {palm['at_failure']:.1f} cm at the failure frame."
    )
    if hand:
        sentences.append(
            f" The reference hand goes from {hand['finger_closure_rad_window_start']:.2f} rad of"
            f" mean finger flexion to {hand['finger_closure_rad_at_failure']:.2f} rad"
            + (", so it opens." if hand["opens_in_window"] else ", so it does not open.")
        )
    if contact:
        if contact["any_transition"]:
            moves = ", ".join(
                f"{t['hand']} {'on' if t['active_at_failure'] else 'off'}"
                f" ({t['changes_in_window']} change{'s' if t['changes_in_window'] != 1 else ''})"
                for t in contact["transitions"]
            )
            sentences.append(f" The reference's own contact flags change here: {moves}.")
        else:
            hands = contact["hands_active_at_failure"]
            sentences.append(
                " The reference's contact flags do not change in the window"
                + (
                    f" ({' and '.join(hands)} active throughout)."
                    if hands
                    else " (no hand active)."
                )
            )
    if height["support_height"] is not None:
        sentences.append(
            f" The object is at {height['at_failure']:.2f} m with the support top at"
            f" {height['support_height']:.2f} m, so it"
            + (" reaches it." if height["reaches_support"] else " does not reach it.")
        )
    else:
        sentences.append(
            f" The clip ships no support primitive; the object is at {height['at_failure']:.2f} m"
            f" at the failure frame, from {height['window_start']:.2f} m when the window opened."
        )
    sentences.append(
        f" The body is moving at {record['body_speed_m_s']['in_window_mean']:.2f} m/s on average"
        + (
            ", so the robot is walking through this."
            if record["body_speed_m_s"]["moving"]
            else ", so the robot is close to stationary."
        )
    )
    sentences.append(f" **Mechanism: {record['mechanism']}.**")
    return "".join(sentences)


def render(records: list[dict[str, Any]], inertia: list[dict[str, Any]]) -> str:
    lines = [
        "# What happens at the failure moment",
        "",
        "Five clips fail while the object is nearly still, so the speed correlation that explains"
        " e04, e07, e14 and e25 cannot explain them. This is the one-second window before each"
        " one, measured rather than watched.",
        "",
        "**Where each number comes from.** The object track is the policy's own, from the rollout"
        " the entry ships. The hands, the contact flags and the body are the *reference's*: the"
        " rollout export carries the object alone, so there is no record of where the policy's own"
        ' hands were. "The hand opens" below always means the hand the policy was tracking'
        " opened. Whether its own hand followed is not in the data, and the distinction matters"
        " most for exactly these clips -- a policy that never closed its hand and a policy whose"
        " reference told it to let go look identical in the object track.",
        "",
    ]
    for record in records:
        lines += [record["paragraph"], ""]
    stale = [r for r in records if r["mechanism"].startswith("the shipping rollout does not fail")]
    if stale:
        lines += [
            "## Two of the five frames are from a rollout we no longer ship",
            "",
            "The failure frames come from the first wave's triage. "
            + " and ".join(
                f"**{r['tag']}** now ships the {r['taxonomy_class'].replace('_', ' ')} rollout"
                " from a second-wave arm, whose object is"
                f" {r['achieved_off_reference_cm']['at_failure']:.1f} cm off its reference at"
                f" frame {r['failure_frame']}"
                for r in stale
            )
            + ". Neither has a loss event anywhere in the clip under the taxonomy's rules, so there"
            " is no moment to explain: what they lose is the angle, which is the"
            " held-but-drifting class and a different fix. The windows are reported above all the"
            " same, because 'nothing happens here' is the answer to the question asked.",
            "",
        ]
    lines += [
        "## The mechanisms",
        "",
        "| clip | object | AUC | failure frame | off reference at window start | object speed"
        " in window | reference palm on object | mechanism |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['tag']} | {record['object']} | {record['auc']:.3f} |"
            f" {record['failure_frame']} |"
            f" {record['achieved_off_reference_cm']['window_start']:.0f} cm |"
            f" {record['object_speed_m_s']['in_window_max']:.2f} m/s |"
            f" {record['reference_palm_to_reference_object_cm']['window_start']:.0f} cm |"
            f" {record['mechanism']} |"
        )
    if inertia:
        lines += [
            "",
            "## The objects are all 1 kg boxes, and some are far too easy to spin",
            "",
            "`soma_to_g1.py` writes every object URDF with mass 1.0 kg and inertia diag 0.01."
            " `I_ratio_sim` is that 0.01 against the object's own shape inertia at the same 1 kg:"
            " below 1 means the simulated object spins more easily than the real shape would, and"
            " `1/ratio` is how many times too easy. A long thin object is the worst case, because"
            " its real inertia about the short axes is large while the URDF's is not.",
            "",
            "| clip | object | longest axis m | shape inertia max | I_ratio_sim | times too easy"
            " to spin | AUC |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in inertia[:12]:
            lines.append(
                f"| {row['tag']} | {row['object']} | {row['longest_m']:.2f} |"
                f" {row['inertia_max_kg_m2']:.4f} | {row['i_ratio_sim']:.2f} |"
                f" {row['times_too_easy_to_spin']}x | {row['auc']:.3f} |"
            )
        lines += [
            "",
            "This is reported, not acted on. The URDFs are not edited here: whether object physics"
            " may be changed at all is a rules question for Luai, and if the answer is no, the"
            " finding still stands as a reason the long thin objects behave as they do.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clips", required=True, help="eXX=FRAME,eYY=FRAME")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path, required=True, help="the staged soma/ tree")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True, help="for the kit's vertex sampler")
    parser.add_argument("--taxonomy", type=Path, help="results/failure_taxonomy_wave2.json")
    parser.add_argument(
        "--physics", type=Path, help="the object-physics table, for the inertia note"
    )
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", dest="record_out", type=Path, required=True)
    args = parser.parse_args(argv)

    wanted = {}
    for item in args.clips.split(","):
        tag, _, frame = item.partition("=")
        if not frame:
            raise SystemExit(f"--clips wants eXX=FRAME, got {item!r}")
        wanted[tag.strip()] = int(frame)

    taxonomy = {}
    if args.taxonomy and args.taxonomy.is_file():
        taxonomy = {c["tag"]: c for c in json.loads(args.taxonomy.read_text())["clips"]}
    metadata = {f"e{e.index:02d}": e for e in dataset_episodes(args.dataset_root)}
    heights = support_heights(args.dataset_root)

    # The kit's own sampler, so "the object's surface" is the surface the score is computed over.
    if str(args.kit_root) not in sys.path:
        sys.path.insert(0, str(args.kit_root))
    from v2dlb.vertices import build_blob

    _blob, vertex_cloud = build_blob(
        args.dataset_root / "track_2" / TIER1 / "mesh",
        sorted({metadata[tag].object_name for tag in wanted}),
    )

    records = []
    for tag, frame in wanted.items():
        episode = metadata[tag]
        achieved, reference_object = read_rollout(
            args.rollouts / f"episode_{episode.index:06d}.parquet"
        )
        state = reference_state(motion_for(args.motion_root, episode.sequence_id))
        record = describe(
            tag=tag,
            failure_frame=frame,
            achieved=achieved,
            reference_object=reference_object,
            state=state,
            vertices=np.asarray(vertex_cloud[episode.object_name], dtype=np.float64),
            support_height_m=heights.get(episode.index),
            window=args.window,
        )
        record["mechanism"] = mechanism(record)
        record["object"] = episode.object_name
        record["auc"] = float((taxonomy.get(tag) or {}).get("kit_add_auc", 0.0))
        record["taxonomy_class"] = (taxonomy.get(tag) or {}).get("failure_class")
        record["taxonomy_frame"] = (taxonomy.get(tag) or {}).get("failure_frame")
        record["paragraph"] = paragraph(record, episode.object_name, record["auc"])
        records.append(record)

    inertia = inertia_note(args.physics)
    args.record_out.parent.mkdir(parents=True, exist_ok=True)
    args.record_out.write_text(
        json.dumps(
            {
                "what": "the one-second window before each clip's failure frame, measured",
                "window_frames": args.window,
                "hands_are_the_reference": True,
                "clips": records,
                "inertia": inertia,
            },
            indent=1,
        )
        + "\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(records, inertia))
    for record in records:
        print(f"{record['tag']} f{record['failure_frame']:4d}  {record['mechanism']}")
    print(f"wrote {args.out} and {args.record_out}")
