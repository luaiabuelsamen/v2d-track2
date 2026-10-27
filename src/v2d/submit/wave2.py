"""Stage the final-candidate Tier 2 entry: per clip, whatever the adoption record says ships.

The first wave's entry was staged from one run directory, so a small script could walk it. After
the second wave a clip's shipping rollout can come from any of seven places -- the first-wave run
or one of the arms A, B1, B2, B3, B4, B5, B4x -- and `results/second_wave_adoption.json` is the
only record of which. That record is written by `v2d adopt-check` under the registered input-only
rule, so this command never re-decides anything: it reads `clips[tag]["ships"]` and stages it.

Two things it does check rather than assume, because both would corrupt the entry silently:

- **The frame reference per clip.** The kit packer converts a rollout out of the policy's frame
  using the reference that policy tracked. Those differ across arms: every second-wave arm trained
  on tetool's bridge_v1 clean retarget (B5 included, which is why B5's variable is fingers and not
  the reference), while the first wave's e01 trained on the released input. The root is therefore
  taken from the *shipping* arm's own `provenance.json`, falling back to the released dataset only
  for a clip that arm's provenance does not list, and the choice is recorded per episode.
- **Completeness.** A Tier 2 entry is 30 episodes, and the adoption record holds only the clips
  the second wave re-ran -- 29, because nothing re-ran e21. A clip the record does not mention
  still has to ship, so `--first-wave-results` supplies it from that run's own z-drop selection.
  A clip still awaiting its arm's result, or a checkpoint directory missing its rollout, is a
  refusal rather than a short CSV; `--allow-partial` exists for a dry run and stamps the record
  so a partial pack cannot be mistaken for the entry.

    v2d stage-entry-wave2 --adoption results/second_wave_adoption.json \
      --stage-root $S --first-wave-provenance results/t2_combined_provenance.json \
      --first-wave-results $S/t2_combined_run/results --released $S/data --out $S/t2_entry_wave2

The output is the two trees `v2d pack --tier tier2` wants (`--rollout-root`, `--frame-references`)
plus `inputs.json`, the per-episode provenance the packed record quotes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..dataset import object_pose

TIER2_EPISODES = 30
FIRST_WAVE = "first wave"
DEFAULT_TIER = "tier2"


def ships(clip: dict[str, Any]) -> dict[str, Any]:
    """The rollout this clip submits: its `ships` block, or the shipped one if nothing replaced it.

    `adopt-check` writes `ships` for every decided clip, but a clip no arm re-ran keeps only
    `shipped`, and both have to pack the same way.
    """
    if clip.get("ships"):
        return dict(clip["ships"])
    shipped = clip["shipped"]
    return {
        "arm": FIRST_WAVE,
        "selected_environment_steps": shipped["selected_environment_steps"],
        "selected_checkpoint_sha256": shipped["selected_checkpoint_sha256"],
        "checkpoint_dir": shipped["checkpoint_dir"],
    }


def reference_roots(
    stage_root: Path, first_wave_provenance: Path | None
) -> dict[str, dict[str, str]]:
    """arm -> {clip tag: the reference dataset root that arm trained that clip on}."""
    roots: dict[str, dict[str, str]] = {}
    if first_wave_provenance and first_wave_provenance.exists():
        record = json.loads(first_wave_provenance.read_text())
        roots[FIRST_WAVE] = dict(record.get("ref_dataset_by_clip", {}))
    for provenance in sorted(stage_root.glob("sw_*/provenance.json")):
        arm = provenance.parent.name[len("sw_") :]
        record = json.loads(provenance.read_text())
        roots[arm] = dict(record.get("ref_dataset_by_clip", {}))
    return roots


def reference_root(
    roots: dict[str, dict[str, str]], arm: str, tag: str, released: Path
) -> tuple[Path, bool]:
    """Where `arm` read `tag`'s reference, and whether that is the released input."""
    root = roots.get(arm, {}).get(tag)
    if root is None:
        return released, True
    return Path(root), Path(root) == released


def first_wave_only(results: Path, *, skip: set[str]) -> dict[str, dict[str, Any]]:
    """Clips the second wave never re-ran, shaped like an adoption entry so `ships` reads them.

    `adopt-check` records only the clips that had a candidate, so a clip no arm touched (e21) is
    absent from it entirely. Dropping it would pack 29 episodes; re-deciding it would be a second
    selection rule. Its first-wave `selection.json` is the same z-drop pick that is on the board.
    """
    found: dict[str, dict[str, Any]] = {}
    for selection_path in sorted(results.glob("e[0-9][0-9]/selection.json")):
        tag = selection_path.parent.name
        if tag in skip:
            continue
        selection = json.loads(selection_path.read_text())["episodes"][tag]
        found[tag] = {
            "episode": int(selection["episode"]),
            "shipped": {
                "checkpoint_dir": str(
                    selection_path.parent / f"policy_{selection['selected_environment_steps']}"
                ),
                "selected_environment_steps": selection["selected_environment_steps"],
                "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
            },
        }
    return found


def stage(
    adoption: Path,
    out: Path,
    *,
    stage_root: Path,
    released: Path,
    first_wave_provenance: Path | None = None,
    first_wave_results: Path | None = None,
    tier: str = DEFAULT_TIER,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Write the rollout and frame-reference trees, and return the record that describes them."""
    record = json.loads(adoption.read_text())
    awaiting = dict(record.get("clips_awaiting_results") or {})
    if awaiting and not allow_partial:
        pending = ", ".join(f"{arm} {' '.join(clips)}" for arm, clips in sorted(awaiting.items()))
        raise SystemExit(f"refusing to stage: still awaiting {pending} (--allow-partial to force)")

    clips = dict(record["clips"])
    if first_wave_results is not None:
        clips.update(first_wave_only(first_wave_results, skip=set(clips)))
    if len(clips) != TIER2_EPISODES and not allow_partial:
        raise SystemExit(
            f"refusing to stage: {len(clips)} clips decided, a Tier 2 entry needs {TIER2_EPISODES}"
        )

    rollouts, frame_refs = out / "rollouts", out / "frame_refs"
    for directory in (rollouts, frame_refs):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)

    roots = reference_roots(stage_root, first_wave_provenance)
    episodes: list[dict[str, Any]] = []
    for tag in sorted(clips):
        clip = clips[tag]
        episode = int(clip["episode"])
        ship = ships(clip)
        cell = Path(ship["checkpoint_dir"])
        parquet, metrics = cell / "objects.parquet", cell / "metrics.json"
        for needed in (parquet, metrics):
            if not needed.exists():
                raise SystemExit(
                    f"refusing to stage: {tag} ships {ship['arm']} but {needed} is missing"
                )
        shutil.copy2(parquet, rollouts / f"episode_{episode:06d}.parquet")
        shutil.copy2(metrics, rollouts / f"episode_{episode:06d}.json")

        root, is_released = reference_root(roots, ship["arm"], tag, released)
        pose, _visible = object_pose(root, tier, episode)
        np.savez(
            frame_refs / f"episode_{episode:06d}.npz",
            coordinate_frame=np.array("released_track2_world"),
            episode_index=np.array(episode, dtype=np.int64),
            frame_index=np.arange(len(pose), dtype=np.int64),
            pose_xyzw=pose.reshape(len(pose), 1, 7).astype(np.float64),
        )
        episodes.append(
            {
                "episode": episode,
                "tag": tag,
                "ships_from": ship["arm"],
                "selected_environment_steps": ship["selected_environment_steps"],
                "checkpoint_sha256": ship["selected_checkpoint_sha256"],
                "checkpoint_dir": str(cell),
                "frames": len(pose),
                "frame_reference_root": str(root),
                "frame_reference_is_released_input": is_released,
            }
        )

    by_arm: dict[str, int] = {}
    for entry in episodes:
        by_arm[entry["ships_from"]] = by_arm.get(entry["ships_from"], 0) + 1
    staged = {
        "what": "the final-candidate Tier 2 entry's inputs: the adopted rollout per episode",
        "reference_tier": tier,
        "adoption_record": str(adoption),
        "selection_rule": "z-drop per clip, then the registered input-only adoption rule",
        "partial": bool(awaiting) or len(clips) != TIER2_EPISODES,
        "awaiting": awaiting,
        "rollouts": str(rollouts),
        "frame_references": str(frame_refs),
        "episodes_by_arm": dict(sorted(by_arm.items())),
        "distinct_checkpoint_digests": len({e["checkpoint_sha256"] for e in episodes}),
        "episodes": episodes,
    }
    (out / "inputs.json").write_text(json.dumps(staged, indent=2) + "\n")
    return staged


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--adoption", type=Path, required=True, metavar="JSON")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument("--stage-root", type=Path, required=True, help="holds sw_<arm>/")
    parser.add_argument("--released", type=Path, required=True, help="the released dataset root")
    parser.add_argument("--first-wave-provenance", type=Path)
    parser.add_argument(
        "--first-wave-results",
        type=Path,
        help="the first wave's results dir, for clips the second wave never re-ran",
    )
    parser.add_argument(
        "--tier",
        default=DEFAULT_TIER,
        choices=["tier1", "tier2"],
        help="the tier whose released object pose the rollouts are converted against",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="stage what has landed; the record is stamped partial and must not be submitted",
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    staged = stage(
        args.adoption,
        args.out,
        stage_root=args.stage_root,
        released=args.released,
        first_wave_provenance=args.first_wave_provenance,
        first_wave_results=args.first_wave_results,
        tier=args.tier,
        allow_partial=args.allow_partial,
    )
    counts = ", ".join(f"{arm} {n}" for arm, n in staged["episodes_by_arm"].items())
    print(f"{len(staged['episodes'])} episodes staged from: {counts}")
    print(f"{staged['distinct_checkpoint_digests']} distinct checkpoint digests")
    released = [e["tag"] for e in staged["episodes"] if e["frame_reference_is_released_input"]]
    cleaned = len(staged["episodes"]) - len(released)
    print(f"frame reference: {cleaned} cleaned, {released or 'none'} released")
    if staged["partial"]:
        print("PARTIAL: not a submittable entry")
    print(f"wrote {args.out / 'inputs.json'}")
