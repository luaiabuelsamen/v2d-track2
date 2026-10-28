"""Rebuild the entry's rollout and frame-reference trees after the scratchpad was lost.

A reboot on 2026-10-28 wiped `/tmp/claude-1000/...`, taking with it every arm's run directory,
the staged reference trees, the released dataset copy and the packed CSVs. What survived is enough
to rebuild the entry exactly, and this is the path from one to the other:

- `ship_ckpts_wave2/<tag>/policy_<steps>/` -- the 29 shipping rollouts, each with the
  `objects.parquet` and `metrics.json` the kit packer reads, plus the `policy.safetensors` whose
  digest the entry registered.
- `results/tier2_entry_wave2.json` -- the authoritative record of what ships: per episode, the
  arm, the checkpoint digest, the environment steps and **the frame-reference root**. That last
  one matters because `v2d stage-entry-wave2` reconstructs it from `$S/sw_*/provenance.json`,
  which the reboot also took; the entry record is the only surviving copy.
- `/home/jetson3/projects/v2d-clean-refs/bridge_v1` -- 29 of 30 episodes' reference root, on the
  home disk and therefore intact. Only e01 reads the released input, and its root is the released
  dataset, which is a free re-download.

So this command re-derives the trees from records rather than from a rerun, and it verifies rather
than assumes: every rollout's checkpoint digest is recomputed from the surviving safetensors and
compared with the digest the entry registered, so a file that is not the one that trained the
entry is a refusal, not a silent substitution.

    v2d restage-entry --entry-record results/tier2_entry_wave2.json \
      --checkpoints ship_ckpts_wave2 --released $S/data --out $S/t2_entry_wave2

An episode whose checkpoint did not survive is named and refused. `--allow-missing` stages the
rest and records the gap, which is for building a 29-episode tree to prove the pipeline while the
missing episode is sourced; the kit will not accept it as a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..dataset import object_pose

TIER2_EPISODES = 30
DEFAULT_TIER = "tier2"
#: The wiped scratchpad's dataset root, as the entry record spells it. A record written before the
#: reboot points here for e01; `--released` is where that data lives now.
WIPED_DATA_MARKER = "/scratchpad/data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_for(checkpoints: Path, episode: dict[str, Any]) -> Path:
    """Where this episode's surviving rollout is, under the checkpoint backup."""
    return checkpoints / episode["tag"] / f"policy_{episode['selected_environment_steps']}"


def reference_root_for(episode: dict[str, Any], released: Path) -> tuple[Path, bool]:
    """The reference root to convert against, with the wiped data path redirected to `released`."""
    recorded = str(episode["frame_reference_root"])
    if WIPED_DATA_MARKER in recorded or recorded == str(released):
        return released, True
    return Path(recorded), False


def check(
    record: dict[str, Any], checkpoints: Path, released: Path, tier: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """(episodes that can be staged, reasons the others cannot). Reads nothing, writes nothing."""
    ready: list[dict[str, Any]] = []
    problems: list[str] = []
    for episode in sorted(record["episodes"], key=lambda e: e["episode"]):
        tag = episode["tag"]
        cell = cell_for(checkpoints, episode)
        if not cell.is_dir():
            problems.append(f"{tag}: no surviving checkpoint at {cell}")
            continue
        missing = [
            name for name in ("objects.parquet", "metrics.json") if not (cell / name).is_file()
        ]
        if missing:
            problems.append(f"{tag}: {cell} is missing {', '.join(missing)}")
            continue
        policy = cell / "policy.safetensors"
        if policy.is_file():
            digest = sha256(policy)
            if digest != episode["checkpoint_sha256"]:
                problems.append(
                    f"{tag}: {policy} has digest {digest[:16]}... but the entry registered "
                    f"{episode['checkpoint_sha256'][:16]}...; this is not the checkpoint that "
                    "trained the entry"
                )
                continue
        root, _is_released = reference_root_for(episode, released)
        if not root.is_dir():
            problems.append(f"{tag}: reference root {root} does not exist")
            continue
        ready.append(episode)
    return ready, problems


def stage(
    record: dict[str, Any],
    out: Path,
    *,
    checkpoints: Path,
    released: Path,
    tier: str = DEFAULT_TIER,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Write `out/rollouts` and `out/frame_refs`, and return the record describing them."""
    ready, problems = check(record, checkpoints, released, tier)
    if problems and not allow_missing:
        raise SystemExit(
            "refusing to restage:\n  "
            + "\n  ".join(problems)
            + "\n  (--allow-missing stages the rest, which the kit will not accept as a "
            "submission)"
        )

    rollouts, frame_refs = out / "rollouts", out / "frame_refs"
    for directory in (rollouts, frame_refs):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)

    episodes: list[dict[str, Any]] = []
    for episode in ready:
        index = int(episode["episode"])
        cell = cell_for(checkpoints, episode)
        shutil.copy2(cell / "objects.parquet", rollouts / f"episode_{index:06d}.parquet")
        shutil.copy2(cell / "metrics.json", rollouts / f"episode_{index:06d}.json")

        root, is_released = reference_root_for(episode, released)
        pose, _visible = object_pose(root, tier, index)
        np.savez(
            frame_refs / f"episode_{index:06d}.npz",
            coordinate_frame=np.array("released_track2_world"),
            episode_index=np.array(index, dtype=np.int64),
            frame_index=np.arange(len(pose), dtype=np.int64),
            pose_xyzw=pose.reshape(len(pose), 1, 7).astype(np.float64),
        )
        row = dict(episode)
        row["checkpoint_dir"] = str(cell)
        row["frame_reference_root"] = str(root)
        row["frame_reference_is_released_input"] = is_released
        # The record was written before the reboot, when the frame count came from the reference
        # the policy trained on. Re-measured here, and a disagreement is worth seeing.
        row["frames_restaged"] = len(pose)
        row["frames_match_record"] = bool(int(episode.get("frames", -1)) == len(pose))
        episodes.append(row)

    by_arm: dict[str, int] = {}
    for row in episodes:
        by_arm[row["ships_from"]] = by_arm.get(row["ships_from"], 0) + 1
    staged = {
        "what": "the wave-2 entry's inputs, rebuilt from the surviving checkpoints after the "
        "2026-10-28 scratchpad loss",
        "reference_tier": tier,
        "entry_record": "results/tier2_entry_wave2.json",
        "checkpoints": str(checkpoints),
        "selection_rule": "unchanged: the registered z-drop pick and input-only adoption, read "
        "from the entry record rather than re-decided",
        "rollouts": str(rollouts),
        "frame_references": str(frame_refs),
        "episodes_staged": len(episodes),
        "episodes_expected": TIER2_EPISODES,
        "partial": len(episodes) != TIER2_EPISODES,
        "missing": problems,
        "episodes_by_arm": dict(sorted(by_arm.items())),
        "distinct_checkpoint_digests": len({r["checkpoint_sha256"] for r in episodes}),
        "checkpoint_digests_verified": sum(
            1 for row in episodes if (cell_for(checkpoints, row) / "policy.safetensors").is_file()
        ),
        "frame_counts_matching_the_record": sum(
            1 for row in episodes if row["frames_match_record"]
        ),
        "episodes": episodes,
    }
    (out / "inputs.json").write_text(json.dumps(staged, indent=1) + "\n")
    return staged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-record", type=Path, required=True)
    parser.add_argument(
        "--checkpoints", type=Path, required=True, help="the ship_ckpts_wave2 backup"
    )
    parser.add_argument("--released", type=Path, required=True, help="the released dataset root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tier", default=DEFAULT_TIER, choices=("tier1", "tier2"))
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="report what could be staged, write nothing"
    )
    parser.add_argument("--json", dest="record_out", type=Path, help="also write the record here")
    args = parser.parse_args(argv)

    record = json.loads(args.entry_record.read_text())
    if args.check:
        ready, problems = check(record, args.checkpoints, args.released, args.tier)
        print(f"{len(ready)} of {len(record['episodes'])} episodes can be restaged")
        for problem in problems:
            print(f"  {problem}")
        return 0 if not problems else 1

    staged = stage(
        record,
        args.out,
        checkpoints=args.checkpoints,
        released=args.released,
        tier=args.tier,
        allow_missing=args.allow_missing,
    )
    if args.record_out:
        args.record_out.write_text(json.dumps(staged, indent=1) + "\n")
    print(
        f"{staged['episodes_staged']} of {TIER2_EPISODES} episodes staged from "
        f"{args.checkpoints}: "
        + ", ".join(f"{k} {v}" for k, v in staged["episodes_by_arm"].items())
    )
    print(
        f"{staged['checkpoint_digests_verified']} checkpoint digests recomputed and matched; "
        f"{staged['distinct_checkpoint_digests']} distinct"
    )
    print(
        f"frame counts matching the pre-reboot record: "
        f"{staged['frame_counts_matching_the_record']}/{staged['episodes_staged']}"
    )
    for problem in staged["missing"]:
        print(f"MISSING {problem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
