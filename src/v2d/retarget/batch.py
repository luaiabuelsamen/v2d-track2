"""Retarget every released Track 2 episode to G1 on CPU, validating each clip.

Per episode: build the sequence folder `soma_to_g1.py` wants, retarget it, validate the result
against the released object trajectory, then delete the (regenerable) input folder, because the
GLB-to-OBJ conversion it holds is ~22 MB and the Jetson has a 20 GB disk floor.

Resumable: an episode whose output parquet already exists and validates is skipped.

`--human-npz` is forwarded to the converter alone, since the validator checks the object
trajectory and the human override does not change it.

`--object-npz` / `--object-from-tier` are forwarded to both the converter and the validator, so
a reference-cleanup arm is retargeted from its own object trajectory and checked against the
same one. Give each arm its own `--seq-root/--out-root/--log-dir/--validation-dir`.

Usage:
  python scripts/retarget_all.py --dataset-root DATA --repo REPO --soma-root SOMA \
      --seq-root SEQ --out-root OUT --tiers tier1 tier2 --workers 2
      [--object-npz DIR | --object-from-tier tier1]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: The repo root, three levels up from src/v2d/retarget/.
REPO_ROOT = HERE.parents[2]
PYTHON = sys.executable
#: Child stages are spawned as modules, not as file paths. Paths were fragile for a concrete
#: reason: moving these files during a live batch made 11 of 30 clips fail with "can't open
#: file", because the parent resolves each child at spawn time. `-m` resolves through the
#: installed package instead. They stay separate processes on purpose, so an OOM kill costs one
#: clip rather than the batch, and so PYTHONPATH can be stripped per child.
SEQUENCES_MODULE = "v2d.retarget.sequences"
VALIDATE_MODULE = "v2d.retarget.validate"
TIER_DIRS = {"tier1": "tier_1_multiview_caption", "tier2": "tier_2_synthetic_noise"}
DISK_FLOOR_GB = 20.0
MEM_FLOOR_MB = 2500.0


def free_disk_gb(path: Path) -> float:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / 1e9


def available_mem_mb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0
    return float("inf")


def check_floors(path: Path) -> str | None:
    disk = free_disk_gb(path)
    if disk < DISK_FLOOR_GB:
        return f"disk {disk:.1f} GB is under the {DISK_FLOOR_GB} GB floor"
    memory = available_mem_mb()
    if memory < MEM_FLOOR_MB:
        return f"MemAvailable {memory:.0f} MB is under the {MEM_FLOOR_MB:.0f} MB floor"
    return None


def run(command: list[str], log: Path, env: dict | None = None) -> int:
    with log.open("ab") as handle:
        handle.write(b"\n$ " + " ".join(command).encode() + b"\n")
        handle.flush()
        return subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT, env=env)


def episode_metadata(dataset_root: Path, tier: str) -> list[dict]:
    path = dataset_root / "track_2" / TIER_DIRS[tier] / "meta" / "episodes_metadata.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["episode_index"])


def process(args, tier: str, row: dict, abort: dict) -> dict:
    episode = int(row["episode_index"])
    sequence_id = row["sequence_id"]
    tag = f"{tier}/e{episode:02d}"
    seq_root = args.seq_root / tier
    out_root = args.out_root / tier
    sequence_dir = seq_root / f"sequence_id={sequence_id}"
    out_parquet = (
        out_root
        / "whole_body"
        / "soma"
        / f"sequence_id={sequence_id}"
        / "robot_name=g1"
        / "data.parquet"
    )
    log = args.log_dir / f"{tier}_e{episode:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    record = {"tier": tier, "episode": episode, "sequence_id": sequence_id, "object": row["object"]}

    if abort.get("reason"):
        return {**record, "status": "aborted", "detail": abort["reason"]}
    reason = check_floors(args.out_root)
    if reason:
        abort["reason"] = reason
        return {**record, "status": "aborted", "detail": reason}

    validation_out = args.validation_dir / f"{tier}_e{episode:02d}.json"
    if out_parquet.is_file() and validation_out.is_file():
        try:
            previous = json.loads(validation_out.read_text())
            if previous.get("pass"):
                return {**record, "status": "skipped", **_summary(previous)}
        except json.JSONDecodeError:
            pass

    override: list[str] = []
    if args.object_npz is not None:
        override = ["--object-npz", str(args.object_npz)]
    elif args.object_from_tier is not None:
        override = ["--object-from-tier", args.object_from_tier]
    # The human override goes only to the converter: the validator checks the object trajectory,
    # which is not what --human-npz changes.
    human = ["--human-npz", str(args.human_npz)] if args.human_npz is not None else []

    started = time.time()
    # 1. Build the sequence folder.
    code = run(
        [
            PYTHON,
            "-m",
            SEQUENCES_MODULE,
            "--dataset-root",
            str(args.dataset_root),
            "--tier",
            tier,
            "--out",
            str(seq_root),
            "--soma-neutral",
            str(args.soma_root / "SOMA_neutral.npz"),
            "--episodes",
            str(episode),
            *override,
            *human,
        ],
        log,
    )
    if code != 0:
        return {**record, "status": "convert_failed", "log": str(log)}

    # 2. Retarget. Keep the thread count low: several of these run at once.
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    env.pop("PYTHONPATH", None)  # a PyTorch source tree on the global path shadows the venv
    code = run(
        [
            PYTHON,
            "-I",
            str(args.repo / "robotic_grounding" / "scripts" / "retarget" / "soma_to_g1.py"),
            str(sequence_dir),
            "--save",
            "--soma-data-root",
            str(args.soma_root),
            "--motion-root",
            str(out_root),
            "--fps",
            "30",
        ],
        log,
        env,
    )
    if code != 0:
        return {**record, "status": "retarget_failed", "log": str(log)}

    # 3. Validate against the released trajectory.
    code = run(
        [
            PYTHON,
            "-m",
            VALIDATE_MODULE,
            "--motion-parquet",
            str(out_parquet),
            "--dataset-root",
            str(args.dataset_root),
            "--episode",
            str(episode),
            "--tier",
            tier,
            *override,
            "-o",
            str(validation_out),
        ],
        log,
    )
    elapsed = round(time.time() - started, 1)
    if not validation_out.is_file():
        return {**record, "status": "validate_failed", "seconds": elapsed, "log": str(log)}
    validation = json.loads(validation_out.read_text())

    # 4. Drop the regenerable input folder; its mesh conversion is the bulk of the disk cost.
    if validation.get("pass") and not args.keep_sequences:
        shutil.rmtree(sequence_dir, ignore_errors=True)

    status = "ok" if (code == 0 and validation.get("pass")) else "validation_failed"
    print(
        f"[{tag}] {status} in {elapsed}s  residual={validation.get('rigid_fit_residual_mm')}mm "
        f"orient={validation.get('orientation_residual_max_deg')}deg",
        flush=True,
    )
    return {**record, "status": status, "seconds": elapsed, **_summary(validation)}


def _summary(validation: dict) -> dict:
    return {
        "frames": validation.get("frames_compared"),
        "rigid_fit_residual_mm": validation.get("rigid_fit_residual_mm"),
        "orientation_residual_max_deg": validation.get("orientation_residual_max_deg"),
        "relative_rotation_max_diff_deg": validation.get("relative_rotation_max_diff_deg"),
        "pass": validation.get("pass"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--repo", type=Path, required=True, help="sparse checkout holding robotic_grounding/"
    )
    parser.add_argument("--soma-root", type=Path, required=True, help="staged body_models/soma")
    parser.add_argument("--seq-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--tiers", nargs="+", default=["tier1"], choices=sorted(TIER_DIRS))
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2, help="OMP threads per worker")
    parser.add_argument("--keep-sequences", action="store_true")
    source = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--human-npz",
        type=Path,
        default=None,
        help="cleaned human root from clean_root_planting.py; forwarded to the converter only",
    )
    source.add_argument(
        "--object-npz",
        type=Path,
        default=None,
        help="object trajectory from episode_%%06d.npz in this directory",
    )
    source.add_argument("--object-from-tier", default=None, choices=sorted(TIER_DIRS))
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args(argv)
    for directory in (args.log_dir, args.validation_dir, args.out_root, args.seq_root):
        directory.mkdir(parents=True, exist_ok=True)

    reason = check_floors(args.out_root)
    if reason:
        raise SystemExit(f"refusing to start: {reason}")

    abort: dict = {}
    results = []
    for tier in args.tiers:
        rows = [
            r
            for r in episode_metadata(args.dataset_root, tier)
            if args.episodes is None or int(r["episode_index"]) in args.episodes
        ]
        print(f"\n=== {tier}: {len(rows)} episodes, {args.workers} workers ===", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # `tier` is bound as a default so the closure cannot pick up a later value: the
            # map is consumed inside this iteration today, but that is an accident of
            # scheduling, not a guarantee.
            results += list(
                pool.map(lambda row, _tier=tier: process(args, _tier, row, abort), rows)
            )

    # Record which reference these motions came from, so a downstream consumer (the Brev gate)
    # cannot train on the raw noisy reference by accident.
    source = str(args.object_npz or args.object_from_tier or "released tier")
    provenance = {
        "object_source": source,
        "cleaned_reference": bool(args.object_npz or args.object_from_tier),
        "cleaned_human": bool(args.human_npz),
        "human_source": str(args.human_npz or "released tier"),
        "tiers": list(args.tiers),
        "dataset_root": str(args.dataset_root),
        "retarget_commit": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or None,
    }
    (args.out_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    counts: dict[str, int] = {}
    for item in results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    passed = [r for r in results if r["status"] in ("ok", "skipped")]
    summary = {
        "object_source": str(args.object_npz or args.object_from_tier or "released tier"),
        "counts": counts,
        "episodes_total": len(results),
        "episodes_passed": len(passed),
        "worst_rigid_fit_residual_mm": max((r.get("rigid_fit_residual_mm") or 0) for r in passed)
        if passed
        else None,
        "worst_orientation_residual_deg": max(
            (r.get("orientation_residual_max_deg") or 0) for r in passed
        )
        if passed
        else None,
        "abort_reason": abort.get("reason"),
        "free_disk_gb_after": round(free_disk_gb(Path.cwd()), 1),
    }
    print("\n" + json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"summary": summary, "episodes": results}, indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote {args.out}")
    sys.exit(0 if summary["episodes_passed"] == summary["episodes_total"] else 1)


if __name__ == "__main__":
    main()
