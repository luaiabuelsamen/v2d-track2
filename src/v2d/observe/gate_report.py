"""Turn a finished Brev gate run into a W&B run and a private Hugging Face upload.

The host does the logging, not the instance. `evaluate_policy.py` calls `wandb.init` itself, but
giving a rented VM a W&B token to make that work is not a trade worth making, so the container
runs with `WANDB_MODE=disabled` and everything is logged here from the files copied back.

What gets recorded, which is the project's observability contract: the config, the git sha, the
provenance of the reference the run trained against, learning curves where the run emitted them,
per-clip scores, and cost in dollars and GPU-hours.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hf_store import open_store
from .wandb_run import start_run


@dataclass
class GateRun:
    """Everything a gate run produced, read off disk."""

    phase: str
    directory: Path
    instance_type: str | None = None
    usd_per_hour: float | None = None
    wall_hours: float | None = None
    extrapolation: dict[str, Any] = field(default_factory=dict)
    eval_metrics: dict[str, Any] = field(default_factory=dict)
    local_scores: dict[str, Any] = field(default_factory=dict)
    files: list[Path] = field(default_factory=list)

    @property
    def usd(self) -> float | None:
        if self.wall_hours is None or self.usd_per_hour is None:
            return None
        return round(self.wall_hours * self.usd_per_hour, 4)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def collect(directory: Path, phase: str = "probe") -> GateRun:
    """Read a `results/gate/` directory into a `GateRun`."""
    run = GateRun(phase=phase, directory=directory)
    if not directory.is_dir():
        return run
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            run.files.append(path)
    for name, slot in (
        (f"gate_{phase}.json", "extrapolation"),
        ("extrapolation.json", "extrapolation"),
        ("b1_eval_metrics.json", "eval_metrics"),
        ("eval_metrics.json", "eval_metrics"),
        ("b1_local_score.json", "local_scores"),
    ):
        path = directory / name
        if path.is_file():
            setattr(run, slot, {**getattr(run, slot), **_read_json(path)})
    return run


def report(
    directory: Path,
    *,
    phase: str = "probe",
    instance_type: str | None = None,
    usd_per_hour: float | None = None,
    wall_hours: float | None = None,
    motion_root: Path | None = None,
    hf_repo: str | None = "v2d-track2",
    project: str = "v2d-track2",
    upload_suffixes: tuple[str, ...] = (".json", ".log", ".txt", ".parquet", ".safetensors"),
) -> dict[str, Any]:
    """Log a gate run to W&B and push its artefacts to the private HF repo.

    Returns a record of what was logged and where, which the caller writes to `results/`.
    """
    run = collect(directory, phase)
    run.instance_type = instance_type
    run.usd_per_hour = usd_per_hour
    run.wall_hours = wall_hours

    config: dict[str, Any] = {
        "phase": phase,
        "instance_type": instance_type,
        "usd_per_hour": usd_per_hour,
        "results_dir": str(directory),
    }

    stored: list[str] = []
    store = open_store(hf_repo) if hf_repo else None
    if store is not None and store.active:
        uploads = [
            path
            for path in run.files
            if path.suffix in upload_suffixes and path.stat().st_size < 2_000_000_000
        ]
        stored = store.upload_all(uploads, prefix=f"gate/{phase}/{directory.name}")

    logged = False
    url = None
    reason = None
    with start_run(
        f"gate-{phase}",
        config,
        project=project,
        tags=["brev", "gate", phase],
        motion_root=motion_root,
    ) as handle:
        logged = handle.active
        url = handle.url
        reason = handle.reason
        if handle.active:
            summary: dict[str, Any] = {}
            for key, value in run.extrapolation.items():
                if isinstance(value, (int, float)):
                    summary[f"probe/{key}"] = value
            for key, value in run.eval_metrics.items():
                if isinstance(value, (int, float)):
                    summary[f"eval/{key}"] = value
            for key, value in run.local_scores.items():
                if isinstance(value, (int, float)):
                    summary[f"score/{key}"] = value
            summary["artifacts/hf_repo"] = store.repo_id if store and store.active else None
            summary["artifacts/files"] = len(run.files)
            handle.summary(summary)
            if wall_hours is not None and usd_per_hour is not None:
                handle.log_cost(wall_hours, usd_per_hour)

    return {
        "what": f"observability record for the {phase} gate run",
        "phase": phase,
        "instance_type": instance_type,
        "usd_per_hour": usd_per_hour,
        "wall_hours": wall_hours,
        "usd": run.usd,
        "files_found": [str(p.relative_to(directory)) for p in run.files],
        "extrapolation": run.extrapolation,
        "eval_metrics": run.eval_metrics,
        "local_scores": run.local_scores,
        "wandb": {"logged": logged, "url": url, "project": project, "reason": reason},
        "huggingface": {
            "repo": store.repo_id if store and store.active else None,
            "uploaded": stored,
            "reason": store.reason if store else "not requested",
        },
    }
