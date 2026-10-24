"""A W&B run that degrades to a no-op instead of failing.

What a Brev run is required to record, per the project's observability rule: its config, the git
sha it ran from, the provenance of the reference it trained against, learning curves, per-clip
scores, and cost in dollars and GPU-hours. `Run` takes all of those; without a credential every
method still works and does nothing.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Final

from .credentials import wandb_available, which_credentials

PROJECT: Final[str] = "v2d-track2"


def git_sha(repo: Path | None = None) -> str | None:
    """The commit this code is running from, or None outside a git tree."""
    try:
        done = subprocess.run(
            ["git", "-C", str(repo or Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def git_dirty(repo: Path | None = None) -> bool | None:
    """Whether the working tree had uncommitted changes, so a curve is never misattributed."""
    try:
        done = subprocess.run(
            [
                "git",
                "-C",
                str(repo or Path(__file__).resolve().parents[3]),
                "status",
                "--porcelain",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(done.stdout.strip())


def reference_provenance(start: Path | None) -> dict[str, Any] | None:
    """The `provenance.json` governing a motion tree, found by walking up from `start`.

    This is the field that says which cleaners produced the reference a run trained against. A
    learning curve without it cannot be compared to any other run.
    """
    if start is None:
        return None
    here = start if start.is_dir() else start.parent
    for directory in [here, *here.parents]:
        candidate = directory / "provenance.json"
        if candidate.is_file():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                return json.loads(candidate.read_text())
    return None


class Run:
    """A thin wrapper over a W&B run, or a no-op with the same surface."""

    def __init__(self, handle: Any | None, reason: str | None = None) -> None:
        self._handle = handle
        self.reason = reason

    @property
    def active(self) -> bool:
        return self._handle is not None

    @property
    def url(self) -> str | None:
        return getattr(self._handle, "url", None) if self._handle else None

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        """One point on a learning curve, or any scalar."""
        if self._handle is not None:
            self._handle.log(dict(metrics), step=step)

    def summary(self, values: Mapping[str, Any]) -> None:
        """Final figures: per-clip scores, cost, GPU-hours."""
        if self._handle is not None:
            self._handle.summary.update(dict(values))

    def log_cost(self, hours: float, usd_per_hour: float, gpus: int = 1) -> None:
        """Cost is a first-class metric here: the credit is finite and revocable."""
        self.summary(
            {
                "cost/gpu_hours": round(hours * gpus, 4),
                "cost/wall_hours": round(hours, 4),
                "cost/usd_per_hour": usd_per_hour,
                "cost/usd": round(hours * usd_per_hour, 4),
            }
        )

    def artifact(self, path: Path, name: str, kind: str = "results") -> None:
        """Attach a file, when one is small enough to belong in the run rather than in HF."""
        if self._handle is None:
            return
        import wandb

        item = wandb.Artifact(name=name, type=kind)
        item.add_file(str(path))
        self._handle.log_artifact(item)

    def finish(self) -> None:
        if self._handle is not None:
            self._handle.finish()
            self._handle = None


@contextlib.contextmanager
def start_run(
    name: str,
    config: Mapping[str, Any] | None = None,
    *,
    project: str = PROJECT,
    tags: list[str] | None = None,
    motion_root: Path | None = None,
    offline: bool = False,
) -> Iterator[Run]:
    """A W&B run for the duration of the block, or a no-op if W&B is unusable.

    `motion_root` is where the reference provenance is looked up, so every run carries which
    cleaners produced the data it saw.
    """
    if not wandb_available():
        yield Run(None, reason="no W&B credential found (checked env and ~/.netrc)")
        return

    import wandb

    payload: dict[str, Any] = {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "credential_source": which_credentials()["wandb"],
        **dict(config or {}),
    }
    provenance = reference_provenance(motion_root)
    if provenance is not None:
        payload["reference_provenance"] = provenance

    handle = None
    try:
        handle = wandb.init(
            project=project,
            name=name,
            config=payload,
            tags=tags or [],
            mode="offline" if offline or os.environ.get("WANDB_MODE") == "offline" else "online",
            reinit=True,
        )
        yield Run(handle)
    except Exception as error:
        yield Run(None, reason=f"W&B unavailable: {error}")
    finally:
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.finish()
