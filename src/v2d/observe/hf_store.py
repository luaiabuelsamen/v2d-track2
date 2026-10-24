"""Checkpoints and results to a PRIVATE Hugging Face repo, or nowhere at all.

The repo is created private and this module has no code path that makes one public or flips an
existing one: `private=True` is passed on creation and `open_store` refuses a `private=False`
caller outright. Weights trained on challenge data and per-clip scores before a submission are
not things to publish by accident.

Like the W&B wrapper, every method is a no-op without a credential.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .credentials import hf_available, which_credentials

DEFAULT_REPO: Final[str] = "v2d-track2"


@dataclass
class HfStore:
    """A handle on one private repo, or an inert placeholder."""

    repo_id: str | None
    api: Any | None
    reason: str | None = None

    @property
    def active(self) -> bool:
        return self.api is not None and self.repo_id is not None

    @property
    def url(self) -> str | None:
        return f"https://huggingface.co/{self.repo_id}" if self.active else None

    def upload(self, path: Path, destination: str | None = None) -> str | None:
        """Put one file in the repo, returning its path there."""
        if not self.active:
            return None
        target = destination or path.name
        with contextlib.suppress(Exception):  # artefact storage must never fail a GPU run
            self.api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=target,
                repo_id=self.repo_id,
                repo_type="model",
            )
            return target
        return None

    def upload_all(self, paths: Iterable[Path], prefix: str = "") -> list[str]:
        """Put several files in the repo under an optional prefix."""
        stored = []
        for path in paths:
            name = f"{prefix.rstrip('/')}/{path.name}" if prefix else path.name
            if self.upload(path, name):
                stored.append(name)
        return stored


def open_store(
    repo: str = DEFAULT_REPO, *, owner: str | None = None, private: bool = True, create: bool = True
) -> HfStore:
    """A private repo handle, creating the repo if it does not exist.

    `private=False` is rejected rather than honoured. Making this repo public is not a thing this
    code is allowed to do, so there is no argument value that does it.
    """
    if not private:
        raise ValueError(
            "this store is private-only by design; publishing checkpoints or pre-submission "
            "scores is not something this code does"
        )
    if not hf_available():
        return HfStore(
            None, None, reason="no Hugging Face credential found (checked env and the cached token)"
        )

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        who = api.whoami()
        namespace = owner or who.get("name")
        repo_id = repo if "/" in repo else f"{namespace}/{repo}"
        if create:
            api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        if getattr(info, "private", True) is False:
            return HfStore(None, None, reason=f"{repo_id} exists and is PUBLIC; refusing to use it")
    except Exception as error:
        return HfStore(None, None, reason=f"Hugging Face unavailable: {error}")
    return HfStore(repo_id, api)


def describe() -> dict[str, str | None]:
    """Where credentials were found, for a run's config. Never the credentials."""
    return which_credentials()
