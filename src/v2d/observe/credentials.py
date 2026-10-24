"""Finding W&B and Hugging Face credentials that are already on the machine.

Environment variables are the documented path, but on this box both services were logged in
interactively instead: W&B writes a machine entry to `~/.netrc` and the Hugging Face CLI caches
a token under `~/.cache/huggingface/`. Checking only the environment would report "not
configured" on a machine where both are in fact ready, so all the usual locations are consulted.

Nothing here returns, logs or prints a secret -- only whether one exists and where it came from.
"""

from __future__ import annotations

import netrc
import os
from pathlib import Path
from typing import Final

WANDB_HOSTS: Final[tuple[str, ...]] = ("api.wandb.ai", "wandb.ai")
WANDB_ENV: Final[tuple[str, ...]] = ("WANDB_API_KEY",)
HF_ENV: Final[tuple[str, ...]] = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
HF_TOKEN_FILES: Final[tuple[str, ...]] = (
    "~/.cache/huggingface/token",
    "~/.huggingface/token",
)


def _env(names: tuple[str, ...]) -> str | None:
    """The first of `names` set to something non-empty in the environment."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return name
    return None


def _netrc_host() -> str | None:
    """A W&B host with an entry in `~/.netrc`, if the file is readable and names one."""
    path = Path("~/.netrc").expanduser()
    if not path.is_file():
        return None
    try:
        hosts = netrc.netrc(str(path)).hosts
    except (netrc.NetrcParseError, OSError):
        return None
    return next((host for host in WANDB_HOSTS if host in hosts), None)


def _hf_token_file() -> str | None:
    """A cached Hugging Face token file with content, if one exists."""
    for candidate in HF_TOKEN_FILES:
        path = Path(candidate).expanduser()
        if path.is_file() and path.read_text().strip():
            return candidate
    return None


def wandb_available() -> bool:
    """Is there a W&B credential and the library to use it?"""
    import importlib.util

    if importlib.util.find_spec("wandb") is None:
        return False
    return _env(WANDB_ENV) is not None or _netrc_host() is not None


def hf_available() -> bool:
    """Is there a Hugging Face credential and the library to use it?"""
    import importlib.util

    if importlib.util.find_spec("huggingface_hub") is None:
        return False
    return _env(HF_ENV) is not None or _hf_token_file() is not None


def which_credentials() -> dict[str, str | None]:
    """Where each credential was found, for logging. Never the credential itself."""
    return {
        "wandb": (
            f"env:{_env(WANDB_ENV)}"
            if _env(WANDB_ENV)
            else (f"netrc:{_netrc_host()}" if _netrc_host() else None)
        ),
        "huggingface": (
            f"env:{_env(HF_ENV)}"
            if _env(HF_ENV)
            else (f"file:{_hf_token_file()}" if _hf_token_file() else None)
        ),
    }
