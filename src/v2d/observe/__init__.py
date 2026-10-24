"""Observability for Brev runs: Weights & Biases for metrics, Hugging Face for artefacts.

Both are **optional and fail-soft**. A run must never break, stall or change behaviour because a
credential is missing or a service is unreachable -- GPU minutes are the expensive thing here,
not telemetry. Every entry point in this package returns a working no-op object when its
credential is absent, so callers need no conditionals.

Credentials are discovered, never created. This package does not call `wandb login` or
`hf auth login`, and it will not print or copy a token.
"""

from .credentials import hf_available, wandb_available, which_credentials
from .hf_store import HfStore, open_store
from .wandb_run import Run, start_run

__all__ = [
    "HfStore",
    "Run",
    "hf_available",
    "open_store",
    "start_run",
    "wandb_available",
    "which_credentials",
]
