"""Parsing `brev search` and picking a GPU that actually meets FlashCHORD's VRAM floor.

This exists because the previous picker was an awk script that guessed which column was which,
and it guessed wrong. Against the live table it read `TARGET_DISK` (100) as the VRAM and
`CAPABILITY` (8.6) as the hourly price, so a 16 GB V100 sorted ahead of a 48 GB A6000 and the
dry run proposed a $7/h card that cannot run the job at all. Money and a wasted hour ride on
this, so it is parsed properly and tested.

Two traps in the real table, both load-bearing:

- **Columns are not tokens.** 12 of 343 rows have a space inside the GPU name ("RTX PRO Server
  6000"), so splitting on whitespace shifts every later field -- including the price -- on
  exactly the rows describing the largest cards. Columns are therefore sliced at the character
  offsets of the header's own tokens.
- **VRAM/GPU is the floor, not TOTAL_VRAM.** FlashCHORD needs >=46,000 MiB free on one device, so
  a 2x16 GB box advertising TOTAL_VRAM 32 is not a candidate and neither is 4x24.

There is no fallback. If the table cannot be parsed or nothing clears the floor, the caller gets
nothing and must not launch; guessing a default is how the original went wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

#: FlashCHORD wants >=46,000 MiB free, which in practice means a 48 GB card.
MIN_VRAM_GB: Final[float] = 46.0

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("TYPE", "VRAM/GPU", "TOTAL_VRAM", "$/HR")
_MONEY = re.compile(r"^\$?([0-9]+(?:\.[0-9]+)?)$")


@dataclass(frozen=True)
class Offer:
    """One purchasable instance type, as the live table describes it."""

    instance_type: str
    provider: str
    gpu: str
    count: int
    vram_per_gpu_gb: float
    total_vram_gb: float
    usd_per_hour: float
    #: vCPUs matter more than the price difference suggests. The same A6000 ships with 6 vCPUs
    #: from one provider and 28 from another, and FlashCHORD's startup is single-threaded and
    #: CPU-bound (scene build, convex decomposition, Warp kernel compilation), with ONNX policy
    #: inference on top. A 6-vCPU box was observed at 101 % of one core and 0 % GPU.
    vcpus: int = 0

    def meets(self, min_vram_gb: float = MIN_VRAM_GB) -> bool:
        """Does a single GPU on this instance clear the VRAM floor?"""
        return self.vram_per_gpu_gb >= min_vram_gb

    def describe(self) -> str:
        cpus = f", {self.vcpus} vCPU" if self.vcpus else ""
        return (
            f"{self.instance_type} ({self.gpu} x{self.count}, "
            f"{self.vram_per_gpu_gb:g} GB/GPU{cpus}, ${self.usd_per_hour:.2f}/h)"
        )


def _column_spans(header: str) -> dict[str, slice]:
    """Character spans for each header token, so a value containing spaces stays in its column."""
    spans: dict[str, slice] = {}
    matches = list(re.finditer(r"\S+", header))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(header) + 1024
        spans[match.group(0)] = slice(match.start(), end)
    return spans


def _number(cell: str) -> float | None:
    match = _MONEY.match(cell.strip().replace(",", ""))
    return float(match.group(1)) if match else None


def parse_search(text: str) -> tuple[list[Offer], list[str]]:
    """Parse `brev search` output into offers, plus the lines that could not be parsed.

    Unparsed lines are returned rather than dropped so a layout change is visible instead of
    silently shrinking the candidate set.
    """
    lines = text.splitlines()
    header = next(
        (line for line in lines if all(column in line for column in _REQUIRED_COLUMNS)), None
    )
    if header is None:
        return [], [line for line in lines if line.strip()]

    spans = _column_spans(header)
    offers: list[Offer] = []
    skipped: list[str] = []
    for line in lines:
        if line == header or not line.strip():
            continue
        cell = {name: line[span].strip() for name, span in spans.items()}
        vram = _number(cell.get("VRAM/GPU", ""))
        total = _number(cell.get("TOTAL_VRAM", ""))
        price = _number(cell.get("$/HR", ""))
        count = _number(cell.get("COUNT", "1"))
        vcpus = _number(cell.get("VCPUS", "0"))
        if not cell.get("TYPE") or vram is None or price is None:
            skipped.append(line)
            continue
        offers.append(
            Offer(
                instance_type=cell["TYPE"],
                provider=cell.get("PROVIDER", ""),
                gpu=cell.get("GPU", ""),
                count=int(count) if count else 1,
                vram_per_gpu_gb=vram,
                total_vram_gb=total if total is not None else vram,
                usd_per_hour=price,
                vcpus=int(vcpus) if vcpus else 0,
            )
        )
    return offers, skipped


def eligible(
    offers: list[Offer], min_vram_gb: float = MIN_VRAM_GB, min_vcpus: int = 0
) -> list[Offer]:
    """Offers clearing the VRAM floor, cheapest first, then by most vCPUs at equal price.

    Price stays the primary key, so this never spends more than it has to. The vCPU tie-break
    costs nothing and matters: among the $0.68/h A6000s the vCPU count varies 6 to 12 for the
    same money.
    """
    return sorted(
        (offer for offer in offers if offer.meets(min_vram_gb) and offer.vcpus >= min_vcpus),
        key=lambda offer: (offer.usd_per_hour, -offer.vcpus, offer.instance_type),
    )


def pick_cheapest(
    offers: list[Offer], min_vram_gb: float = MIN_VRAM_GB, min_vcpus: int = 0
) -> Offer | None:
    """The cheapest offer clearing the floor, or None. Never a card below it, never a guess."""
    ranked = eligible(offers, min_vram_gb, min_vcpus)
    return ranked[0] if ranked else None


def lookup(offers: list[Offer], instance_type: str) -> Offer | None:
    """One named instance type, for validating an explicit `--type` against the live table."""
    return next((offer for offer in offers if offer.instance_type == instance_type), None)


# ------------------------------------------------------------------ instance readiness
#: Statuses from which an instance will never become usable, so waiting is pointless.
DEAD_STATUSES: Final[tuple[str, ...]] = ("FAILED", "TERMINATED", "STOPPED", "DELETING", "DELETED")


def is_shell_ready(row: str) -> bool:
    """Is this `brev ls` row ready to accept SSH?

    `STATUS=RUNNING` is not the question, and assuming it was cost a probe run: brev reports
    RUNNING while the SHELL column still reads NOT READY, and ssh then fails with
    `kex_exchange_identification: Connection closed by remote host`. The SHELL column is the one
    that matters -- and because "NOT READY" contains "READY", it has to be excluded first.
    """
    upper = row.upper()
    if "NOT READY" in upper:
        return False
    return "RUNNING" in upper and "READY" in upper


def is_dead(row: str) -> bool:
    """Has this instance reached a state it will not come back from?"""
    return any(status in row.upper() for status in DEAD_STATUSES)
