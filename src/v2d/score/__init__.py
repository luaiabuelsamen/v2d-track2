"""Scoring, from the organisers' own metric code outwards.

`track2` wraps `metric_code/track_2/*.py` from the submission kit after sha256-checking each
file against `KIT_MANIFEST.json`, so a local number is the host's by construction rather than by
reimplementation. `contact` extracts hand-object contact coverage from the retargeter's logs,
which is the training signal the scored metrics do not measure. `prereg` applies the
pre-registered decision rules mechanically.
"""
