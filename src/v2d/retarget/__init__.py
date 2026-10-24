"""Retargeting a released episode onto the Unitree G1 + Dex3.

`sequences` converts a released parquet into the folder layout the official `soma_to_g1.py`
expects, `batch` drives it over every clip with the resource floors this machine needs,
`validate` checks each result against its input, and `qa` aggregates the retargeter's own
diagnostics.

The validation bar is strict on purpose: one rigid transform must explain each retargeted object
trajectory exactly, because that is precisely what the submission packer has to undo. All 60
Tier-1/2 clips pass at 0.0 mm residual.
"""
