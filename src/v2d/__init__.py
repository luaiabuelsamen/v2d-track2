"""Track 2 of the NVIDIA V2D Challenge: robotic grounding on a Unitree G1 + Dex3.

The pipeline, end to end:

    released episode  ->  clean the Tier 2 reference  ->  retarget to G1  ->  score / package

`v2d --help` lists every stage. Each stage writes a JSON record under `results/`, and every
number quoted in the README or in docs/ is reproducible from one of those commands.
"""

__version__ = "0.1.0"
