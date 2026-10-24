"""Submission packaging, built on the kit's own packer rather than a reimplementation of it.

See `docs/SUBMISSION.md`: the path is proven end to end, and it deliberately cannot emit a
submittable CSV from the kinematic retarget, whose output reproduces the released input to
0.000085 mm and would therefore hand the scoring reference back as an entry.
"""
