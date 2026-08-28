"""Make the vendored DA3 importable without touching anything outside this package.

`third_party/depth_anything_3` uses absolute self-imports (`from depth_anything_3.cfg import ...`),
so it has to sit on `sys.path` as a top-level package. The project registers its other vendored
libraries (botsort, ultralytics, ...) in `pyproject.toml`, but this package deliberately does not
edit any existing project file, so it puts `third_party` on the path itself.

Import this module before anything that reaches for `depth_anything_3`.
"""
import os
import sys

# .../src/matching/core/box_refinement/_bootstrap.py -> .../src/matching/third_party
_THIRD_PARTY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party",
)

if os.path.isdir(_THIRD_PARTY) and _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)
