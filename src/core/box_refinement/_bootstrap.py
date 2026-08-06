
import os
import sys

# .../src/matching/core/box_refinement/_bootstrap.py -> .../src/matching/third_party
_THIRD_PARTY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party",
)

if os.path.isdir(_THIRD_PARTY) and _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)
