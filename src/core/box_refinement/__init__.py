"""Point-cloud-guided 3D box refinement for Track 1.

Takes the submission produced by the multi-camera tracking stage, rebuilds a metric point cloud
per frame with Depth Anything 3, and refines each box's footprint and yaw against that cloud. The
track ids, the class ids and the frame ids are passed through untouched, so detection and
association are unchanged: only the geometry of the boxes moves.

Two ways in:

* the CLI -- `refine_scene25` / `refine_scene27`: a submission `.txt` goes in, a refined one comes
  out. This owns the frame loop.
* `BoxRefiner` -- one call per frame, so a pipeline that already iterates frames keeps its own loop.
"""

from . import _bootstrap  # noqa: F401  (puts third_party/ on sys.path for DA3; must be first)
from .refiner import BoxRefiner

__all__ = ["BoxRefiner"]
