# Online fisheye localization refinement

This package is the Warehouse 027 **localization** refiner. It is independent
from `matching/core/box_refinement` (the DA3 3D-box refiner).

Status: this commit provides the tested library API. The repository's current
MTMC runner still writes rows only after `tracker.finalize()`, so it does **not**
call this final-stage API yet. An online runner must expose current-frame rows
and invoke the call shown below immediately before writing them.

It is online in the AIC sense: one call consumes only the current frame's
camera observations and current Track 1 rows. It does not scan a dataset, open
videos, read/rewrite a submission, cache all frames, or use future frames.

## What it changes

- Cameras `Camera_0000`--`Camera_0003` are reference cameras.
- Only candidates from fisheye cameras `Camera_0004`--`Camera_0006` are refined.
- Cross-camera observations are grouped with BEV + optional ReID.
- The nearest reference camera by world-ray distance supplies the target when
  camera centres are passed. Without them, the deterministic fallback is the
  highest-confidence reference observation.
- Matching to output rows is deterministic, one-to-one, and uses a stricter
  gate in crowded areas.
- At most one row can be refined from one physical cross-camera group, even if
  that person appears in several fisheye cameras.
- Only person `x/y` may change. Scene/class/object/frame, `z`, size, height and
  yaw pass through unchanged.
- A far reference ray (default `>8 m`) or an ankle-edge warning keeps the
  current "before" row instead of moving it.

"Fisheye zone" here means the fisheye camera group `0004`--`0006`. Spatial
camera-zone filtering remains upstream; the refiner consumes observations that
the online detector/localizer has already accepted and never opens zone JSON.

The production path here is the tested reference-camera consensus swap. It
does not fit/apply a lens-distortion model itself. Optional batch-only features
from the old review app (manual regions, temporal rollback, visualization and
full-submission export) are intentionally outside this online module.

## Final-layer call

```python
from core.localization_refinement import (
    OnlineFisheyeLocalizationRefiner,
    camera_centers_from_calibration,
    observations_from_cam_observations,
)

refiner = OnlineFisheyeLocalizationRefiner(
    camera_centers_xyz=camera_centers_from_calibration(calibration_by_camera)
)

for frame_id, images in image_stream:
    # All three values below belong to this frame only.
    cam_observations = detector_and_localizer(images)
    current_rows = tracker.current_track1_rows(frame_id)
    observations = observations_from_cam_observations(
        cam_observations, camera_ids, frame_id
    )

    result = refiner.refine_frame(frame_id, observations, current_rows)
    writer.write_rows(result.rows)  # last stage
```

`tracker.current_track1_rows()` and `writer.write_rows()` above name the two
interfaces an online runner needs; they are integration placeholders, not
methods exposed by the current batch runner.

## Test

```bash
PYTHONPATH=src python -m pytest -q tests/test_online_localization_refiner.py
```
