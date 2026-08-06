

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .types import LocalizationObservation, Track1Row


def observations_from_cam_observations(
    observations: Sequence[Any],
    camera_ids: Sequence[str],
    frame_id: int,
    ankle_conf_threshold: float = 0.8,
    ankle_edge_margin_px: float = 20.0,
) -> list[LocalizationObservation]:
    """Convert existing ``CamObservation`` objects without reading any files."""

    converted: list[LocalizationObservation] = []
    for observation in observations:
        camera_index = int(observation.cam_idx)
        if camera_index < 0 or camera_index >= len(camera_ids):
            raise IndexError(
                f"cam_idx={camera_index} is outside camera_ids (size={len(camera_ids)})"
            )
        xy = np.asarray(observation.world_xy, dtype=np.float64).reshape(2)
        confidence = observation.confidence
        explicit_edge_reason = getattr(
            observation, "reference_edge_reason", None
        )
        edge_reason = explicit_edge_reason or _ankle_edge_reason(
            getattr(observation, "pose_keypoints", None),
            getattr(observation, "image_size", None),
            confidence_threshold=ankle_conf_threshold,
            edge_margin_px=ankle_edge_margin_px,
        )
        converted.append(
            LocalizationObservation(
                camera_id=str(camera_ids[camera_index]),
                frame_id=int(frame_id),
                class_id=int(observation.class_id),
                world_xy=(float(xy[0]), float(xy[1])),
                confidence=1.0 if confidence is None else float(confidence),
                local_track_id=observation.local_track_id,
                embedding=getattr(observation, "embedding", None),
                visibility=getattr(observation, "visibility", None),
                reference_edge_reason=edge_reason,
            )
        )
    return converted


def observation_from_world_detection(detection: Any) -> LocalizationObservation:
    """Convert a pixel2world-style ``WorldDetection`` by public attributes."""

    point = np.asarray(detection.world_point, dtype=np.float64).reshape(-1)
    if point.size < 2:
        raise ValueError("world_point must have at least x and y")
    return LocalizationObservation(
        camera_id=str(detection.camera_id),
        frame_id=int(detection.frame_id),
        class_id=int(detection.class_id),
        world_xy=(float(point[0]), float(point[1])),
        confidence=float(detection.confidence),
        local_track_id=getattr(detection, "local_track_id", None),
        embedding=getattr(detection, "embedding", None),
        visibility=getattr(detection, "visibility", None),
        reference_edge_reason=getattr(detection, "reference_edge_reason", None),
    )


def camera_centers_from_calibration(
    calibrations: Mapping[str, Any],
) -> dict[str, tuple[float, float, float]]:
    """Build world camera centres from AI City world-to-camera extrinsics."""

    centers: dict[str, tuple[float, float, float]] = {}
    for camera_id, calibration in calibrations.items():
        if hasattr(calibration, "camera_center"):
            center = np.asarray(calibration.camera_center, dtype=np.float64).reshape(3)
        else:
            if not isinstance(calibration, Mapping):
                raise TypeError(
                    f"Unsupported calibration value for {camera_id}: "
                    f"{type(calibration).__name__}"
                )
            if "extrinsicMatrix" in calibration:
                extrinsic = np.asarray(
                    calibration["extrinsicMatrix"], dtype=np.float64
                ).reshape(3, 4)
                rotation = extrinsic[:, :3]
                translation = extrinsic[:, 3]
            elif "R" in calibration and "t" in calibration:
                rotation = np.asarray(calibration["R"], dtype=np.float64).reshape(3, 3)
                translation = np.asarray(calibration["t"], dtype=np.float64).reshape(3)
            else:
                raise KeyError(
                    f"{camera_id} calibration needs camera_center, "
                    "extrinsicMatrix, or R/t"
                )
            center = -rotation.T @ translation
        centers[str(camera_id)] = tuple(float(value) for value in center)
    return centers


def track1_row_from_value(value: Any) -> Track1Row:
    """Normalize a canonical row or the dict shape used by the MTMC writer."""

    if isinstance(value, Track1Row):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "rows must be Track1Row or mapping objects; "
            f"got {type(value).__name__}"
        )

    x_key, y_key = _xy_keys(value)
    z_key = "z" if "z" in value else "location_z"
    return Track1Row(
        scene_id=int(value["scene_id"]),
        class_id=int(value["class_id"]),
        object_id=int(value["object_id"]),
        frame_id=int(value["frame_id"]),
        x=float(value[x_key]),
        y=float(value[y_key]),
        z=float(value[z_key]),
        width=float(value["width"]),
        length=float(value["length"]),
        height=float(value["height"]),
        yaw=float(value["yaw"]),
    )


def replace_row_xy(value: Any, x: float, y: float) -> Any:
    """Return a new row of the same public shape, changing only x/y."""

    if isinstance(value, Track1Row):
        return value.with_xy(x, y)
    if not isinstance(value, Mapping):
        raise TypeError(
            "rows must be Track1Row or mapping objects; "
            f"got {type(value).__name__}"
        )
    x_key, y_key = _xy_keys(value)
    copied = dict(value)
    copied[x_key] = float(x)
    copied[y_key] = float(y)
    return copied


def _xy_keys(value: Mapping[str, Any]) -> tuple[str, str]:
    if "x" in value and "y" in value:
        return "x", "y"
    if "location_x" in value and "location_y" in value:
        return "location_x", "location_y"
    raise KeyError("row mapping needs x/y or location_x/location_y")


def _ankle_edge_reason(
    pose_keypoints: Any,
    image_size: Any,
    *,
    confidence_threshold: float,
    edge_margin_px: float,
) -> str | None:
    """Return the legacy before-row fallback reason from current-frame pose."""

    if image_size is None:
        return None
    try:
        height, width = int(image_size[0]), int(image_size[1])
    except (TypeError, ValueError, IndexError):
        return None
    if height <= 0 or width <= 0:
        return None
    if not isinstance(pose_keypoints, (list, tuple)):
        return "no ankle"

    confident_ankles = 0
    ankle_near_edge = False
    for index in (15, 16):
        if index >= len(pose_keypoints):
            continue
        keypoint = pose_keypoints[index]
        try:
            if isinstance(keypoint, Mapping):
                x = float(keypoint["x"])
                y = float(keypoint["y"])
                score = float(keypoint.get("score", 1.0))
            else:
                x = float(keypoint[0])
                y = float(keypoint[1])
                score = float(keypoint[2]) if len(keypoint) >= 3 else 1.0
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if score < confidence_threshold or not np.isfinite((x, y)).all():
            continue
        confident_ankles += 1
        outside = x < 0 or x >= width or y < 0 or y >= height
        if not outside and (
            x <= edge_margin_px
            or x >= width - 1 - edge_margin_px
            or y <= edge_margin_px
            or y >= height - 1 - edge_margin_px
        ):
            ankle_near_edge = True

    # Preserve the old TargetResolver priority exactly.
    if confident_ankles == 0:
        return "no ankle"
    if confident_ankles == 1:
        return "1 ankle"
    if ankle_near_edge:
        return "ankle edge"
    return None
