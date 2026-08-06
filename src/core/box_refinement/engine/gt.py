
from __future__ import annotations

import json

import numpy as np

try:
    import ijson  # type: ignore

    _HAS_IJSON = True
except Exception:  # noqa: BLE001
    _HAS_IJSON = False

# Edge index pairs for an 8-corner box (used by the frontend too).
BOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
    [4, 5], [5, 6], [6, 7], [7, 4],  # top
    [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
]


def _euler_to_R(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(roll) @ Rx(pitch). For ground objects only yaw matters."""
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(roll), np.sin(roll)
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _box_corners(center, size, rot) -> list[list[float]]:
    w, l, h = float(size[0]), float(size[1]), float(size[2])
    hx, hy, hz = w / 2.0, l / 2.0, h / 2.0
    local = np.array(
        [
            [-hx, -hy, -hz],
            [+hx, -hy, -hz],
            [+hx, +hy, -hz],
            [-hx, +hy, -hz],
            [-hx, -hy, +hz],
            [+hx, -hy, +hz],
            [+hx, +hy, +hz],
            [-hx, +hy, +hz],
        ]
    )
    R = _euler_to_R(float(rot[0]), float(rot[1]), float(rot[2]))
    world = (R @ local.T).T + np.asarray(center, dtype=np.float64)
    return world.tolist()


def _objects_for_frame(gt_path: str, frame_id: int) -> list:
    key = str(frame_id)
    if _HAS_IJSON:
        with open(gt_path, "rb") as f:
            for k, v in ijson.kvitems(f, ""):
                if k == key:
                    return v
        return []
    # Fallback: full load (slow, memory heavy) only if ijson unavailable.
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(key, [])


def get_gt_boxes(gt_path: str, frame_id: int) -> dict:
    """Return GT 3D boxes for one frame, ready for the viewer."""
    raw = _objects_for_frame(gt_path, frame_id)
    boxes = []
    for obj in raw:
        center = obj.get("3d location") or obj.get("3d_location")
        size = obj.get("3d bounding box scale") or obj.get("3d_bounding_box_scale")
        rot = obj.get("3d bounding box rotation") or obj.get(
            "3d_bounding_box_rotation"
        ) or [0.0, 0.0, 0.0]
        if center is None or size is None:
            continue
        boxes.append(
            {
                "object_type": obj.get("object type") or obj.get("object_type", "obj"),
                "object_id": obj.get("object id") or obj.get("object_id"),
                "center": [float(c) for c in center],
                "size": [float(s) for s in size],
                "yaw": float(rot[2]) if len(rot) > 2 else 0.0,
                "corners": _box_corners(center, size, rot),
            }
        )
    return {"frame_id": frame_id, "count": len(boxes), "edges": BOX_EDGES, "boxes": boxes}
