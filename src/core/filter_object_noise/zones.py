
import glob
import json
import os

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

ZONE_BUFFER_M = 1.0    # grow each polygon, so someone standing on a boundary is still covered
FALLBACK_CAMS = 2      # a point outside every zone falls back to this many best-looking cameras


def _map_to_world(points, scale, translation, map_height):
    pts = np.asarray(points, dtype=np.float64)
    return np.column_stack([
        pts[:, 0] / scale - translation["x"],
        (map_height - pts[:, 1]) / scale - translation["y"],
    ])


def load_zones(zone_dir: str, calibration: dict, cameras) -> dict:
    """camera id -> prepared world-space polygons.

    Only cameras present in `cameras` are kept. That filter is load-bearing: the zone directory also
    holds `Camera_combined.json`, a merged overlay that is not a real camera, and letting it through
    would put a camera id into the assignment that has no calibration and no video.
    """
    sensors = [s for s in calibration["sensors"] if s.get("type") == "camera"]
    if not sensors:
        raise ValueError("calibration has no cameras")
    scale = float(sensors[0]["scaleFactor"])
    translation = sensors[0]["translationToGlobalCoordinates"]

    known = set(cameras)
    zones = {}
    for path in sorted(glob.glob(os.path.join(zone_dir, "Camera_*.json"))):
        cam = os.path.basename(path)[:-len(".json")]
        if cam not in known:
            continue
        data = json.load(open(path, encoding="utf-8"))
        map_height = float(data.get("imageHeight") or 1080)
        polygons = []
        for shape in data.get("shapes", []):
            if shape.get("label") != cam:
                continue
            poly = Polygon(_map_to_world(shape["points"], scale, translation, map_height))
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty:
                polygons.append(poly.buffer(ZONE_BUFFER_M))
        if polygons:
            zones[cam] = [prep(p) for p in polygons]
    return zones


def assign_cameras(zones, xy, usable_by_cam, height_by_cam, cameras):
    """For each detection, the cameras that should look at it.

    A detection is assigned to every camera whose zone contains it and that can actually see it. A
    detection outside every zone -- rare, and usually a box that has drifted off the floor plan --
    falls back to the cameras that see it biggest, so it still gets judged rather than silently kept.

    Args:
        zones: from :func:`load_zones`.
        xy: (N, 2) world positions.
        usable_by_cam: camera id -> (N,) bool from the projection.
        height_by_cam: camera id -> (N,) on-screen height, used to rank the fallback.
        cameras: every camera id, in a stable order.

    Returns:
        `(assigned, n_fallback)` where `assigned` is a list of N camera-id lists.
    """
    zone_cams = sorted(zones)
    assigned, n_fallback = [], 0

    for i in range(len(xy)):
        point = Point(float(xy[i, 0]), float(xy[i, 1]))
        cams = [c for c in zone_cams
                if usable_by_cam[c][i] and any(z.contains(point) for z in zones[c])]
        if not cams:
            seen = sorted(((height_by_cam[c][i], c) for c in cameras if usable_by_cam[c][i]),
                          reverse=True)
            cams = [c for _, c in seen[:FALLBACK_CAMS]]
            n_fallback += 1
        assigned.append(cams)
    return assigned, n_fallback
