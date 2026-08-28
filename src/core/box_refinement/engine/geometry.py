"""Geometry helpers shared by both scenes.

``bev_iou`` used to live inside the mean-shift module, which meant the synthetic scene had to reach
into the real-world scene to borrow it. It is plain rotated-rectangle geometry with no scene
knowledge, so it belongs here and the dependency runs one way: scenes -> engine.
"""
import cv2
import numpy as np


def bev_iou(b1, b2):
    """IoU of two BEV (ground-plane) rectangles ``b = (cx, cy, w, l, yaw_rad)``."""
    r1 = ((float(b1[0]), float(b1[1])), (float(b1[2]), float(b1[3])), float(np.degrees(b1[4])))
    r2 = ((float(b2[0]), float(b2[1])), (float(b2[2]), float(b2[3])), float(np.degrees(b2[4])))
    try:
        ret, region = cv2.rotatedRectangleIntersection(r1, r2)
    except Exception:  # noqa: BLE001
        return 0.0
    if ret == 0 or region is None or len(region) < 3:
        return 0.0
    inter = float(cv2.contourArea(region))
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return float(inter / union) if union > 0 else 0.0
