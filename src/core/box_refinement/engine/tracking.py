"""OC-SORT-inspired per-track stabiliser (IoU-only, keeps the submission object_id).

The submission already gives the track id, so we do NOT re-associate. Per object_id we run a
constant-velocity (alpha-beta) filter on the BEV centre and use BEV-IoU between each frame's
refined box and the filter's PREDICTION as a robustness gate:

  * IoU >= iou_thresh  -> accept the detection, smooth it in (reduces jitter).
  * IoU <  iou_thresh  -> the refine JUMPED (wrong cluster) -> coast on the prediction.
  * long frame gap     -> observation-centric reset: trust the new detection, zero velocity
                          (OC-SORT's observation-centric recovery, avoids a stale-velocity drift).
  * N consecutive rejects -> accept the detection anyway (the object genuinely moved/teleported).

Only the x,y position is stabilised (size/yaw kept from the refine). No new frames are invented —
output is keyed by the frames the object already appears in. numpy-free; reuses ``_bev_iou``.
"""
from __future__ import annotations

import math

from .geometry import bev_iou as _bev_iou


def _yaw(o):
    if "yaw" in o and o["yaw"] is not None:
        return float(o["yaw"])                       # radians
    return math.radians(float(o.get("yaw_deg", 0.0)))


def track_sequence(obs, iou_thresh=0.2, smooth=0.5, max_gap=30, reset_after=3):
    """obs = list of {frame_id, x, y, w, l, yaw|yaw_deg} for ONE object_id (any order).
    Returns {frame_id: (x, y)} with the BEV centre stabilised."""
    obs = sorted(obs, key=lambda o: o["frame_id"])
    out = {}
    if not obs:
        return out
    alpha = max(0.05, min(1.0, 1.0 - float(smooth)))   # position gain (smooth=0 -> 1 = no smoothing)
    beta = alpha * alpha / (2.0 - alpha)               # critically-damped velocity gain
    x, y, vx, vy = float(obs[0]["x"]), float(obs[0]["y"]), 0.0, 0.0
    pf = obs[0]["frame_id"]
    out[pf] = (round(x, 4), round(y, 4))
    rejects = 0
    for o in obs[1:]:
        dt = max(1, int(o["frame_id"]) - int(pf))
        px, py = x + vx * dt, y + vy * dt              # predict
        ox, oy = float(o["x"]), float(o["y"])
        yaw, w, l = _yaw(o), float(o["w"]), float(o["l"])
        if dt > max_gap:                               # observation-centric recovery after a long gap
            x, y, vx, vy, rejects = ox, oy, 0.0, 0.0, 0
        else:
            iou = _bev_iou((ox, oy, w, l, yaw), (px, py, w, l, yaw))
            if iou >= iou_thresh or rejects >= reset_after:
                rx, ry = ox - px, oy - py              # accept + smooth
                x, y = px + alpha * rx, py + alpha * ry
                vx, vy = vx + beta * rx / dt, vy + beta * ry / dt
                rejects = 0
            else:                                      # reject a jump -> coast on the prediction
                x, y = px, py
                rejects += 1
        out[int(o["frame_id"])] = (round(x, 4), round(y, 4))
        pf = o["frame_id"]
    return out
