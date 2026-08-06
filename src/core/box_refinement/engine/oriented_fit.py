
import numpy as np


def _best_offset(coords, width, slide, step=0.05):
    """Offset d in [-slide, slide] that puts the most points inside [d-width/2, d+width/2]."""
    if len(coords) == 0:
        return 0.0
    ds = np.arange(-slide, slide + 1e-9, step)
    half = width / 2.0
    # counts[i] = #{|coords - ds[i]| < half}
    counts = ((np.abs(coords[None, :] - ds[:, None]) < half).sum(axis=1))
    return float(ds[int(np.argmax(counts))])


def fit_box(P, seed_xy, ref_yaw, w, l, h, floor_z=0.0, *,
            search_margin=0.7, lift=0.5, floor_drop=0.12, ceil_extra=0.0,
            yaw_window=1.6, yaw_step=0.04, pos_slide=0.45, min_pts=40, yaw_bias=0.15):
    """Fit a fixed-size oriented box to the floated point cluster of one Nova/Transporter.

    Returns dict(x, y, z, yaw, fit, body) or None if too few points (caller keeps prev/submission).
    - search_margin : ROI half-width beyond the footprint, around seed_xy (continuity anchor)
    - lift          : raise the z ceiling by this much above the box top (catch DA3-floated points)
    - floor_drop    : drop points below floor_z + this (kills the horizontal floor layer)
    - ceil_extra    : extra ceiling cut for tall noise (Transporter shelf) — ceiling = floor + h + lift - ceil_extra
    - yaw_window    : search yaw within +/- this of ref_yaw (~pi/2 covers all unique rect orientations)
    - pos_slide     : how far the footprint may slide off seed to chase the cluster
    - yaw_bias      : penalty per radian of |yaw - ref_yaw| (favours continuity over a marginally
                      denser but rotated blob from another camera)
    """
    cx0, cy0 = float(seed_xy[0]), float(seed_xy[1])
    R = max(w, l) / 2.0 + search_margin
    z_lo = floor_z + floor_drop
    z_hi = floor_z + h + lift - ceil_extra
    sel = ((np.abs(P[:, 0] - cx0) < R) & (np.abs(P[:, 1] - cy0) < R)
           & (P[:, 2] > z_lo) & (P[:, 2] < z_hi))
    body = P[sel][:, :2]
    if len(body) < min_pts:
        return None
    bx = body[:, 0] - cx0
    by = body[:, 1] - cy0
    best = None
    for dyaw in np.arange(-yaw_window, yaw_window + 1e-9, yaw_step):
        yaw = ref_yaw + dyaw
        c, s = np.cos(-yaw), np.sin(-yaw)
        u = bx * c - by * s            # along w
        v = bx * s + by * c            # along l
        du = _best_offset(u, w, pos_slide)
        dv = _best_offset(v, l, pos_slide)
        cnt = int(((np.abs(u - du) < w / 2.0) & (np.abs(v - dv) < l / 2.0)).sum())
        score = cnt * (1.0 - yaw_bias * abs(dyaw))      # bias toward the reference yaw
        if best is None or score > best[0]:
            wx = cx0 + du * np.cos(yaw) - dv * np.sin(yaw)
            wy = cy0 + du * np.sin(yaw) + dv * np.cos(yaw)
            best = (score, cnt, wx, wy, yaw)
    _, cnt, wx, wy, yaw = best
    # normalise yaw to the representative within +/-pi/2 of ref_yaw (box is symmetric under +pi)
    while yaw - ref_yaw > np.pi / 2:
        yaw -= np.pi
    while yaw - ref_yaw < -np.pi / 2:
        yaw += np.pi
    return {"x": wx, "y": wy, "z": floor_z, "yaw": float(yaw), "fit": cnt, "body": int(len(body))}
