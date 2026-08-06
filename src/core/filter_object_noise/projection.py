
import numpy as np

IMG_W, IMG_H = 1920, 1080

MIN_HEIGHT_PX = 8      # below this a person is too small for the pose model to say anything
MIN_VIS_RATIO = 0.1    # fraction of the projected box that lands inside the image
CROP_MARGIN = 0.15     # widen the crop, so a slightly-off box still contains the whole person


def project_boxes(boxes, K, E):
    """Project N 3D boxes into one camera.

    Args:
        boxes: (N, 7) array of `(x, y, z, w, l, h, yaw)` in world meters.
        K: (3, 3) intrinsics.
        E: (3, 4) world-to-camera extrinsics.

    Returns:
        `(bbox, usable, height_px)` where `bbox` is (N, 4) `x1,y1,x2,y2` widened by `CROP_MARGIN`
        and clipped to the image, `usable` is (N,) bool, and `height_px` is (N,) the on-screen
        height of the tight box -- what the zone fallback ranks cameras by.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 4)), np.zeros(0, dtype=bool), np.zeros(0)

    x, y, z, w, l, h, yaw = boxes.T
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    hx, hy, hz = w / 2, l / 2, h / 2

    # the 8 corners, as signs on each half-extent
    sx = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=np.float64)
    sy = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=np.float64)
    sz = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float64)
    lx = sx[None, :] * hx[:, None]
    ly = sy[None, :] * hy[:, None]
    lz = sz[None, :] * hz[:, None]

    world = np.stack([
        cos_y[:, None] * lx - sin_y[:, None] * ly + x[:, None],
        sin_y[:, None] * lx + cos_y[:, None] * ly + y[:, None],
        lz + z[:, None],
    ], axis=-1).reshape(-1, 3)                                  # (N*8, 3)

    cam = (E[:, :3] @ world.T).T + E[:, 3]                      # world -> camera
    depth = cam[:, 2]
    in_front = (depth > 1e-5).reshape(n, 8).all(axis=1)         # any corner behind -> unusable

    u = (K[0, 0] * cam[:, 0] / np.maximum(depth, 1e-5) + K[0, 2]).reshape(n, 8)
    v = (K[1, 1] * cam[:, 1] / np.maximum(depth, 1e-5) + K[1, 2]).reshape(n, 8)

    x1, x2 = u.min(axis=1), u.max(axis=1)
    y1, y2 = v.min(axis=1), v.max(axis=1)
    cx1, cy1 = np.clip(x1, 0, IMG_W), np.clip(y1, 0, IMG_H)
    cx2, cy2 = np.clip(x2, 0, IMG_W), np.clip(y2, 0, IMG_H)

    full_area = np.maximum(x2 - x1, 1e-6) * np.maximum(y2 - y1, 1e-6)
    vis_ratio = ((cx2 - cx1) * (cy2 - cy1)) / full_area
    height_px = cy2 - cy1

    usable = (in_front & (height_px >= MIN_HEIGHT_PX) & (vis_ratio >= MIN_VIS_RATIO)
              & (cx2 > cx1) & (cy2 > cy1))

    mx = (x2 - x1) * CROP_MARGIN / 2
    my = (y2 - y1) * CROP_MARGIN / 2
    bbox = np.stack([
        np.clip(x1 - mx, 0, IMG_W), np.clip(y1 - my, 0, IMG_H),
        np.clip(x2 + mx, 0, IMG_W), np.clip(y2 + my, 0, IMG_H),
    ], axis=1)
    return bbox, usable, height_px
