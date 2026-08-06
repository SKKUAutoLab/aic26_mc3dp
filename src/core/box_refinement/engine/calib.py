
from __future__ import annotations

import json
import os
import re

import numpy as np
import open3d as o3d


def load_calibration(calib_path: str) -> dict:
    with open(calib_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sensor_map(calib: dict) -> dict:
    """camera id -> sensor dict, for the ``type == "camera"`` sensors only."""
    return {s["id"]: s for s in calib.get("sensors", []) if s.get("type") == "camera"}


def to_4x4_extrinsic(E):
    """SmartSpaces 3x4 [R|t] world-to-camera -> 4x4 homogeneous (DA3 wants [N, 4, 4])."""
    E = np.asarray(E, dtype=np.float32)
    if E.shape == (4, 4):
        return E
    if E.shape == (3, 4):
        E4 = np.eye(4, dtype=np.float32)
        E4[:3, :4] = E
        return E4
    raise ValueError(f"Invalid extrinsic shape: {E.shape}")


def to_4x4(E):
    """Same conversion in float64, for the geometry/fusion side."""
    E = np.asarray(E, dtype=np.float64)
    if E.shape == (4, 4):
        return E
    if E.shape == (3, 4):
        E4 = np.eye(4, dtype=np.float64)
        E4[:3, :4] = E
        return E4
    raise ValueError(E.shape)


def extract_camera_id_from_filename(path: str) -> str | None:
    m = re.search(r"(Camera_\d+)", os.path.basename(path))
    return m.group(1) if m else None


def camera_sort_key(path: str):
    cid = extract_camera_id_from_filename(path) or ""
    m = re.search(r"(\d+)", cid)
    return int(m.group(1)) if m else 0


def build_pcd_from_npz(npz_path, conf_percentile=60, stride=4):
    """Back-project every camera's depth in ``results.npz`` into ONE world-frame coloured cloud.

    ``conf_percentile`` drops the least-confident pixels (0 = keep all); ``stride`` subsamples the
    pixel grid. The depths must come from a DA3 run with ``align_to_input_ext_scale=True``, or the
    cloud is not metric and nothing downstream is valid.
    """
    d = np.load(npz_path)
    depth, conf = d["depth"], d.get("conf")
    K, E = d["intrinsics"], d["extrinsics"]
    image = d.get("image")

    pts_all, cols_all = [], []
    N, H, W = depth.shape
    for i in range(N):
        dep = depth[i]
        Ki = np.asarray(K[i], dtype=np.float64)
        Ei = to_4x4(E[i])

        us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
        us, vs = us.ravel(), vs.ravel()
        z = dep[vs, us]

        keep = z > 0
        if conf is not None and conf_percentile and conf_percentile > 0:
            c = conf[i][vs, us]
            keep &= c >= np.percentile(c[z > 0], conf_percentile) if np.any(z > 0) else keep
        if not np.any(keep):
            continue
        us, vs, z = us[keep], vs[keep], z[keep]

        pix = np.stack([us, vs, np.ones_like(us)], axis=0).astype(np.float64)
        cam = (np.linalg.inv(Ki) @ pix) * z                      # p_cam = depth * inv(K) @ [u,v,1]
        world = (np.linalg.inv(Ei) @ np.vstack([cam, np.ones((1, cam.shape[1]))]))[:3].T
        pts_all.append(world)

        if image is not None:
            img = image[i]
            if img.dtype != np.uint8:                            # DA3 may hand back float [0,1]
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            cols_all.append(img[vs, us].astype(np.float64) / 255.0)

    pcd = o3d.geometry.PointCloud()
    if pts_all:
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts_all, axis=0))
        if cols_all:
            pcd.colors = o3d.utility.Vector3dVector(np.concatenate(cols_all, axis=0))
    return pcd
