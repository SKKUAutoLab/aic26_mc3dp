"""Apply per-camera 'zone' polygons to a DA3 export → drop depth INSIDE the zones.

Scene 27's calibration (K/E) is imperfect → strong distortion in some image regions
(e.g. the shelving). The user hand-labels those regions as polygons in
``{scene}/zone/Camera_XXXX.json`` (LabelMe format, full-image pixel coords). DA3 must
run on the whole image, so instead we **zero out the depth inside the zone polygons**
on the result and save it as a sibling export tagged ``_zone``. Fusing that export
keeps only the depth OUTSIDE the zones → fewer viewpoint-noise points.

``results.npz`` keys: ``image`` (N,H,W,3), ``depth`` (N,H,W), ``conf`` (N,H,W),
``intrinsics`` (N,3,3), ``extrinsics`` (N,3,4). The depth map (H,W) is the DA3
processing size, so polygons are scaled from each JSON's imageWidth/Height.
"""
from __future__ import annotations

import json
import os
import shutil
from glob import glob

import cv2
import numpy as np


def list_zone_dirs(scene_dir: str) -> list:
    """Names of zone folders in a scene (subdirs starting 'zone' that hold ≥1 *.json)."""
    out = []
    if not os.path.isdir(scene_dir):
        return out
    for name in sorted(os.listdir(scene_dir)):
        d = os.path.join(scene_dir, name)
        if name.startswith("zone") and os.path.isdir(d) and glob(os.path.join(d, "*.json")):
            out.append(name)
    return out


def load_zones(scene_dir: str, zone_name: str = "zone", zone_dir: str = "") -> dict:
    """camera_id -> list of (polygon Nx2 float in image px, (img_w, img_h)).

    ``zone_dir`` points straight at a directory of polygons and wins over ``scene_dir``/``zone_name``.
    The scene-27 polygons are a hand-refined asset that lives WITH this package, not in the dataset
    and not in the project's own ``regions.zone_*`` configs -- those are a different set of zones.
    """
    zdir = zone_dir or os.path.join(scene_dir, zone_name or "zone")
    out: dict[str, list] = {}
    if not os.path.isdir(zdir):
        return out
    for f in sorted(glob(os.path.join(zdir, "*.json"))):
        cid = os.path.splitext(os.path.basename(f))[0]
        try:
            j = json.load(open(f, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        iw, ih = int(j.get("imageWidth", 1920)), int(j.get("imageHeight", 1080))
        polys = []
        for s in j.get("shapes", []):
            if s.get("shape_type") == "polygon" and s.get("points"):
                polys.append((np.asarray(s["points"], dtype=np.float64), (iw, ih)))
        if polys:
            out[cid] = polys
    return out


def apply(export_dir: str, scene_dir: str, zone_name: str = "zone",
          out_dir: str | None = None, zone_dir: str = "") -> dict:
    """Write a sibling ``*_<zone_name>`` export with depth zeroed inside the zone polygons."""
    zone_name = zone_name or "zone"
    npz_path = os.path.join(export_dir, "exports", "npz", "results.npz")
    cam_path = os.path.join(export_dir, "metadata", "camera_ids.json")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"results.npz not found: {npz_path}")
    zones = load_zones(scene_dir, zone_name, zone_dir)
    if not zones:
        raise FileNotFoundError(
            f"no zone polygons under {zone_dir or os.path.join(scene_dir, zone_name)}")

    cam_ids = json.load(open(cam_path, encoding="utf-8"))["camera_ids"]
    data = dict(np.load(npz_path))
    depth = data["depth"]                      # (N, H, W)
    conf = data.get("conf")
    N, H, W = depth.shape

    n_masked, px_dropped = 0, 0
    for i, cid in enumerate(cam_ids):
        z = zones.get(cid)
        if not z:
            continue
        mask = np.zeros((H, W), dtype=np.uint8)
        for poly, (iw, ih) in z:
            p = np.round(poly * np.array([W / iw, H / ih])).astype(np.int32)
            cv2.fillPoly(mask, [p], 1)
        m = mask.astype(bool)
        depth[i][m] = 0.0                      # invalid depth → dropped by min-depth filter
        if conf is not None:
            conf[i][m] = 0.0
        n_masked += 1
        px_dropped += int(m.sum())

    data["depth"] = depth
    if conf is not None:
        data["conf"] = conf

    out_dir = out_dir or (export_dir.rstrip("/") + "_" + zone_name)
    os.makedirs(os.path.join(out_dir, "exports", "npz"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metadata"), exist_ok=True)
    np.savez(os.path.join(out_dir, "exports", "npz", "results.npz"), **data)

    # carry over the sidecar files fusion needs
    sky = os.path.join(export_dir, "exports", "npz", "sky.npy")
    if os.path.isfile(sky):
        shutil.copy(sky, os.path.join(out_dir, "exports", "npz", "sky.npy"))
    shutil.copy(cam_path, os.path.join(out_dir, "metadata", "camera_ids.json"))
    rc = os.path.join(export_dir, "metadata", "run_config.json")
    if os.path.isfile(rc):
        cfg = json.load(open(rc, encoding="utf-8"))
        cfg["zone_masked"] = True
        cfg["zone_cameras"] = n_masked
        json.dump(cfg, open(os.path.join(out_dir, "metadata", "run_config.json"), "w"),
                  indent=2)

    return {"zone_export_dir": out_dir, "cameras_masked": n_masked,
            "pixels_dropped": px_dropped, "cameras_total": N}
