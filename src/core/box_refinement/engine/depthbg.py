
from __future__ import annotations

import glob
import os

import numpy as np

from . import pipeline


def _to_4x4(E):
    M = np.eye(4, dtype=np.float64)
    M[:3, :4] = np.asarray(E, dtype=np.float64)
    return M


def _load(export_dir):
    d = np.load(os.path.join(export_dir, "exports", "npz", "results.npz"))
    return (d["depth"].astype(np.float32), d["conf"].astype(np.float32),
            d["intrinsics"].astype(np.float64), d["extrinsics"].astype(np.float64),
            d["image"])


def find_exports(output_root, split, scene, model_name, res, zone, frames=None, zone_name="zone"):
    """Export dirs (containing exports/npz/results.npz) for a scene at a given res/zone."""
    model_san = pipeline.sanitize_model_name(model_name)
    suffix = f"{model_san}_res_{res}" + pipeline.zone_suffix(zone, zone_name)
    out = []
    base = os.path.join(output_root, split, scene)
    pat = os.path.join(base, "frame_*", suffix, "exports", "npz", "results.npz")
    for npz in sorted(glob.glob(pat)):
        ed = os.path.dirname(os.path.dirname(os.path.dirname(npz)))  # .../<suffix>
        fid = None
        m = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(npz)))))
        if m.startswith("frame_"):
            try:
                fid = int(m.split("_")[1])
            except ValueError:
                fid = None
        if frames is not None and fid not in frames:
            continue
        out.append((fid, ed))
    # de-dup by frame id (keep first)
    seen, res_out = set(), []
    for fid, ed in out:
        if fid in seen:
            continue
        seen.add(fid); res_out.append((fid, ed))
    return res_out


def build_bg(export_dirs, percentile=90.0, min_valid=2, valid_depth=0.05,
             progress=lambda m, f: None):
    """Background model from N frames. bg[N_cam,H,W] = high-percentile depth per pixel.

    Processed per camera (bounds peak memory; nanpercentile on the whole 4-D stack is huge).
    """
    d0 = np.load(os.path.join(export_dirs[0], "exports", "npz", "results.npz"))
    K, E = d0["intrinsics"].astype(np.float64), d0["extrinsics"].astype(np.float64)
    F = len(export_dirs)
    N, H, W = d0["depth"].shape
    depths = np.empty((F, N, H, W), dtype=np.float32)
    for j, ed in enumerate(export_dirs):
        progress(f"load depth {j + 1}/{F}", 0.5 * j / F)
        depths[j] = np.load(os.path.join(ed, "exports", "npz", "results.npz"))["depth"]
    bg = np.zeros((N, H, W), dtype=np.float32)
    cnt = np.zeros((N, H, W), dtype=np.int32)
    for i in range(N):
        progress(f"bg camera {i + 1}/{N}", 0.5 + 0.5 * i / N)
        Di = depths[:, i, :, :]                          # (F, H, W)
        valid = Di > valid_depth
        cnt[i] = valid.sum(axis=0)
        Dm = np.where(valid, Di, np.nan)
        with np.errstate(all="ignore"):
            bgi = np.nanpercentile(Dm, percentile, axis=0)
        bg[i] = np.where(cnt[i] >= min_valid, np.nan_to_num(bgi), 0.0)
    return {"bg": bg, "count": cnt, "K": K, "E": E, "n_frames": F,
            "percentile": float(percentile)}


def save_bg(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, bg=model["bg"], count=model["count"], K=model["K"],
                        E=model["E"], n_frames=model["n_frames"], percentile=model["percentile"])


def load_bg(path):
    d = np.load(path)
    return {"bg": d["bg"], "count": d["count"], "K": d["K"], "E": d["E"],
            "n_frames": int(d["n_frames"]), "percentile": float(d["percentile"])}


def _crop_points_xy(pts, cols, margin=0.0, percentile=0.5):
    if pts is None or len(pts) == 0 or not margin or float(margin) <= 0:
        return pts, cols, 0, None
    p = max(0.0, min(float(percentile or 0.0), 20.0))
    if p > 0:
        lo = np.percentile(pts[:, :2], p, axis=0)
        hi = np.percentile(pts[:, :2], 100.0 - p, axis=0)
    else:
        lo = pts[:, :2].min(axis=0)
        hi = pts[:, :2].max(axis=0)
    m = float(margin)
    crop = [float(lo[0] + m), float(hi[0] - m), float(lo[1] + m), float(hi[1] - m)]
    if crop[0] >= crop[1] or crop[2] >= crop[3]:
        return pts, cols, 0, None
    keep = ((pts[:, 0] >= crop[0]) & (pts[:, 0] <= crop[1]) &
            (pts[:, 1] >= crop[2]) & (pts[:, 1] <= crop[3]))
    removed = int((~keep).sum())
    if removed <= 0:
        return pts, cols, 0, crop
    return pts[keep], cols[keep] if len(cols) else cols, removed, crop


def dynamic_points(export_dir, model, margin=0.25, min_bg_count=2, valid_depth=0.05,
                   conf_percentile=0.0, indoor_crop_margin=0.0,
                   indoor_crop_percentile=0.5):
    """World points (+colors) of ONLY the dynamic pixels of a frame (depth < bg - margin)."""
    dpt, conf, K, E, image = _load(export_dir)
    bg, cnt = model["bg"], model["count"]
    N, H, W = dpt.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([us.ravel(), vs.ravel(), np.ones(H * W)], axis=0).astype(np.float64)  # (3, HW)
    Pall, Call = [], []
    n_dyn = n_valid = 0
    for i in range(N):
        d = dpt[i].ravel(); b = bg[i].ravel(); bc = cnt[i].ravel(); c = conf[i].ravel()
        okpix = (d > valid_depth)
        n_valid += int(okpix.sum())
        dyn = okpix & (bc >= min_bg_count) & (b > valid_depth) & (d < b - margin)
        if conf_percentile and conf_percentile > 0 and okpix.any():
            thr = np.percentile(c[okpix], conf_percentile)
            dyn &= c >= thr
        if not dyn.any():
            continue
        n_dyn += int(dyn.sum())
        c2w = np.linalg.inv(_to_4x4(E[i]))
        rays = np.linalg.inv(K[i]) @ pix[:, dyn]          # (3, M)
        Xc = np.vstack([rays * d[dyn][None, :], np.ones((1, int(dyn.sum())))])
        Xw = (c2w @ Xc)[:3].T
        Pall.append(Xw)
        Call.append(image[i].reshape(-1, 3)[dyn].astype(np.float64) / 255.0)
    pts = np.concatenate(Pall) if Pall else np.empty((0, 3))
    cols = np.concatenate(Call) if Call else np.empty((0, 3))
    pts, cols, n_crop, crop_bounds = _crop_points_xy(
        pts, cols, margin=indoor_crop_margin, percentile=indoor_crop_percentile)
    return pts, cols, {"n_valid": n_valid, "n_dynamic_raw": n_dyn,
                       "n_dynamic": int(len(pts)),
                       "n_indoor_crop_removed": n_crop,
                       "indoor_crop_margin": float(indoor_crop_margin or 0.0),
                       "indoor_crop_percentile": float(indoor_crop_percentile or 0.0),
                       "indoor_crop_bounds": crop_bounds,
                       "kept_frac": round(len(pts) / max(1, n_valid), 4)}


def write_dynamic_ply(export_dir, model, out_path, margin=0.3,
                      indoor_crop_margin=0.0, indoor_crop_percentile=0.5):
    """Backproject the dynamic pixels of a frame and write them to ``out_path``. Stats dict."""
    import open3d as o3d
    pts, cols, st = dynamic_points(
        export_dir, model, margin=margin,
        indoor_crop_margin=indoor_crop_margin,
        indoor_crop_percentile=indoor_crop_percentile)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    if len(cols):
        pc.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(out_path, pc)
    return st


def ensure_zone_exports(output_root, root, split, scene, model_name, res, zone_name, frames=None):
    """Re-apply the zone mask to the zone-INDEPENDENT full exports → create the
    ``_<zone_name>`` siblings without re-running DA3 (the npz is the same; only the mask
    differs). Returns the number of zone exports created."""
    from . import zonemask
    scene_dir = os.path.join(root, split, scene)
    made = 0
    for _fid, fexp in find_exports(output_root, split, scene, model_name, res, False, frames, ""):
        zexp = fexp + "_" + (zone_name or "zone")
        if not os.path.isfile(os.path.join(zexp, "exports", "npz", "results.npz")):
            try:
                zonemask.apply(fexp, scene_dir, zone_name)
                made += 1
            except (FileNotFoundError, KeyError, ValueError):
                pass
    return made


def build_for_scene(*, output_root, split, scene, model_name, res, zone=True,
                    percentile=90.0, frames=None, out_path=None, zone_name="zone", root=""):
    if zone and root:                     # re-mask any MISSING zone exports from full npz (no DA3)
        ensure_zone_exports(output_root, root, split, scene, model_name, res, zone_name, frames)
    exps = find_exports(output_root, split, scene, model_name, res, zone, frames, zone_name)
    if not exps:
        raise FileNotFoundError(f"no npz found for {scene} res {res} zone={zone_name if zone else 0}")
    model = build_bg([ed for _f, ed in exps], percentile=percentile)
    tag = pipeline.zone_suffix(zone, zone_name)
    sdir = os.path.join(output_root, "final", scene)
    if out_path is None:
        out_path = os.path.join(sdir, f"bg_res{res}{tag}.npz")
    save_bg(model, out_path)
    # also a VERSIONED copy (frames + percentile) so the user can pick it later
    import shutil
    ver = os.path.join(sdir, f"bg_res{res}{tag}_f{len(exps)}_p{int(round(percentile))}.npz")
    shutil.copy(out_path, ver)
    return {"bg_path": out_path, "version_path": ver, "n_frames": len(exps),
            "frames": [f for f, _ in exps], "percentile": percentile,
            "shape": list(model["bg"].shape)}
