
from __future__ import annotations

import hashlib
import json
import os
import threading

import numpy as np
import open3d as o3d
from PIL import Image

from . import gt
from .calib import to_4x4  # noqa: F401


# --------------------------------------------------------------------------- #
# GT box geometry / colors
# --------------------------------------------------------------------------- #
def _type_color(name: str) -> np.ndarray:
    import colorsys
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return np.array(colorsys.hsv_to_rgb((h % 360) / 360.0, 0.7, 0.95))


def _gt_box_geoms(root, split, scene, frame_id) -> list:
    gt_path = os.path.join(root, split, scene, "ground_truth.json")
    if not os.path.isfile(gt_path):
        return []
    data = gt.get_gt_boxes(gt_path, frame_id)
    edges = np.asarray(data["edges"], dtype=np.int32)
    geoms = []
    for box in data["boxes"]:
        corners = np.asarray(box["corners"], dtype=np.float64)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines = o3d.utility.Vector2iVector(edges)
        col = _type_color(box["object_type"])
        ls.colors = o3d.utility.Vector3dVector(np.tile(col, (len(edges), 1)))
        geoms.append(ls)
    return geoms


# --------------------------------------------------------------------------- #
# Shared filters
# --------------------------------------------------------------------------- #
def _conf_threshold(conf_i, sky_i, conf_thresh_percentile, ensure=90.0, floor=1.05):
    pool = conf_i[~sky_i] if (sky_i is not None and (~sky_i).sum() > 10) else conf_i
    lower = np.percentile(pool, conf_thresh_percentile)
    upper = np.percentile(pool, ensure)
    return min(max(floor, lower), upper)


def _valid_mask(d, c, sky_i, conf_thresh_percentile, min_depth, max_depth, conf_floor=1.05):
    valid = np.isfinite(d) & (d > 0)
    if min_depth and min_depth > 0:
        valid &= (d >= min_depth)
    if max_depth and max_depth > 0:
        valid &= (d <= max_depth)
    if sky_i is not None:
        valid &= ~sky_i
    # conf_floor < ~1.01 effectively disables confidence gating (keep all valid depth);
    # useful for models whose conf is ~1.0 on most pixels (very conservative).
    if c is not None and conf_floor >= 1.01:
        valid &= (c >= _conf_threshold(c, sky_i, conf_thresh_percentile, floor=conf_floor))
    return valid


def indoor_bounds_from_scene(root: str, split: str, scene: str):
    """Return official map XY bounds [xmin, xmax, ymin, ymax] in world meters.

    AICity calibration stores the map transform as:
    pixel_x = (world_x + tx) * scaleFactor
    pixel_y = (world_y + ty) * scaleFactor
    """
    scene_dir = os.path.join(root, split, scene)
    calib_path = os.path.join(scene_dir, "calibration.json")
    map_path = os.path.join(scene_dir, "map.png")
    if not (os.path.isfile(calib_path) and os.path.isfile(map_path)):
        return None
    try:
        calib = json.load(open(calib_path, encoding="utf-8"))
        sf, trans = None, None
        for sensor in calib.get("sensors", []):
            if sensor.get("type") == "camera":
                sf = sensor.get("scaleFactor")
                trans = sensor.get("translationToGlobalCoordinates")
                if sf and trans:
                    break
        if not (sf and trans):
            return None
        sf = float(sf)
        tx, ty = float(trans["x"]), float(trans["y"])
        with Image.open(map_path) as img:
            width, height = img.size
        return [-tx, width / sf - tx, -ty, height / sf - ty]
    except Exception:  # noqa: BLE001
        return None


def _crop_xy_pcd(pcd, bounds, margin, percentile=0.0):
    if not bounds or margin is None or float(margin) <= 0 or len(pcd.points) == 0:
        if margin is None or float(margin) <= 0 or len(pcd.points) == 0:
            return pcd, 0, None
        pts0 = np.asarray(pcd.points)
        p = max(0.0, min(float(percentile or 0.0), 20.0))
        if p > 0:
            lo = np.percentile(pts0[:, :2], p, axis=0)
            hi = np.percentile(pts0[:, :2], 100.0 - p, axis=0)
        else:
            lo = pts0[:, :2].min(axis=0)
            hi = pts0[:, :2].max(axis=0)
        xmin, ymin = lo.tolist()
        xmax, ymax = hi.tolist()
    else:
        xmin, xmax, ymin, ymax = [float(v) for v in bounds]
    margin = float(margin)
    crop = [xmin + margin, xmax - margin, ymin + margin, ymax - margin]
    if crop[0] >= crop[1] or crop[2] >= crop[3]:
        return pcd, 0, None
    pts = np.asarray(pcd.points)
    keep = ((pts[:, 0] >= crop[0]) & (pts[:, 0] <= crop[1]) &
            (pts[:, 1] >= crop[2]) & (pts[:, 1] <= crop[3]))
    removed = int((~keep).sum())
    if removed <= 0:
        return pcd, 0, crop
    return pcd.select_by_index(np.flatnonzero(keep).tolist()), removed, crop


def _load(npz_path, sky_path):
    data = np.load(npz_path)
    out = {
        "depth": data["depth"].astype(np.float32),
        "conf": data["conf"].astype(np.float32) if "conf" in data else None,
        "K": data["intrinsics"].astype(np.float64),
        "E": data["extrinsics"].astype(np.float64),
        "image": data["image"].astype(np.uint8) if "image" in data else None,
        "sky": np.load(sky_path) if (sky_path and os.path.isfile(sky_path)) else None,
    }
    return out


# --------------------------------------------------------------------------- #
# union / voxel point builder
# --------------------------------------------------------------------------- #
def _build_pcd(d_data, conf_thresh_percentile, stride, min_depth, max_depth, conf_floor=1.05):
    depth, conf, K, E, image, sky = (
        d_data["depth"], d_data["conf"], d_data["K"], d_data["E"], d_data["image"], d_data["sky"]
    )
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3).astype(np.float64)
    stride = max(int(stride), 1)
    stride_mask = np.zeros((H, W), dtype=bool)
    stride_mask[::stride, ::stride] = True
    stride_flat = stride_mask.reshape(-1)

    all_pts, all_col = [], []
    for i in range(N):
        valid = _valid_mask(depth[i], conf[i] if conf is not None else None,
                            sky[i] if sky is not None else None,
                            conf_thresh_percentile, min_depth, max_depth, conf_floor)
        idx = np.flatnonzero(valid.reshape(-1) & stride_flat)
        if len(idx) == 0:
            continue
        K_inv = np.linalg.inv(K[i])
        c2w = np.linalg.inv(to_4x4(E[i]))
        rays = K_inv @ pix[idx].T
        z = depth[i].reshape(-1)[idx][None, :]
        X_cam = rays * z
        X_cam_h = np.vstack([X_cam, np.ones((1, X_cam.shape[1]))])
        X_world = (c2w @ X_cam_h)[:3].T.astype(np.float32)
        col = (image[i].reshape(-1, 3)[idx].astype(np.float32) / 255.0
               if image is not None else np.full((len(idx), 3), 0.6, np.float32))
        all_pts.append(X_world)
        all_col.append(col)

    if not all_pts:
        return o3d.geometry.PointCloud(), 0
    pts = np.concatenate(all_pts, 0)
    col = np.concatenate(all_col, 0)
    finite = np.isfinite(pts).all(axis=1)
    pts, col = pts[finite], col[finite]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64))
    return pcd, int(len(pts))


# --------------------------------------------------------------------------- #
# Per-camera clouds + ICP synchronization (GT-free de-ghost) + consolidation
# --------------------------------------------------------------------------- #
def _camera_pcd(d_data, i, conf_thresh_percentile, stride, min_depth, max_depth, conf_floor=1.05):
    depth, conf, K, E, image, sky = (
        d_data["depth"], d_data["conf"], d_data["K"], d_data["E"], d_data["image"], d_data["sky"]
    )
    H, W = depth[i].shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3).astype(np.float64)
    stride = max(int(stride), 1)
    sflat = np.zeros((H, W), bool); sflat[::stride, ::stride] = True; sflat = sflat.reshape(-1)
    valid = _valid_mask(depth[i], conf[i] if conf is not None else None,
                        sky[i] if sky is not None else None,
                        conf_thresh_percentile, min_depth, max_depth, conf_floor)
    idx = np.flatnonzero(valid.reshape(-1) & sflat)
    if len(idx) == 0:
        return None
    c2w = np.linalg.inv(to_4x4(E[i]))
    rays = np.linalg.inv(K[i]) @ pix[idx].T
    z = depth[i].reshape(-1)[idx][None, :]
    Xc = np.vstack([rays * z, np.ones((1, len(idx)))])
    Xw = (c2w @ Xc)[:3].T.astype(np.float64)
    col = (image[i].reshape(-1, 3)[idx].astype(np.float64) / 255.0
           if image is not None else np.full((len(idx), 3), 0.6))
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(Xw)
    p.colors = o3d.utility.Vector3dVector(col)
    return p


def _icp_sync(pcds, max_corr=0.2, voxel=0.05):
    """Greedy per-camera point-to-plane ICP to an accumulating reference.

    Calibration already aligns clouds well; this only refines the small residual
    (rigid) per-camera misalignment that causes ghosting. Conservative
    max_correspondence_distance + a fitness gate keep it from breaking sparse-overlap
    cameras (low-overlap cameras are left at their calibration pose).
    """
    order = sorted(range(len(pcds)), key=lambda k: len(pcds[k].points), reverse=True)
    out = [None] * len(pcds)
    acc = o3d.geometry.PointCloud()
    n_aligned = 0
    for rank, k in enumerate(order):
        full = pcds[k]
        src = full.voxel_down_sample(voxel)
        if rank == 0:
            out[k] = full
            acc += src
            continue
        tgt = acc
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, max_corr, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
        )
        moved = o3d.geometry.PointCloud(full)
        if reg.fitness > 0.2:
            moved.transform(reg.transformation)
            n_aligned += 1
        out[k] = moved
        acc += moved.voxel_down_sample(voxel)
    return out, n_aligned


def _merge(pcds):
    m = o3d.geometry.PointCloud()
    for p in pcds:
        if p is not None and len(p.points) > 0:
            m += p
    return m


def _postclean(pcd, statistical, nb_neighbors, std_ratio,
               radius_outlier, radius_nb, radius, cluster_denoise, cluster_eps, min_cluster):
    """Make points cluster-friendly: drop floaters + tiny noise specks."""
    if len(pcd.points) == 0:
        return pcd
    if statistical:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if radius_outlier and len(pcd.points) > radius_nb:
        pcd, _ = pcd.remove_radius_outlier(nb_points=int(radius_nb), radius=float(radius))
    if cluster_denoise and len(pcd.points) > min_cluster:
        labels = np.asarray(pcd.cluster_dbscan(eps=float(cluster_eps), min_points=10))
        if labels.size and labels.max() >= 0:
            counts = np.bincount(labels[labels >= 0])
            big = {c for c in range(len(counts)) if counts[c] >= int(min_cluster)}
            keep = np.array([l in big for l in labels])
            if keep.any():
                pcd = pcd.select_by_index(np.flatnonzero(keep))
    return pcd


# --------------------------------------------------------------------------- #
# TSDF (Open3D ScalableTSDFVolume) — extrinsic passed is world-to-camera
# --------------------------------------------------------------------------- #
def _fuse_tsdf(d_data, conf_thresh_percentile, min_depth, max_depth,
               voxel_length, sdf_trunc, out_mesh_path, conf_floor=1.05):
    depth, conf, K, E, image, sky = (
        d_data["depth"], d_data["conf"], d_data["K"], d_data["E"], d_data["image"], d_data["sky"]
    )
    N, H, W = depth.shape
    depth_trunc = float(max_depth) if max_depth and max_depth > 0 else 1000.0
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length), sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for i in range(N):
        d = depth[i].copy()
        valid = _valid_mask(d, conf[i] if conf is not None else None,
                            sky[i] if sky is not None else None,
                            conf_thresh_percentile, min_depth, max_depth, conf_floor)
        d[~valid] = 0.0
        color = (np.ascontiguousarray(image[i]) if image is not None
                 else np.full((H, W, 3), 128, np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(np.ascontiguousarray(d.astype(np.float32))),
            depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
        )
        intr = o3d.camera.PinholeCameraIntrinsic(
            W, H, float(K[i][0, 0]), float(K[i][1, 1]), float(K[i][0, 2]), float(K[i][1, 2]))
        vol.integrate(rgbd, intr, to_4x4(E[i]))  # world-to-camera

    pcd = vol.extract_point_cloud()
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_mesh_path, mesh)
    return pcd, int(len(mesh.vertices))


# --------------------------------------------------------------------------- #
# Geometric consistency (COLMAP/MVS style) — keep points agreed across views
# --------------------------------------------------------------------------- #
def _fuse_consistency(d_data, conf_thresh_percentile, stride, min_depth, max_depth,
                      min_views, rel_eps, merge_voxel, conf_floor=1.05):
    depth, conf, K, E, image, sky = (
        d_data["depth"], d_data["conf"], d_data["K"], d_data["E"], d_data["image"], d_data["sky"]
    )
    N, H, W = depth.shape
    E4 = np.stack([to_4x4(E[i]) for i in range(N)])  # w2c
    c2w = np.stack([np.linalg.inv(E4[i]) for i in range(N)])

    masks = [_valid_mask(depth[i], conf[i] if conf is not None else None,
                         sky[i] if sky is not None else None,
                         conf_thresh_percentile, min_depth, max_depth, conf_floor)
             for i in range(N)]

    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([us, vs, np.ones_like(us)], axis=-1).astype(np.float64)
    stride = max(int(stride), 1)
    smask = np.zeros((H, W), bool); smask[::stride, ::stride] = True

    all_pts, all_col = [], []
    for i in range(N):
        sel = masks[i] & smask
        ys, xs = np.nonzero(sel)
        if len(xs) == 0:
            continue
        d_i = depth[i][ys, xs]
        rays = np.linalg.inv(K[i]) @ pix[ys, xs].T          # 3xM
        Xc = rays * d_i[None, :]
        Xw = (c2w[i][:3, :3] @ Xc + c2w[i][:3, 3:4])        # 3xM world
        votes = np.ones(Xw.shape[1], dtype=np.int32)        # self counts as 1
        for j in range(N):
            if j == i:
                continue
            Xcj = E4[j][:3, :3] @ Xw + E4[j][:3, 3:4]
            zj = Xcj[2]
            front = zj > 1e-6
            uvj = (K[j] @ Xcj)
            uj = np.round(uvj[0] / np.where(front, zj, 1)).astype(int)
            vj = np.round(uvj[1] / np.where(front, zj, 1)).astype(int)
            inb = front & (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H)
            cons = np.zeros(Xw.shape[1], dtype=bool)
            if inb.any():
                ui, vi = uj[inb], vj[inb]
                dmeas = depth[j][vi, ui]
                valid_j = masks[j][vi, ui] & (dmeas > 0)
                rel = np.abs(zj[inb] - dmeas) / np.maximum(dmeas, 1e-6)
                cons[inb] = valid_j & (rel < rel_eps)
            votes += cons.astype(np.int32)
        keep = votes >= int(min_views)
        if not keep.any():
            continue
        all_pts.append(Xw[:, keep].T.astype(np.float32))
        if image is not None:
            all_col.append(image[i][ys, xs][keep].astype(np.float32) / 255.0)
        else:
            all_col.append(np.full((int(keep.sum()), 3), 0.6, np.float32))

    if not all_pts:
        return o3d.geometry.PointCloud(), 0
    pts = np.concatenate(all_pts, 0); col = np.concatenate(all_col, 0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64))
    raw = len(pts)
    if merge_voxel and merge_voxel > 0:  # merge near-duplicate consistent points
        pcd = pcd.voxel_down_sample(float(merge_voxel))
    return pcd, raw


# --------------------------------------------------------------------------- #
def _plane_stats(pcd):
    pts = np.asarray(pcd.points)
    if len(pts) < 1000:
        return None
    plane_model, inliers = pcd.segment_plane(0.05, 3, 2000)
    a, b, c, d = [float(x) for x in plane_model]
    dist = np.abs(pts @ np.array([a, b, c]) + d) / (np.linalg.norm([a, b, c]) + 1e-9)
    return {"a": a, "b": b, "c": c, "d": d,
            "inlier_ratio": round(len(inliers) / max(len(pts), 1), 3),
            "median_dist": float(np.median(dist[inliers]))}


def fuse(
    *,
    export_dir: str,
    method: str = "union",
    conf_thresh_percentile: float = 40.0,
    conf_floor: float = 1.05,   # <1.01 = keep all valid depth
    stride: int = 1,
    min_depth: float = 0.0,
    max_depth: float = 0.0,
    outlier_removal: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    use_refined: bool = False,
    voxel_size: float = 0.03,
    voxel_length: float = 0.03,
    sdf_trunc: float = 0.12,
    min_views: int = 2,
    rel_eps: float = 0.02,
    # consolidation (make points beautiful + cluster-friendly)
    icp_sync: bool = False,
    icp_max_corr: float = 0.2,
    radius_outlier: bool = False,
    radius_nb: int = 16,
    radius: float = 0.1,
    cluster_denoise: bool = False,
    cluster_eps: float = 0.2,
    min_cluster: int = 40,
    remove_static: str = "",  # path to a static-bg .ply → subtract its voxels (keep dynamic)
    static_voxel: float = 0.06,
    indoor_crop_margin: float = 0.0,  # crop X/Y inward from indoor/cloud bounds
    indoor_bounds=None,               # [xmin, xmax, ymin, ymax] in world meters
    indoor_crop_percentile: float = 0.5,  # robust cloud edge percentile if no bounds supplied
) -> dict:
    base = os.path.join(export_dir, "exports", "npz", "results.npz")
    refined = os.path.join(export_dir, "exports", "npz", "results_refined.npz")
    npz_path = refined if (use_refined and os.path.isfile(refined)) else base
    used_refined = npz_path == refined
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"results.npz not found: {npz_path}")
    sky_path = os.path.join(export_dir, "exports", "npz", "sky.npy")
    d_data = _load(npz_path, sky_path)

    pc_dir = os.path.join(export_dir, "pointcloud")
    os.makedirs(pc_dir, exist_ok=True)
    mesh_path = None
    n_icp_aligned = 0
    N = d_data["depth"].shape[0]

    if method == "tsdf":
        mesh_path = os.path.join(pc_dir, "tsdf_mesh.ply")
        pcd, raw_count = _fuse_tsdf(d_data, conf_thresh_percentile, min_depth, max_depth,
                                    voxel_length, sdf_trunc, mesh_path, conf_floor)
    elif method == "consistency":
        pcd, raw_count = _fuse_consistency(d_data, conf_thresh_percentile, stride,
                                           min_depth, max_depth, min_views, rel_eps,
                                           voxel_size, conf_floor)
    else:  # union or voxel — build per camera so we can ICP-synchronize cameras
        pcds = [_camera_pcd(d_data, i, conf_thresh_percentile, stride, min_depth, max_depth,
                            conf_floor)
                for i in range(N)]
        pcds = [p for p in pcds if p is not None and len(p.points) > 0]
        if icp_sync and len(pcds) >= 2:
            pcds, n_icp_aligned = _icp_sync(pcds, max_corr=icp_max_corr, voxel=max(voxel_size, 0.03))
        pcd = _merge(pcds)
        raw_count = int(len(pcd.points))
        if method == "voxel" and len(pcd.points) > 0:
            pcd = pcd.voxel_down_sample(float(voxel_size))

    # Indoor trim: cut inward from XY bounds to remove wall-border points before
    # the heavier cleanup/subtraction. If no map bounds are supplied, use robust
    # cloud bounds so a few outliers do not define the room edge.
    pcd, n_indoor_crop_removed, indoor_crop_bounds = _crop_xy_pcd(
        pcd, indoor_bounds, indoor_crop_margin, indoor_crop_percentile)

    # Consolidation: make points beautiful + cluster-friendly (drop floaters + specks).
    pcd = _postclean(pcd, outlier_removal and method != "tsdf", nb_neighbors, std_ratio,
                     radius_outlier, radius_nb, radius, cluster_denoise, cluster_eps, min_cluster)

    # Optional background subtraction: drop points sitting in the scene's static voxels.
    n_static_removed = 0
    if remove_static and os.path.isfile(remove_static) and len(pcd.points) > 0:
        from . import staticbg
        pcd, n_static_removed = staticbg.subtract(pcd, remove_static, static_voxel)

    plane = _plane_stats(pcd)
    ply_path = os.path.join(pc_dir, "estimated_fused.ply")
    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)  # FULL cloud always saved

    n_full = int(len(pcd.points))

    return {
        "method": method,
        "used_refined": used_refined,
        "ply_path": ply_path,
        "mesh_path": mesh_path,
        "num_points": n_full, "raw_points": int(raw_count),
        "n_indoor_crop_removed": int(n_indoor_crop_removed),
        "indoor_crop_margin": float(indoor_crop_margin or 0.0),
        "indoor_crop_percentile": float(indoor_crop_percentile or 0.0),
        "indoor_crop_bounds": indoor_crop_bounds,
        "n_static_removed": int(n_static_removed),
        "min_depth": min_depth, "max_depth": max_depth,
        "sky_mask_applied": bool(d_data["sky"] is not None),
        "icp_sync": bool(icp_sync), "icp_cameras_aligned": int(n_icp_aligned),
        "bbox": bbox, "plane": plane,
    }


# --------------------------------------------------------------------------- #
# Native Open3D window (separate program on the server display)
# --------------------------------------------------------------------------- #
def _open_native_viewer(ply_path, gt_geoms, axis_size=3.0):
    def _show():
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
            o3d.visualization.draw_geometries([pcd, frame, *gt_geoms],
                                              window_name="DA3 fused point cloud + GT boxes")
        except Exception as exc:  # noqa: BLE001
            print(f"[fusion] native viewer failed: {exc}")
    threading.Thread(target=_show, daemon=True).start()


def open_native(export_dir, root=None, split=None, scene=None, frame_id=None) -> dict:
    ply_path = os.path.join(export_dir, "pointcloud", "estimated_fused.ply")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError("Fuse first: estimated_fused.ply not found.")
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0))) if len(pts) else 10.0
    gt_geoms = []
    if root and split and scene and frame_id is not None:
        try:
            gt_geoms = _gt_box_geoms(root, split, scene, frame_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[fusion] GT boxes skipped: {exc}")
    _open_native_viewer(ply_path, gt_geoms, axis_size=min(extent * 0.1, 5.0))
    return {"opened": True, "ply_path": ply_path, "num_gt_boxes": len(gt_geoms)}
