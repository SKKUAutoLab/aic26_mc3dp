
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import open3d as o3d

# Voxel key encoding: integer grid index -> single int64 key.
# OFF makes indices non-negative; M bounds each axis. M^3 = 2^60 < int64 max,
# and the ±OFF range (±2^19 voxels) covers ±26 km at 5 cm — ample for a warehouse.
_OFF = 1 << 19
_M = 1 << 20
_M2 = _M * _M


def _voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    """Map Nx3 world points to int64 voxel keys on a grid anchored at origin."""
    idx = np.floor(points / voxel).astype(np.int64)
    ijk = idx + _OFF
    if ijk.min() < 0 or ijk.max() >= _M:
        raise ValueError(
            f"point out of voxel-grid range (idx in [{idx.min()}, {idx.max()}]); "
            f"increase _M or check units"
        )
    return ijk[:, 0] * _M2 + ijk[:, 1] * _M + ijk[:, 2]


def _read_ply(path: str):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)  # (N,3) float64, world metres
    cols = np.asarray(pcd.colors)  # (N,3) float64 in [0,1] (empty if none)
    if cols.shape[0] != pts.shape[0]:
        cols = np.zeros_like(pts)
    return pts, cols


def build_static(
    frame_paths: list[str],
    voxel: float,
    min_frames: int,
) -> tuple[o3d.geometry.PointCloud, dict]:
    n = len(frame_paths)

    # Pass A: per-frame occupied voxel sets -> how many frames each voxel appears in.
    per_frame_unique = []
    total_in = 0
    for p in frame_paths:
        pts, _ = _read_ply(p)
        total_in += pts.shape[0]
        keys = _voxel_keys(pts, voxel)
        per_frame_unique.append(np.unique(keys))
        del pts, keys
    all_unique = np.concatenate(per_frame_unique)
    del per_frame_unique
    voxel_ids, frame_count = np.unique(all_unique, return_counts=True)
    del all_unique
    n_voxels_total = voxel_ids.shape[0]

    kept_sorted = voxel_ids[frame_count >= min_frames]  # already sorted by np.unique
    nkept = kept_sorted.shape[0]
    del voxel_ids, frame_count
    if nkept == 0:
        raise RuntimeError(
            f"no voxel reached min_frames={min_frames} of {n}; lower it or raise --voxel"
        )

    # Pass B: accumulate centroid + mean colour over kept voxels only.
    sum_xyz = np.zeros((nkept, 3), dtype=np.float64)
    sum_rgb = np.zeros((nkept, 3), dtype=np.float64)
    cnt = np.zeros(nkept, dtype=np.int64)
    for p in frame_paths:
        pts, cols = _read_ply(p)
        keys = _voxel_keys(pts, voxel)
        loc = np.searchsorted(kept_sorted, keys)
        np.clip(loc, 0, nkept - 1, out=loc)
        mask = kept_sorted[loc] == keys
        if not mask.any():
            del pts, cols, keys, loc
            continue
        idx = loc[mask]
        pk, ck = pts[mask], cols[mask]
        cnt += np.bincount(idx, minlength=nkept)
        for a in range(3):
            sum_xyz[:, a] += np.bincount(idx, weights=pk[:, a], minlength=nkept)
            sum_rgb[:, a] += np.bincount(idx, weights=ck[:, a], minlength=nkept)
        del pts, cols, keys, loc, mask, idx, pk, ck

    centroid = sum_xyz / cnt[:, None]
    colour = sum_rgb / cnt[:, None]

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(centroid)
    out.colors = o3d.utility.Vector3dVector(np.clip(colour, 0.0, 1.0))

    stats = {
        "n_frames": n,
        "min_frames": min_frames,
        "voxel": voxel,
        "total_input_points": int(total_in),
        "voxels_total": int(n_voxels_total),
        "voxels_static": int(nkept),
        "static_fraction": round(nkept / max(n_voxels_total, 1), 4),
    }
    return out, stats


def process_scene(scene: str, pd_bev_root: str, voxel: float, min_frames: int,
                  overwrite: bool) -> None:
    scene_dir = os.path.join(pd_bev_root, scene)
    frame_paths = sorted(glob.glob(os.path.join(scene_dir, "frame_*.ply")))
    if not frame_paths:
        print(f"[static] SKIP {scene}: no frame_*.ply in {scene_dir}")
        return

    out_path = os.path.join(scene_dir, "static.ply")
    if os.path.isfile(out_path) and not overwrite:
        print(f"[static] {scene}: exists, skip ({out_path}) — use --overwrite")
        return

    k = min(min_frames, len(frame_paths))
    print(f"[static] {scene}: {len(frame_paths)} frames, voxel={voxel}m, keep>={k}/{len(frame_paths)}")
    t0 = time.time()
    pcd, stats = build_static(frame_paths, voxel=voxel, min_frames=k)
    o3d.io.write_point_cloud(out_path, pcd)
    dt = time.time() - t0
    print(
        f"[static] {scene}: {stats['total_input_points']:,} in -> "
        f"{stats['voxels_static']:,} static pts "
        f"({stats['static_fraction'] * 100:.1f}% of {stats['voxels_total']:,} voxels), "
        f"{dt:.1f}s -> {out_path}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pd-bev-root", default="/home/vsw/ws/data/pd_bev",
                   help="root holding <scene>/frame_*.ply")
    p.add_argument("--scenes", nargs="*", default=["Warehouse_026", "Warehouse_027"],
                   help="scenes to process (default: Warehouse_026 Warehouse_027)")
    p.add_argument("--voxel", type=float, default=0.05, help="voxel size in metres")
    p.add_argument("--min-frames", type=int, default=4,
                   help="keep voxels occupied in >= this many frames")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    root = os.path.abspath(os.path.expanduser(args.pd_bev_root))
    print(f"[static] pd_bev_root={root}")
    for scene in args.scenes:
        process_scene(scene, root, args.voxel, args.min_frames, args.overwrite)


if __name__ == "__main__":
    main()
