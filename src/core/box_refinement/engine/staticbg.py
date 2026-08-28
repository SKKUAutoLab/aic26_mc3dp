
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import open3d as o3d

from . import config, dataset, fusion, pipeline, static_point


def static_dir(output_root: str) -> str:
    return os.path.join(output_root, "static")


def _tag(voxel: float, min_frac: float, apply_zone: bool, zone_name: str = "zone") -> str:
    """Encode the build params into the filename so versions don't clash."""
    zt = (zone_name or "zone") if apply_zone else "full"
    return f"{zt}_v{int(round(voxel * 1000)):03d}_f{int(round(min_frac * 100)):02d}"


def list_static(output_root: str, scene: str) -> list:
    """All static-bg .ply versions for a scene, each with its build params."""
    out = []
    for ply in sorted(glob.glob(os.path.join(static_dir(output_root), f"{scene}*.ply"))):
        meta = {}
        mp = os.path.splitext(ply)[0] + ".json"
        if os.path.isfile(mp):
            try:
                meta = json.load(open(mp, encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
        out.append({"static_ply": ply, "label": os.path.basename(ply),
                    "voxel": meta.get("voxel"), "min_frames": meta.get("min_frames"),
                    "n_frames": meta.get("n_frames"),
                    "n_static_points": meta.get("n_static_points")})
    return out


def static_voxel_of(static_ply: str, default: float = 0.06) -> float:
    mp = os.path.splitext(static_ply)[0] + ".json"
    if os.path.isfile(mp):
        try:
            return float(json.load(open(mp, encoding="utf-8")).get("voxel", default))
        except Exception:  # noqa: BLE001
            pass
    return default


def build(*, root: str, split: str, scene: str, output_root: str,
          frame_step: int = 100, voxel: float = 0.06, min_frac: float = 0.6,
          model_name: str = config.DEFAULT_MODEL, process_res: int = 1008,
          gpu_id: int = 0, apply_zone: bool = True, zone_name: str = "zone",
          progress=lambda m, f: None) -> dict:
    info = dataset.scan_scene(root, split, scene)
    n_frames = info.get("num_frames") or 0
    frames = list(range(0, n_frames, frame_step)) or [0]

    model_san = pipeline.sanitize_model_name(model_name)
    ply_paths = []
    for k, f in enumerate(frames):
        frac = 0.05 + 0.85 * k / len(frames)
        # Reuse the per-frame fused .ply if it already exists (voxel/min_frac only affect
        # the aggregation, not DA3) → re-building a version is seconds, not minutes.
        run_dir = dataset.frame_run_dir(output_root, split, scene, f)
        export = os.path.join(run_dir, f"{model_san}_res_{process_res}") + pipeline.zone_suffix(apply_zone, zone_name)
        ply = os.path.join(export, "pointcloud", "estimated_fused.ply")
        if os.path.isfile(ply):
            progress(f"reuse frame {f} ({k + 1}/{len(frames)})", frac)
            ply_paths.append(ply)
            continue
        progress(f"DA3 frame {f} ({k + 1}/{len(frames)})", frac)
        res = pipeline.run_inference(
            root=root, split=split, scene=scene, frame_id=f, cameras="all",
            model_name=model_name, process_res=process_res, export_format="npz",
            gpu_id=gpu_id, output_root=output_root, apply_zone=apply_zone, zone_name=zone_name)
        zinfo = res.get("zone")
        exp = (zinfo["zone_export_dir"] if apply_zone and zinfo and not zinfo.get("error")
               else res["export_dir"])
        fr = fusion.fuse(export_dir=exp, method="union")
        ply_paths.append(fr["ply_path"])

    progress("Aggregating static voxels...", 0.92)
    min_frames = max(2, int(round(min_frac * len(frames))))
    pcd, meta = static_point.build_static(ply_paths, voxel, min_frames)

    tag = _tag(voxel, min_frac, apply_zone, zone_name)
    out_ply = os.path.join(static_dir(output_root), f"{scene}__{tag}.ply")
    os.makedirs(os.path.dirname(out_ply), exist_ok=True)
    o3d.io.write_point_cloud(out_ply, pcd)
    meta = {**(meta or {}), "scene": scene, "voxel": voxel, "min_frac": min_frac,
            "min_frames": min_frames, "n_frames": len(frames), "frames": frames,
            "apply_zone": apply_zone, "process_res": process_res, "label": tag,
            "static_ply": out_ply, "n_static_points": int(len(pcd.points))}
    with open(os.path.splitext(out_ply)[0] + ".json", "w") as fp:
        json.dump(meta, fp, indent=2, default=str)
    progress("Done.", 1.0)
    return meta


def subtract(pcd, static_ply: str, voxel: float):
    """Drop points of ``pcd`` whose voxel is occupied in the static cloud."""
    sp = np.asarray(o3d.io.read_point_cloud(static_ply).points)
    if len(sp) == 0 or len(pcd.points) == 0:
        return pcd, 0
    static_keys = np.unique(static_point._voxel_keys(sp, voxel))
    keys = static_point._voxel_keys(np.asarray(pcd.points), voxel)
    loc = np.clip(np.searchsorted(static_keys, keys), 0, len(static_keys) - 1)
    in_static = static_keys[loc] == keys
    keep = np.where(~in_static)[0]
    return pcd.select_by_index(keep.tolist()), int(in_static.sum())


def main():
    ap = argparse.ArgumentParser(description="Build static-background cloud")
    ap.add_argument("--root", default="/home/vsw/ws/data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--output", default=config.DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--voxel", type=float, default=0.06)
    ap.add_argument("--min-frac", type=float, default=0.6)
    ap.add_argument("--res", type=int, default=1008)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-zone", action="store_true")
    a = ap.parse_args()
    meta = build(root=a.root, split=a.split, scene=a.scene, output_root=a.output,
                 frame_step=a.step, voxel=a.voxel, min_frac=a.min_frac,
                 process_res=a.res, gpu_id=a.gpu, apply_zone=not a.no_zone,
                 progress=lambda m, f: print(f"[{f*100:5.1f}%] {m}", flush=True))
    print("STATIC DONE:", json.dumps(meta, default=str))


if __name__ == "__main__":
    main()
