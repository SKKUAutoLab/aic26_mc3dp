
from __future__ import annotations

import json
import os
import re
import shutil
import time
from datetime import datetime

import numpy as np
import torch

from .. import _bootstrap  # noqa: F401  (must precede the depth_anything_3 imports)
from . import config, dataset
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.memory import cleanup_cuda_memory, estimate_memory_requirement
from .calib import build_sensor_map, load_calibration, to_4x4_extrinsic


# (model_name, gpu_id) -> loaded model. We keep at most one model per GPU.
_MODEL_CACHE: dict[tuple[str, int], DepthAnything3] = {}


def sanitize_model_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name)


def zone_suffix(apply_zone: bool, zone_name: str = "zone") -> str:
    """Export-dir suffix for a zone variant: ''/'_zone'/'_zone_new'. Empty when no zone."""
    return ("_" + (zone_name or "zone")) if apply_zone else ""


def build_camera_arrays(calib_path: str, camera_ids: list[str]):
    """Return (K[N,3,3], E[N,4,4], used_ids, skipped) for the given cameras.

    K is the full-resolution calibration intrinsic; DA3's input processor
    rescales it internally to the processing resolution.
    """
    calib = load_calibration(calib_path)
    sensor_map = build_sensor_map(calib)

    Ks, Es, used, skipped = [], [], [], []
    for cam_id in camera_ids:
        sensor = sensor_map.get(cam_id)
        if sensor is None:
            skipped.append(cam_id)
            continue
        Ks.append(np.asarray(sensor["intrinsicMatrix"], dtype=np.float32))
        Es.append(to_4x4_extrinsic(sensor["extrinsicMatrix"]))
        used.append(cam_id)

    if len(used) < 1:
        raise RuntimeError("No calibrated cameras matched the selection.")

    return (
        np.asarray(Ks, dtype=np.float32),
        np.asarray(Es, dtype=np.float32),
        used,
        skipped,
    )


def _get_model(model_name: str, gpu_id: int) -> DepthAnything3:
    key = (model_name, gpu_id)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    # Evict any other model living on the same GPU to bound VRAM usage.
    for (n, g) in list(_MODEL_CACHE.keys()):
        if g == gpu_id:
            del _MODEL_CACHE[(n, g)]
    cleanup_cuda_memory()

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    model.device = device
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def clear_model_cache(gpu_id: int | None = None) -> None:
    """Drop cached DA3 models, usually after CUDA OOM or before a heavy retry."""
    for key in list(_MODEL_CACHE.keys()):
        if gpu_id is None or key[1] == gpu_id:
            del _MODEL_CACHE[key]
    cleanup_cuda_memory()


def _wait_for_file(path: str, timeout: float = 240.0) -> bool:
    """Wait until ``path`` exists and its size is stable (async npz writer)."""
    start = time.time()
    last_size = -1
    stable = 0
    while time.time() - start < timeout:
        if os.path.isfile(path):
            size = os.path.getsize(path)
            if size == last_size and size > 0:
                stable += 1
                if stable >= 2:
                    return True
            else:
                stable = 0
            last_size = size
        time.sleep(0.5)
    return os.path.isfile(path)


def run_inference(
    *,
    root: str,
    split: str,
    scene: str,
    frame_id: int,
    cameras,
    model_name: str,
    process_res: int,
    export_format: str,
    gpu_id: int,
    output_root: str,
    ref_view_strategy: str = "saddle_balanced",
    use_ray_pose: bool = False,
    ckpt_path: str | None = None,
    apply_zone: bool = False,
    zone_name: str = "zone",
    zone_dir: str = "",
    progress=lambda msg, frac: None,
) -> dict:
    progress("Extracting frame...", 0.05)
    # The frames are decoded from the videos into per-camera JPEGs, which DA3 then reads back. The
    # JPEG round-trip looks wasteful -- DA3 accepts arrays -- but it is part of what the results were
    # validated against: JPEG is lossy, and handing DA3 the raw pixels instead moved boxes by up to
    # 0.35 m (measured on scene 25). The JPEGs are deleted again as soon as the frame is refined.
    extract = dataset.extract_frame(root, split, scene, frame_id, cameras, output_root)
    run_dir = extract["run_dir"]
    frames_dir = extract["frames_dir"]
    extracted_ids = [f["camera"] for f in extract["frames"]]
    if not extracted_ids:
        raise RuntimeError("No frames could be extracted.")

    scene_info = dataset.scan_scene(root, split, scene)
    calib_path = scene_info["calibration_path"]
    if calib_path is None:
        raise RuntimeError("calibration.json missing; cannot run DA3 with poses.")

    progress("Building calibration arrays...", 0.1)
    K, E, used_ids, skipped = build_camera_arrays(calib_path, extracted_ids)
    image_paths = [os.path.join(frames_dir, f"{cid}.jpg") for cid in used_ids]

    model_sanitized = sanitize_model_name(model_name)
    export_dir = os.path.join(run_dir, f"{model_sanitized}_res_{process_res}")
    meta_dir = os.path.join(export_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    # Stable camera order -> depth[i].
    with open(os.path.join(meta_dir, "camera_ids.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "camera_ids": used_ids,
                "skipped_no_calibration": skipped,
                "note": "index i corresponds to depth[i] / extrinsics[i] in results.npz",
            },
            f,
            indent=2,
        )

    if "npz" not in export_format:
        export_format = f"npz-{export_format}" if export_format else "npz"

    est_gb = estimate_memory_requirement(len(used_ids), process_res)
    progress(
        f"Loading model {model_name} on cuda:{gpu_id} (~{est_gb:.1f} GB est)...", 0.2
    )
    if ckpt_path:
        # Load DA3 + apply a fine-tuned checkpoint (head/LoRA/full trainable params)
        # for predicting on test scenes with a domain-specialized model.
        from .finetune.evaluate_ft import load_ft_model
        model = load_ft_model(model_name, ckpt_path, torch.device(f"cuda:{gpu_id}"))
    else:
        model = _get_model(model_name, gpu_id)
    cleanup_cuda_memory()

    progress(
        f"Running DA3 on {len(used_ids)} cameras @ res {process_res}...", 0.35
    )
    t0 = time.time()
    try:
        prediction = model.inference(
            image_paths,
            extrinsics=E,
            intrinsics=K,
            align_to_input_ext_scale=True,
            ref_view_strategy=ref_view_strategy,
            use_ray_pose=use_ray_pose,
            process_res=process_res,
            export_dir=export_dir,
            export_format=export_format,
        )
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        clear_model_cache(gpu_id)
        raise RuntimeError(
            f"CUDA out of memory ({exc}). Freed cached model on cuda:{gpu_id}. "
            "Try again on a freer GPU, fewer cameras, or a lower process_res."
        ) from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            clear_model_cache(gpu_id)
            raise RuntimeError(
                f"CUDA out of memory. Freed cached model on cuda:{gpu_id}. "
                "Try again on a freer GPU, fewer cameras, or a lower process_res."
            ) from exc
        raise
    elapsed = time.time() - t0

    npz_path = os.path.join(export_dir, "exports", "npz", "results.npz")
    progress("Finalizing export (npz)...", 0.85)
    _wait_for_file(npz_path)

    # Save the sky mask separately (the library npz does not include it) so the
    # fusion step can drop sky pixels for a much cleaner point cloud.
    sky_path = os.path.join(export_dir, "exports", "npz", "sky.npy")
    has_sky = getattr(prediction, "sky", None) is not None
    if has_sky:
        np.save(sky_path, prediction.sky.astype(bool))

    # Relocate scene.glb -> exports/scene.glb to match the required layout.
    glb_src = os.path.join(export_dir, "scene.glb")
    glb_dst = os.path.join(export_dir, "exports", "scene.glb")
    if os.path.isfile(glb_src):
        os.makedirs(os.path.dirname(glb_dst), exist_ok=True)
        shutil.move(glb_src, glb_dst)

    depth_shape = list(prediction.depth.shape)
    run_config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "root": root,
        "split": split,
        "scene": scene,
        "frame_id": frame_id,
        "model": model_name,
        "ckpt": ckpt_path,
        "process_res": process_res,
        "export_format": export_format,
        "gpu_id": gpu_id,
        "ref_view_strategy": ref_view_strategy,
        "use_ray_pose": use_ray_pose,
        "align_to_input_ext_scale": True,
        "is_metric": int(getattr(prediction, "is_metric", 0) or 0),
        "scale_factor": (
            float(prediction.scale_factor)
            if getattr(prediction, "scale_factor", None) is not None else None
        ),
        "has_sky_mask": bool(has_sky),
        "cameras": used_ids,
        "skipped_no_calibration": skipped,
        "num_cameras": len(used_ids),
        "depth_shape": depth_shape,
        "inference_seconds": round(elapsed, 2),
    }
    with open(os.path.join(meta_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    paths = {
        "run_dir": run_dir,
        "export_dir": export_dir,
        "npz": npz_path,
        "glb": glb_dst if os.path.isfile(glb_dst) else None,
        "frames_dir": frames_dir,
    }
    with open(os.path.join(meta_dir, "paths.json"), "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)

    # Optional zone mask: drop depth inside the per-camera zone polygons → sibling
    # ``*_zone`` export (selectable at Fuse) that keeps only out-of-zone depth.
    zone = None
    if apply_zone:
        progress("Applying zone mask...", 0.95)
        try:
            from . import zonemask
            zone = zonemask.apply(export_dir, scene_info["scene_dir"], zone_name,
                                  zone_dir=zone_dir)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            zone = {"error": str(exc)}

    progress("Done.", 1.0)
    return {
        "run_dir": run_dir,
        "export_dir": export_dir,
        "npz_path": npz_path,
        "glb_path": paths["glb"],
        "cameras": used_ids,
        "skipped": skipped,
        "depth_shape": depth_shape,
        "inference_seconds": round(elapsed, 2),
        "zone": zone,
    }
