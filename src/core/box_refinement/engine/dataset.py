
from __future__ import annotations

import json
import os
from glob import glob

import cv2
import h5py
import numpy as np

from . import config
from .calib import camera_sort_key, extract_camera_id_from_filename


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #
def _is_scene_dir(path: str) -> bool:
	return os.path.isdir(os.path.join(path, "videos")) or os.path.isfile(
		os.path.join(path, "calibration.json")
	)


def scan_root(root: str) -> dict:
	"""List the splits that actually exist and their scene folders."""
	root = os.path.abspath(os.path.expanduser(root))
	splits = []
	if os.path.isdir(root):
		for name in config.SPLIT_CANDIDATES:
			split_dir = os.path.join(root, name)
			if not os.path.isdir(split_dir):
				continue
			scenes = sorted(
				d
				for d in os.listdir(split_dir)
				if _is_scene_dir(os.path.join(split_dir, d))
			)
			splits.append({"name": name, "scenes": scenes})
	return {"root": root, "exists": os.path.isdir(root), "splits": splits}


def _probe_video(path: str) -> dict:
	cap = cv2.VideoCapture(path)
	try:
		info = {
			"num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
			"fps": float(cap.get(cv2.CAP_PROP_FPS)),
			"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
			"height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
		}
	finally:
		cap.release()
	return info


def _list_camera_videos(videos_dir: str) -> list[tuple[str, str]]:
	"""Return [(camera_id, video_path), ...] sorted by camera number."""
	vids = []
	for ext in ("*.mp4", "*.MP4", "*.avi", "*.mkv"):
		vids.extend(glob(os.path.join(videos_dir, ext)))
	out = []
	for v in sorted(vids, key=camera_sort_key):
		cam_id = extract_camera_id_from_filename(v)
		if cam_id is not None:
			out.append((cam_id, v))
	return out


def scan_scene(root: str, split: str, scene: str) -> dict:
	scene_dir = os.path.join(root, split, scene)
	if not os.path.isdir(scene_dir):
		raise FileNotFoundError(f"Scene dir not found: {scene_dir}")

	videos_dir = os.path.join(scene_dir, "videos")
	depth_dir  = os.path.join(scene_dir, "depth_maps")
	calib_path = os.path.join(scene_dir, "calibration.json")
	gt_path    = os.path.join(scene_dir, "ground_truth.json")
	map_path   = os.path.join(scene_dir, "map.png")

	warnings = []

	# Calibration sensor ids.
	calib_ids: set[str] = set()
	if os.path.isfile(calib_path):
		try:
			with open(calib_path, "r", encoding="utf-8") as f:
				calib = json.load(f)
			calib_ids = {
				s["id"] for s in calib.get("sensors", []) if s.get("type") == "camera"
			}
		except Exception as exc:  # noqa: BLE001
			warnings.append(f"Failed to parse calibration.json: {exc}")
	else:
		warnings.append("calibration.json missing")

	cam_videos = _list_camera_videos(videos_dir) if os.path.isdir(videos_dir) else []
	if not cam_videos:
		warnings.append("no camera videos found")

	has_gt_depth = os.path.isdir(depth_dir)
	cameras = []
	for cam_id, vpath in cam_videos:
		cam_gt = has_gt_depth and os.path.isfile(
			os.path.join(depth_dir, f"{cam_id}.h5")
		)
		cameras.append(
			{
				"id": cam_id,
				"video": vpath,
				"in_calib": cam_id in calib_ids,
				"has_gt_depth": cam_gt,
			}
		)

	probe = _probe_video(cam_videos[0][1]) if cam_videos else {
		"num_frames": 0,
		"fps": 0.0,
		"width": 0,
		"height": 0,
	}


	return {
		"split": split,
		"scene": scene,
		"scene_dir": scene_dir,
		"has_calibration": os.path.isfile(calib_path),
		"calibration_path": calib_path if os.path.isfile(calib_path) else None,
		"has_gt_depth": has_gt_depth and any(c["has_gt_depth"] for c in cameras),
		"has_gt_annotation": os.path.isfile(gt_path),
		"ground_truth_path": gt_path if os.path.isfile(gt_path) else None,
		"has_map": os.path.isfile(map_path),
		"cameras": cameras,
		"warnings": warnings,
		**probe,
	}


# --------------------------------------------------------------------------- #
# Output path helpers
# --------------------------------------------------------------------------- #
def frame_run_dir(output_root: str, split: str, scene: str, frame_id: int) -> str:
	return os.path.join(output_root, split, scene, f"frame_{frame_id}")




# --------------------------------------------------------------------------- #
# Single-frame extraction (one seek + read per camera; never loops the video)
# --------------------------------------------------------------------------- #
def extract_one(video_path: str, frame_id: int) -> np.ndarray | None:
	"""Read exactly one frame (BGR) from ``video_path`` at ``frame_id``."""
	cap = cv2.VideoCapture(video_path)
	try:
		total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		if total and frame_id >= total:
			return None
		cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_id))
		ok, frame = cap.read()
		if not ok or frame is None:
			return None
		return frame
	finally:
		cap.release()


def extract_frame(
	root: str,
	split: str,
	scene: str,
	frame_id: int,
	cameras: list[str] | None,
	output_root: str,
) -> dict:
	"""Extract a single synchronized frame from each selected camera."""
	info = scan_scene(root, split, scene)
	cam_map = {c["id"]: c for c in info["cameras"]}

	if cameras is None or cameras == "all" or cameras == ["all"]:
		selected = [c["id"] for c in info["cameras"]]
	else:
		selected = [c for c in cameras if c in cam_map]

	if not selected:
		raise ValueError("No valid cameras selected.")

	num_frames = info.get("num_frames") or 0
	if num_frames and (frame_id < 0 or frame_id >= num_frames):
		raise ValueError(
			f"frame_id {frame_id} out of range [0, {num_frames - 1}]."
		)

	run_dir = frame_run_dir(output_root, split, scene, frame_id)
	frames_dir = os.path.join(run_dir, "frames")
	os.makedirs(frames_dir, exist_ok=True)

	results = []
	warnings = list(info.get("warnings", []))
	for cam_id in selected:
		vpath = cam_map[cam_id]["video"]
		out_path = os.path.join(frames_dir, f"{cam_id}.jpg")
		if os.path.isfile(out_path):
			results.append({"camera": cam_id, "frame_path": out_path, "cached": True})
			continue
		frame = extract_one(vpath, frame_id)
		if frame is None:
			warnings.append(f"{cam_id}: failed to read frame {frame_id}")
			continue
		cv2.imwrite(out_path, frame)
		results.append({"camera": cam_id, "frame_path": out_path, "cached": False})

	return {
		"run_dir": run_dir,
		"frames_dir": frames_dir,
		"frame_id": frame_id,
		"frames": results,
		"warnings": warnings,
	}


# --------------------------------------------------------------------------- #
# GT depth (read a single frame from the per-camera HDF5)
# --------------------------------------------------------------------------- #
def gt_depth_h5_path(root: str, split: str, scene: str, camera_id: str) -> str:
	return os.path.join(root, split, scene, "depth_maps", f"{camera_id}.h5")


def read_gt_depth(h5_path: str, frame_id: int) -> np.ndarray:
	"""Return GT depth in meters with NaN at invalid/sky pixels (H, W) float32."""
	key = f"distance_to_image_plane_{frame_id:05d}.png"
	with h5py.File(h5_path, "r") as f:
		if key not in f:
			raise KeyError(f"{key} not in {h5_path}")
		raw = f[key][()].astype(np.float32)
	invalid = (raw == config.GT_DEPTH_INVALID) | (raw == config.GT_DEPTH_SKY)
	depth_m = raw / config.GT_DEPTH_SCALE
	depth_m[invalid] = np.nan
	return depth_m
