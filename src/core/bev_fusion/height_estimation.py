import collections

import numpy as np
from loguru import logger

from core.bev_fusion.projection import (
	ANKLE_LEFT_IDX,
	ANKLE_RIGHT_IDX,
	image_to_world,
)
from core.utils.filter_ultilities import is_point_in_zone


# COCO-17 head keypoints (nose, eyes, ears): any one confident means the bbox
# top edge really is the top of the head, not a clipped shoulder/torso.
_HEAD_KEYPOINT_IDXS = (0, 1, 2, 3, 4)

# Near-zero closed-form denominator guard (mirrors projection._W3_EPS).
_DENOM_EPS = 1e-9

# Code-level defaults for the `mtmc.height_estimation` config block; the block
# (or any key in it) may be omitted per scene. `enabled: false` (or an absent
# block) keeps the pipeline output byte-identical to the constant-height path.
DEFAULT_HEIGHT_ESTIMATION = {
	"enabled"         : False,
	"min_height_m"    : 1.3,
	"max_height_m"    : 2.1,
	"top_margin_px"   : 8,
	"bottom_margin_px": 8,
	"min_track_score" : 0.5,
}


def resolve_height_estimation_cfg(raw: dict | None) -> dict:
	"""Merge a scene's ``height_estimation`` block over the code-level defaults.

	Shallow merge, mirroring how ``main()`` merges the ``mtmc:`` block over
	``DEFAULT_MTMC``; ``None`` (absent block) yields a copy of the defaults.

	Args:
		raw: The raw ``mtmc.height_estimation`` dict from the scene config, or
			``None`` when the block is absent.

	Returns:
		A new dict with every ``DEFAULT_HEIGHT_ESTIMATION`` key present.
	"""
	return {**DEFAULT_HEIGHT_ESTIMATION, **(raw or {})}


def estimate_height_from_bbox_top(camera_matrix, ground_xy, v_top) -> float | None:
	"""Closed-form height of a vertical segment whose top projects to ``v_top``.

	With ``P = K[R|t]`` and the segment base at world ``(X, Y, 0)``, the image
	row of the top at height ``h`` is ``v(h) = (a + b*h) / (c + d*h)`` where
	``a = P[1] @ [X, Y, 0, 1]``, ``b = P[1, 2]``, ``c = P[2] @ [X, Y, 0, 1]``
	and ``d = P[2, 2]``. Solving ``v(h) = v_top`` gives
	``h = (v_top*c - a) / (b - v_top*d)`` -- the exact inverse of the forward
	model in :func:`matching.core.bev_fusion.projection.pole_pixel_height`.

	Args:
		camera_matrix: The 3x4 projection matrix ``P = K @ [R|t]``, or ``None``.
		ground_xy: World ``(X, Y)`` of the segment base on the Z=0 plane.
		v_top: Image pixel row of the segment's top (the bbox top edge).

	Returns:
		The raw height in meters (range filtering lives in the sampler), or
		``None`` when the denominator is near zero (degenerate viewing
		geometry) or any input/result is non-finite.
	"""
	if camera_matrix is None:
		return None
	p     = np.asarray(camera_matrix, dtype=np.float64)
	v_top = float(v_top)
	base  = np.array([float(ground_xy[0]), float(ground_xy[1]), 0.0, 1.0])
	if not (np.isfinite(v_top) and np.all(np.isfinite(base)) and np.all(np.isfinite(p))):
		return None
	a = float(p[1] @ base)
	b = float(p[1, 2])
	c = float(p[2] @ base)
	d = float(p[2, 2])
	denominator = b - v_top * d
	if abs(denominator) < _DENOM_EPS:
		return None
	height = (v_top * c - a) / denominator
	if not np.isfinite(height):
		return None
	return float(height)


class PersonHeightSampler:
	"""Per-detection height sampler with quality gates and scene diagnostics.

	One instance lives for a whole scene run. :meth:`compute_sample` runs the
	quality gates on a single-camera person detection and, on acceptance,
	returns the closed-form height so the caller can attach it to that frame's
	``CamObservation``. Accepted samples are also recorded per scene and per
	camera FOR LOGGING ONLY (:meth:`log_summary`); they never feed the output
	-- the written height is reduced per frame from the observation members.
	"""

	def __init__(self, cfg: dict | None, ankle_conf_thresh: float,
				 head_conf_thresh: float):
		"""Resolve the config block and reset all counters.

		Args:
			cfg: Raw ``mtmc.height_estimation`` block (see
				:data:`DEFAULT_HEIGHT_ESTIMATION`); ``None`` uses the defaults.
			ankle_conf_thresh: Min keypoint score for BOTH ankles (reuses
				``mtmc.ankle_conf_thresh``); guarantees the ground point is
				independent of the class-constant tracking anchor.
			head_conf_thresh: Min score for at least one head keypoint (reuses
				``mtmc.pose_conf_thresh``); ensures the bbox top is the head.
		"""
		self.cfg               = resolve_height_estimation_cfg(cfg)
		self.ankle_conf_thresh = float(ankle_conf_thresh)
		self.head_conf_thresh  = float(head_conf_thresh)
		self.min_height_m      = float(self.cfg["min_height_m"])
		self.max_height_m      = float(self.cfg["max_height_m"])
		self.top_margin_px     = float(self.cfg["top_margin_px"])
		self.bottom_margin_px  = float(self.cfg["bottom_margin_px"])
		self.min_track_score   = float(self.cfg["min_track_score"])

		self.rejections        = collections.Counter()
		self.accepted          = 0
		self.scene_samples     = []  # diagnostics only, never used in output
		self.samples_by_camera = collections.defaultdict(list)

	def compute_sample(self, instance: dict, camera_name: str, camera_matrix,
					   homography, img_w: int, img_h: int,
					   zone_polygons: list | None = None,
					   zone_dist_thresh: float | None = None) -> float | None:
		"""Run the quality gates on one detection; return its height sample.

		Gates, each rejection counted by reason: ``no_bbox``/``low_score``
		(``min_track_score`` of 0 disables the score gate), ``truncated``
		(bbox touches the top/bottom image margins), ``no_pose``/
		``ankles_unconfident`` (BOTH ankles required), ``no_head_kp``,
		``invalid_ground``, ``out_of_zone``, ``degenerate``, ``out_of_range``
		(rejected, not clamped).

		Args:
			instance: One tracked instance dict; reads ``track_bbox``
				(normalized ``cx, cy, w, h``), ``track_score`` and
				``pose_keypoints`` (COCO-17 ``{"x", "y", "score"}`` list).
			camera_name: Source camera id, used for per-camera diagnostics.
			camera_matrix: The camera's 3x4 ``P = K @ [R|t]``.
			homography: The camera's 3x3 image->world(Z=0) homography.
			img_w: Image width in pixels.
			img_h: Image height in pixels.
			zone_polygons: The camera's BEV zone list (same objects the
				tracker path gates on); ``None``/empty disables the zone gate.
			zone_dist_thresh: Max distance to a zone polygon still counted as
				inside (``mtmc.distance_to_polygon_thresh``); ``None`` uses
				the :func:`is_point_in_zone` default.

		Returns:
			The accepted height sample in meters, or ``None`` when any gate
			rejected the detection.
		"""
		bbox = instance.get("track_bbox")
		if bbox is None:
			return self._reject("no_bbox")
		if self.min_track_score > 0:
			score = instance.get("track_score")
			if score is None or float(score) < self.min_track_score:
				return self._reject("low_score")

		center_x, center_y, box_w, box_h = (float(v) for v in bbox)
		top_px    = (center_y - box_h / 2.0) * img_h
		bottom_px = (center_y + box_h / 2.0) * img_h
		if top_px < self.top_margin_px or bottom_px > img_h - self.bottom_margin_px:
			return self._reject("truncated")

		pose = instance.get("pose_keypoints")
		if (not isinstance(pose, (list, tuple))
				or len(pose) <= max(ANKLE_LEFT_IDX, ANKLE_RIGHT_IDX)):
			return self._reject("no_pose")
		left, right = pose[ANKLE_LEFT_IDX], pose[ANKLE_RIGHT_IDX]
		if (float(left["score"]) < self.ankle_conf_thresh
				or float(right["score"]) < self.ankle_conf_thresh):
			return self._reject("ankles_unconfident")

		if not any(float(pose[i]["score"]) >= self.head_conf_thresh
				   for i in _HEAD_KEYPOINT_IDXS):
			return self._reject("no_head_kp")

		ankle_mid_px = np.array([
			(float(left["x"]) + float(right["x"])) / 2.0,
			(float(left["y"]) + float(right["y"])) / 2.0,
		])
		ground_xy, valid = image_to_world(homography, ankle_mid_px)
		if not valid:
			return self._reject("invalid_ground")

		# Same gate as the tracker path: only applied when the camera has a
		# non-empty zone list.
		if zone_polygons:
			in_zone = (is_point_in_zone(zone_polygons, ground_xy, zone_dist_thresh)
					   if zone_dist_thresh is not None
					   else is_point_in_zone(zone_polygons, ground_xy))
			if not in_zone:
				return self._reject("out_of_zone")

		height = estimate_height_from_bbox_top(camera_matrix, ground_xy, top_px)
		if height is None:
			return self._reject("degenerate")
		if not self.min_height_m <= height <= self.max_height_m:
			return self._reject("out_of_range")

		self.accepted += 1
		self.scene_samples.append(height)
		self.samples_by_camera[camera_name].append(height)
		return height

	def log_summary(self) -> None:
		"""Log acceptance/rejection counts and the scene/per-camera statistics.

		Cross-camera median agreement is the primary no-GT sanity signal; the
		scene median landing near an independently hand-tuned value validates
		the method.
		"""
		total_rejected = sum(self.rejections.values())
		logger.info(f"[height] samples: accepted={self.accepted} "
					f"rejected={total_rejected} "
					f"by_reason={dict(self.rejections)}")
		if not self.scene_samples:
			logger.warning("[height] no accepted samples; every person frame "
						   "falls back to the class-default height.")
			return
		samples = np.asarray(self.scene_samples, dtype=np.float64)
		logger.info(f"[height] scene: n={samples.size} "
					f"median={np.median(samples):.3f} mean={samples.mean():.3f} "
					f"std={samples.std():.3f}")
		for camera_name in sorted(self.samples_by_camera):
			cam = np.asarray(self.samples_by_camera[camera_name], dtype=np.float64)
			logger.info(f"[height]   {camera_name}: n={cam.size} "
						f"median={np.median(cam):.3f}")

	def _reject(self, reason: str) -> None:
		"""Count one rejection under ``reason`` and return ``None``."""
		self.rejections[reason] += 1
		return None
