from typing import Optional

import numpy as np
from loguru import logger

from core.bev_fusion.lift3d import CLASS_DIMENSIONS


# image_to_world validity guards.
_W3_EPS      = 1e-9     # near-zero homogeneous component => point at/over horizon
_MAX_WORLD_M = 1.0e3    # sane upper bound on |world XY| in meters


def camera_matrix_from_intrinsic_extrinsic(intrinsic, extrinsic) -> np.ndarray:
	"""Build the 3x4 camera projection matrix ``P = K @ [R|t]``.

	``intrinsic`` is the 3x3 ``intrinsicMatrix`` (K) and ``extrinsic`` is the
	3x4 ``extrinsicMatrix`` ([R|t], world->camera). Their product is the
	world->image projection matrix used for ground projection. Building it here
	keeps ``intrinsicMatrix``/``extrinsicMatrix`` as the single source of truth
	instead of the (sometimes inconsistent) stored ``cameraMatrix`` field.
	"""
	k  = np.asarray(intrinsic, dtype=np.float64)
	rt = np.asarray(extrinsic, dtype=np.float64)
	if k.shape != (3, 3):
		raise ValueError(f"intrinsicMatrix must be 3x3, got {k.shape}")
	if rt.shape != (3, 4):
		raise ValueError(f"extrinsicMatrix must be 3x4, got {rt.shape}")
	return k @ rt


def ground_homography_from_camera_matrix(camera_matrix) -> np.ndarray:
	"""Build the image->world(Z=0) homography from a 3x4 ``cameraMatrix``.

	For a ground point (X, Y, 0, 1): image = cameraMatrix @ [X, Y, 0, 1]^T
	= P_ground @ [X, Y, 1]^T with P_ground = cameraMatrix[:, [0, 1, 3]].
	The inverse maps an image pixel (u, v, 1) back to (X, Y, 1) on Z=0.
	"""
	camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
	if camera_matrix.shape != (3, 4):
		raise ValueError(f"cameraMatrix must be 3x4, got {camera_matrix.shape}")
	p_ground = camera_matrix[:, [0, 1, 3]]
	return np.linalg.inv(p_ground)


def load_homographies(calib_dict: dict, camera_order: list) -> tuple:
	"""Return ``(homographies, camera_matrices)`` for cameras present in BOTH
	``camera_order`` (those with single-camera output) and ``calib_dict``.

	``homographies`` maps ``camera_name -> H(3x3)`` ground (Z=0) homographies and
	``camera_matrices`` maps ``camera_name -> P(3x4)`` projection matrices, both
	built from each camera's ``intrinsicMatrix`` and ``extrinsicMatrix``
	(``P = K @ [R|t]``). The camera matrix is needed to project an object's world
	height into image space (see :func:`pole_pixel_height`). Mismatches are logged.
	"""
	homographies = {}
	camera_matrices = {}
	for cam in camera_order:
		entry = calib_dict.get(cam)
		if entry is None or entry.get("intrinsicMatrix") is None or entry.get("extrinsicMatrix") is None:
			logger.warning(f"No intrinsic/extrinsic calibration for camera {cam!r}; skipping it in fusion.")
			continue
		camera_matrix = camera_matrix_from_intrinsic_extrinsic(
			entry["intrinsicMatrix"], entry["extrinsicMatrix"])
		homographies[cam] = ground_homography_from_camera_matrix(camera_matrix)
		camera_matrices[cam] = camera_matrix

	missing_output = [c for c in calib_dict if c not in set(camera_order)]
	if missing_output:
		logger.warning(f"Cameras with calibration but no single-camera output (ignored): {missing_output}")
	return homographies, camera_matrices


def image_to_world(H: np.ndarray, uv: np.ndarray) -> tuple:
	"""Map an image pixel ``uv=(u,v)`` to world ground ``(X, Y)`` via ``H``.

	Returns ``(xy(2,), valid)``. ``valid`` is False when the homogeneous
	component is near zero (horizon/degenerate), the result is non-finite, or
	``|xy|`` exceeds a sane bound.
	"""
	H = np.asarray(H, dtype=np.float64)
	p = H @ np.array([float(uv[0]), float(uv[1]), 1.0], dtype=np.float64)
	w3 = p[2]
	if not np.isfinite(w3) or abs(w3) < _W3_EPS:
		return np.array([np.nan, np.nan]), False
	xy    = p[:2] / w3
	valid = bool(np.all(np.isfinite(xy)) and np.linalg.norm(xy) <= _MAX_WORLD_M)
	return xy, valid

# COCO-17 keypoint indices used for the ground anchor.
# {0: 'Nose', 1: 'L_Eye', 2: 'R_Eye', 3: 'L_Ear', 4: 'R_Ear',
#  5: 'L_Shoulder', 6: 'R_Shoulder', 7: 'L_Elbow', 8: 'R_Elbow', 9: 'L_Wrist', 10: 'R_Wrist',
#  11: 'L_Hip', 12: 'R_Hip', 13: 'L_Knee', 14: 'R_Knee', 15: 'L_Ankle', 16: 'R_Ankle'}
HIP_LEFT_IDX,   HIP_RIGHT_IDX   = 11, 12
KNEE_LEFT_IDX,  KNEE_RIGHT_IDX  = 13, 14
ANKLE_LEFT_IDX, ANKLE_RIGHT_IDX = 15, 16

# Ankle-estimation tunables.
_MIN_CONF_KEYPOINT_FOR_ESTIMATE = 8     # require > this many confident keypoints to trust the skeleton
_SHANK_THIGH_RATIO              = 1.0   # ankle ~= knee + ratio*(knee - hip); shank ~= thigh for a straight leg


def get_confident_xy(pose: list, index_keypoint: int, conf_thresh: float) -> Optional[np.ndarray]:
	"""Return ``[x, y]`` for keypoint ``idx`` when its score passes ``conf_thresh``, else ``None``."""
	keypoint = pose[index_keypoint]
	if float(keypoint["score"]) >= conf_thresh:
		return np.array([float(keypoint["x"]), float(keypoint["y"])])
	return None


def estimate_extrapolate_ankle(hip: Optional[np.ndarray],
					   knee: Optional[np.ndarray]) -> Optional[np.ndarray]:
	"""Extrapolate the ankle along the hip->knee leg vector.

	``ankle ~= knee + ratio*(knee - hip)``. Requires both hip and knee (returns
	``None`` otherwise). In image space (y grows downward) the hip sits above the
	knee, so the result lands below the knee.
	"""
	if hip is None or knee is None:
		return None
	return knee + _SHANK_THIGH_RATIO * (knee - hip)


def estimate_ankle_px(pose: list, pose_conf_thresh: float,
					  min_conf_keypoint_for_estimate: int = _MIN_CONF_KEYPOINT_FOR_ESTIMATE) -> Optional[np.ndarray]:
	"""Estimate the ankle ground pixel from the leg kinematic chain.

	Used when neither ankle keypoint is confident. Gated on a reliable skeleton
	(more than ``min_conf_keypoint_for_estimate`` keypoints scoring ``>= pose_conf_thresh``).
	For each leg whose hip AND knee are confident the ankle is extrapolated as
	``knee + (knee - hip)``; the midpoint of the usable legs is returned, or
	``None`` if the skeleton is too sparse or neither leg is usable.

	Args:
		pose (list): the list of keypoints for this instance.
		pose_conf_thresh (float): the confidence threshold for a keypoint to be considered reliable.
		min_conf_keypoint_for_estimate (int): require more than this many confident keypoints to
			trust the skeleton. Defaults to ``_MIN_CONF_KEYPOINT_FOR_ESTIMATE``.
	Returns:
		(Optional[np.ndarray]): the estimated ankle pixel coordinates as a numpy array, or None if estimation is not possible.
	"""
	num_keypoint_fulfill_conf = sum(1 for kp in pose if float(kp["score"]) >= pose_conf_thresh)
	if num_keypoint_fulfill_conf <= min_conf_keypoint_for_estimate:
		return None

	legs = [
		estimate_extrapolate_ankle(get_confident_xy(pose, HIP_LEFT_IDX,  pose_conf_thresh),
						   get_confident_xy(pose, KNEE_LEFT_IDX,  pose_conf_thresh)),
		estimate_extrapolate_ankle(get_confident_xy(pose, HIP_RIGHT_IDX, pose_conf_thresh),
						   get_confident_xy(pose, KNEE_RIGHT_IDX, pose_conf_thresh)),
	]
	estimates = [a for a in legs if a is not None]

	if not estimates:
		return None
		
	return np.mean(estimates, axis=0)


def pole_pixel_height(camera_matrix: np.ndarray, ground_xy: np.ndarray,
					  height_m: float) -> Optional[float]:
	"""Image-space pixel height of a vertical segment of ``height_m`` meters.

	The segment's base sits at world ground point ``ground_xy`` (Z=0); its base
	and top are projected with ``camera_matrix`` (``P = K[R|t]``) and the absolute
	pixel distance between the two image rows is returned. Yields ``None`` when an
	endpoint is degenerate (near the horizon / behind the camera).

	Args:
		camera_matrix (np.ndarray): the 3x4 projection matrix ``P = K @ [R|t]``.
		ground_xy (np.ndarray): world ``(X, Y)`` of the segment base on Z=0.
		height_m (float): segment length in meters (the object's world height).
	Returns:
		(Optional[float]): the segment's pixel height, or None if degenerate.
	"""
	p    = np.asarray(camera_matrix, dtype=np.float64)
	x, y = float(ground_xy[0]), float(ground_xy[1])
	base = p @ np.array([x, y, 0.0,             1.0])
	top  = p @ np.array([x, y, float(height_m), 1.0])
	ok   = (np.isfinite(base[2]) and np.isfinite(top[2])
		  and abs(base[2]) >= _W3_EPS and abs(top[2]) >= _W3_EPS)
	if not ok:
		return None
	return abs(base[1] / base[2] - top[1] / top[2])


def ground_anchor_px(instance: dict, img_w: int, img_h: int,
					 ankle_conf_thresh: float, 
					 pose_conf_thresh: float,
					 min_conf_keypoint_for_estimate: int = _MIN_CONF_KEYPOINT_FOR_ESTIMATE,
					 camera_matrix: Optional[np.ndarray] = None, 
					 projection_type: dict = {}, bbox_ratio_filter: dict = {},
					 skeleton_filter: list = []) -> Optional[np.ndarray]:
	"""Pick the ground-contact pixel on image for an instance.

	The anchoring strategy is selected by the instance's ``class_id`` through
	``projection_type``, a dict mapping a strategy name to the list of class ids
	that use it. Three strategies are recognized:

	``"skeleton"`` (persons / humanoids carrying ``pose_keypoints``): the
	midpoint of the confident left/right ankle keypoints (COCO-17 indices
	15/16); the single confident ankle if only one passes; otherwise the ankle
	estimated from the leg kinematic chain via :func:`estimate_ankle_px`. A pose
	too short to hold both ankles returns ``None``; this path never falls back to
	the bbox.

	``"center_point"``: the bbox center itself -- the normalized ``track_bbox``
	``(center_x, center_y)`` scaled to pixels, with no ground correction.

	``"top_bottom"``: anchor on the ``track_bbox``, corrected by image half
	(chosen by the bbox center ``center_y``). A box in the TOP half (far,
	near-horizontal view) keeps the bbox bottom-center -- its base is the
	reliable ground contact. A box in the BOTTOM half (close, steep top-down
	view, base often occluded/clipped) anchors on the bbox top edge and drops
	DOWN by the object's image-space pixel height: the world class height
	``height_z`` (:data:`CLASS_DIMENSIONS`) projected through ``camera_matrix``
	(``P = K[R|t]``), seeded at the footprint under the bbox bottom-center (see
	:func:`pole_pixel_height`) and clamped within the box. An unknown/missing
	class, absent ``camera_matrix``, or degenerate seed falls back to
	bottom-center.

	A ``class_id`` listed under no strategy -- or one whose required pose/bbox
	data is missing -- yields ``None``.

	Args:
		instance (dict): one tracked instance. Reads ``pose_keypoints`` (list of
			COCO-17 ``{"x", "y", "score"}`` keypoints, or ``None``), ``class_id``
			(int strategy key), and ``track_bbox`` (normalized ``cx, cy, w, h``).
		img_w (int): image width in pixels; denormalizes the anchor/bbox x.
		img_h (int): image height in pixels; denormalizes the anchor/bbox y.
		ankle_conf_thresh (float): min keypoint score to trust an ankle keypoint
			in the ``"skeleton"`` strategy.
		pose_conf_thresh (float): min keypoint score for the skeleton-gate count
			and the hip/knee leg-chain ankle estimate.
		min_conf_keypoint_for_estimate (int): require more than this many confident
			keypoints before trusting the skeleton to extrapolate an ankle.
			Defaults to ``_MIN_CONF_KEYPOINT_FOR_ESTIMATE``.
		camera_matrix (Optional[np.ndarray]): 3x4 projection matrix ``P = K[R|t]``
			used by the ``"top_bottom"`` pole-height drop. ``None`` disables the
			correction (-> bottom-center). Defaults to ``None``.
		projection_type (dict): maps a strategy name (``"skeleton"``,
			``"center_point"``, ``"top_bottom"``) to the list of class ids that use
			it. Required (see Raises).

	Returns:
		(Optional[np.ndarray]): the ground-contact pixel ``[u, v]`` as a numpy
			array, or ``None`` if no anchor is derivable for this instance.

	Raises:
		AttributeError: If ``projection_type`` is ``None`` (no strategy map given).

	Example:
		>>> projection_type = {"skeleton": [0, 4],
		...                    "center_point": [2, 3],
		...                    "top_bottom": [1, 5, 6]}
		>>> ground_anchor_px(instance, 1920, 1080, 0.5, 0.3,
		...                  camera_matrix=P, projection_type=projection_type)
		array([960., 720.])
	"""
	pose        = instance.get("pose_keypoints") if instance.get("pose_keypoints") is not None else None
	class_id    = instance.get("class_id")
	ankle_left  = None
	ankle_right = None

	# get bounding box information
	bbox = instance.get("track_bbox")
	if bbox is None:
		logger.warning(f"Instance with class_id {class_id} has no track_bbox; skipping anchor.")
		return None
	bbox_center_x, bbox_center_y, bbox_w, bbox_h = (float(v) for v in bbox)
	bbox_ratio               = bbox_w / bbox_h if bbox_h != 0 else 0.0

	# check if the instance has at least one confident ankle keypoint.
	is_have_ankle = False
	if isinstance(pose, (list, tuple)) and len(pose) >= max(ANKLE_LEFT_IDX, ANKLE_RIGHT_IDX):
		ankle_left  = pose[ANKLE_LEFT_IDX]
		ankle_right = pose[ANKLE_RIGHT_IDX]
		l_ok  = float(ankle_left["score"])  >= ankle_conf_thresh
		r_ok  = float(ankle_right["score"]) >= ankle_conf_thresh
		if l_ok or r_ok:
			is_have_ankle = True

	# get class_id list for each projection type
	skeleton_classes     = projection_type.get("skeleton", [])
	center_point_classes = projection_type.get("center_point", [])
	top_bottom_classes   = projection_type.get("top_bottom", [])
	
	# DEBUG: Filter ratio of bounding box, only for scene 023-024
	if bbox_ratio_filter is not None and isinstance(bbox_ratio_filter, dict):
		for class_id_index, ratio in bbox_ratio_filter.items():
			if class_id == class_id_index and bbox_ratio > ratio:
				# logger.warning(f"Instance with class_id {class_id} has bbox ratio {bbox_ratio:.3f} exceeding threshold {ratio:.3f}; skipping anchor.")
				return None

	# DEBUG: only for scene 023-024
	if skeleton_filter is not None and class_id in skeleton_filter:
		if not is_have_ankle:
			# logger.warning(f"Instance with class_id {class_id} has no confident ankle keypoints; skipping anchor.")
			return None
		# DEBUG: filter - check if the any ankle keypoint out of bounding box
		# bottom_y  = (bbox_center_y + bbox_h / 2.0) * img_h # y value of the ground-contact pixel on image
		# if (ankle_left is not None and ankle_left["y"] > bottom_y) or (ankle_right is not None and ankle_right["y"] > bottom_y):
		# 	return None

	if skeleton_classes is not None and class_id in skeleton_classes:
		# NOTE: Lv1 - For using ankle keypoint as the anchor, 
		# we need to have a pose and skeleton projection enabled.
		# Persons|FourierGR1T2: anchor from the ankle keypoints (or estimate from the leg chain).
		if isinstance(pose, (list, tuple)):
			# If the pose is too short to contain both ankle keypoints, 
			# we cannot anchor from them.
			if len(pose) <= max(ANKLE_LEFT_IDX, ANKLE_RIGHT_IDX):
				return None

			ankle_left  = pose[ANKLE_LEFT_IDX]
			ankle_right = pose[ANKLE_RIGHT_IDX]
			l_ok  = float(ankle_left["score"])  >= ankle_conf_thresh
			r_ok  = float(ankle_right["score"]) >= ankle_conf_thresh
			if l_ok or r_ok:
				return np.array([(float(ankle_left["x"]) + float(ankle_right["x"])) / 2.0,
								 (float(ankle_left["y"]) + float(ankle_right["y"])) / 2.0])
			
			return estimate_ankle_px(pose, pose_conf_thresh, min_conf_keypoint_for_estimate)

	elif center_point_classes is not None and class_id in center_point_classes:
		# NOTE: Lv1 - For anchor from the center point of the track bbox (normalized cx, cy, w, h).
		anchor_x = bbox_center_x * img_w  # x value of the ground-contact pixel on image
		anchor_y = bbox_center_y * img_h  # y value of the ground-contact pixel on image
		return np.array([anchor_x, anchor_y])
		
	elif top_bottom_classes is not None and class_id in top_bottom_classes:
		# NOTE: Lv1 - For anchor from the track bbox (normalized cx, cy, w, h).
		anchor_x  = bbox_center_x * img_w  # x value of the ground-contact pixel on image
		bottom_y  = (bbox_center_y + bbox_h / 2.0) * img_h # y value of the ground-contact pixel on image
		top_y     = (bbox_center_y - bbox_h / 2.0) * img_h # y value of the top edge of the bbox on image

		# matrix disables the correction (-> bottom-center).
		raw_class_id = instance.get("class_id")
		dims         = CLASS_DIMENSIONS.get(int(raw_class_id)) if raw_class_id is not None else None

		# NOTE: Lv2 - Per-class world height (meters); an unknown/missing class or absent camera
		# For get the ground anchor from the bottom-up of bbox
		# Top half (far, near-horizontal view), unknown class, or no calibration: the
		# bbox bottom edge is the reliable ground contact -> bottom-center, as before.
		if bbox_center_y < 0.5 or dims is None or camera_matrix is None:
			return np.array([anchor_x, bottom_y])

		# NOTE: Lv2 - For get the ground anchor from the top-down of bbox
		# Bottom half (close, steep top-down view): the base is often occluded or
		# clipped, so anchor on the reliable top edge and drop DOWN by the object's
		# IMAGE-space pixel height. That height is the world class height projected
		# through P = K[R|t], seeded at the footprint under the bbox bottom-center.
		
		seed_xy, valid = image_to_world(
			ground_homography_from_camera_matrix(camera_matrix),
			np.array([anchor_x, bottom_y]))
		if not valid:
			return np.array([anchor_x, bottom_y])

		height_z = dims.height
		pole_px  = pole_pixel_height(camera_matrix, seed_xy, height_z)
		if pole_px is None:
			return np.array([anchor_x, bottom_y])

		# Clamp within the box so a degenerate seed can't push past the bbox bottom.
		return np.array([anchor_x, max(top_y + pole_px, bottom_y)])

	return None
