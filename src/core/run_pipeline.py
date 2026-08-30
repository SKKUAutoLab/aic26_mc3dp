import argparse
import glob
import json
import os
import pickle
import tempfile
from copy import deepcopy
from types import SimpleNamespace

import yaml

import cv2
import numpy as np
import torch
from loguru import logger
from shapely.geometry import box
from tqdm import tqdm

from torchreid.scripts.builder import build_config
from torchreid.scripts.default_config import get_default_config
from torchreid.tools.feature_extractor import KPRFeatureExtractor

from botsort.basetrack import BaseTrack
from botsort.bot_sort import BoTSORT

from core.bev_fusion.bev_fusion import BEVFusionTracker, CamObservation
from core.bev_fusion.bev_fusion_nor import BEVFusionTrackerNOR
from core.bev_fusion.height_estimation import (
	PersonHeightSampler,
	resolve_height_estimation_cfg,
)
from core.bev_fusion.lift3d import format_row, lift_to_3d
from core.bev_fusion.projection import (
	ground_anchor_px,
	image_to_world,
	load_homographies,
)
from core.data.class_label import ClassLabels
from core.detectors import load_detectors
from core.pose_estimater import load_pose_estimator
from core.utils.bbox import (
	bbox_cxcywh_norm_to_xywh,
	bbox_cxcywh_norm_to_xyxy,
	bbox_xyxy_to_cxcywh_norm,
)
from core.utils.filter_ultilities import (
	is_bbox_in_zone,
	is_point_in_zone,
	load_zone_bev,
	load_zone_camera,
)
from core.box_refinement import BoxRefiner


def parse_args():
	"""Parse command-line arguments.

	`--timestep-dir`, `--images` and `--image` are mutually exclusive; with none
	of them the full scene named by the config is processed.

	Returns:
		argparse.Namespace: The parsed arguments.
	"""
	parser = argparse.ArgumentParser(
		description="Run the whole MTMC pipeline in one process, frame-major.")
	parser.add_argument("--config", type=str, required=True,
						help="Path to the scene configuration file.")
	group = parser.add_mutually_exclusive_group()
	group.add_argument("--timestep-dir", type=str, default=None,
					   help="Directory holding one image per camera = one timestep, N views.")
	group.add_argument("--images", type=str, default=None,
					   help="Folder of frames from a single camera.")
	group.add_argument("--image", type=str, default=None,
					   help="A single image file.")
	parser.add_argument("--camera-name", type=str, default=None,
						help="Camera identity for --image/--images; only useful when it "
							 "matches a sensor id in data.calibration.file.")
	parser.add_argument("--cameras", type=str, nargs="+", default=None,
						help="Restrict a scene run to a subset of cameras.")
	parser.add_argument("--max-timesteps", type=int, default=None,
						help="Stop after this many timesteps; for bring-up only.")
	return parser.parse_args()


def pin_cuda_visible_devices(config_path):
	"""Pin `CUDA_VISIBLE_DEVICES` from `data.device` and return the parsed config.

	This is what `run_inference.sh` does in bash for the detection and feature
	stages. It has to happen **before CUDA is initialized**, because
	`CUDA_VISIBLE_DEVICES` is read once, at CUDA-context creation -- which is why
	this function loads the YAML with `yaml` alone and touches nothing that
	reaches `torch`.

	Args:
		config_path (str): Path to the scene YAML.
	Returns:
		dict: The parsed configuration, with
		`reidentifier.pose_estimator.device` rewritten to the local index.
	"""
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)

	os.environ["CUDA_VISIBLE_DEVICES"] = str(config["data"]["device"])

	# `PoseEstimaterAdapter` builds an ABSOLUTE device string, f"cuda:{device}".
	# Under the pin above there is exactly one visible GPU, so the only valid
	# ordinal is 0 -- "cuda:3" would raise. `detector.device` is deliberately NOT
	# rewritten: `select_device` re-writes CUDA_VISIBLE_DEVICES from it, and
	# setting it to "0" would re-point the process at physical GPU 0.
	config["reidentifier"]["pose_estimator"]["device"] = "0"

	return config


if __name__ == "__main__":
	_ARGS   = parse_args()
	_CONFIG = pin_cuda_visible_devices(_ARGS.config)


# region Shared helpers

def get_index_file(video_path):
	"""Extract the trailing numeric index from a video filename, e.g. ".../cam_0001.mp4" -> 1, used to sort videos numerically."""
	# Extract the trailing numeric index from a video filename,
	# e.g. ".../cam_01.mp4" -> 1, used to sort videos numerically.
	filename = os.path.splitext(os.path.basename(video_path))[0]
	return int(filename.split("_")[-1])


def load_class_labels(config):
	"""Load class labels from file.
	Args:
		config (dict): Configuration dictionary containing the path to the class labels file.
	Returns:
		ClassLabels: An instance of the ClassLabels class containing the loaded class labels.
	"""
	class_labels = ClassLabels.create_from_file(config['data']['class_labels']['file'])
	return class_labels

# endregion


# region Stage 1: detection

def detect_frame(detector, config, camera_name, frame_index, image_bgr):
	"""Detect every object in one frame and wrap the result as a frame comp.

	This is the per-frame body of the detection stage: a single `detector.detect`
	call on one image, plus the dict that every later stage carries forward. No
	file is written here; the caller owns the pickle and label output.

	Args:
		detector (BaseDetector): Already-constructed detector. Never reloaded per call.
		config (dict): Configuration dictionary; reads `data.scene_name` and `data.scene_id`.
		camera_name (str): Camera identity stored in the comp. It carries the source
			file extension (e.g. "Camera_0000.mp4"), unlike every directory name in
			the pipeline, which uses the stem.
		frame_index (int): Position of this frame in the sorted image list. This is
			the enumerate position, not the number parsed out of the filename; the
			two coincide only because frames are extracted gap-free from 0.
		image_bgr (numpy.ndarray): (H, W, 3) uint8 BGR frame, as cv2.imread returns.
	Returns:
		dict: The frame comp, i.e. `index`, `height`, `width`, `scene_name`,
		`camera_name`, `scene_id` and `instances`. Each instance holds `class_id`,
		`det_bbox` (numpy.ndarray of shape (4,), float32, as cx, cy, w, h
		**normalized** by image size) and `det_score`; `pose_bbox`,
		`pose_keypoints`, `feat_embeddings` and `track_id` are all None on return
		and are filled in by the later stages. None when the detector returns no
		entry for this image, in which case the caller skips the frame entirely.
	"""
	# Detect batch of instances
	batch_instances = detector.detect(
		indexes=[frame_index], images=[image_bgr]
	)

	# Process the detected instances as needed (e.g., save results, etc.)
	result_dict = None
	for image, index_image, instances in zip([image_bgr], [frame_index], batch_instances):
		# create pickle file for each image
		image_height, image_width = image.shape[:2]
		# Create a result dictionary to store the detection results for the current image
		result_dict = {
			# "image"       : image,  # Original image as a numpy array
			"index"       : index_image,  # Frame index in the video
			"height"      : image_height,
			"width"       : image_width,
			"scene_name"  : config['data']['scene_name'],  # scene name
			"camera_name" : camera_name,  # video name is also used as camera name
			"scene_id"    : config['data']['scene_id'],  # scene id, as the define in the config
			"instances"   : [],
		}

		# Process each detected instance in the batch
		for index_in, instance in enumerate(instances):
			class_id   = instance.class_id
			bbox_xywhn = instance.bbox  # bbox is in xywhn format, normalized by image size
			score      = instance.confidence

			# Append the detection result to the result_dict
			# We can also add keypoints and features to the result_dict if needed,
			# but for now we set them to None
			result_dict["instances"].append({
				"class_id"        : class_id,
				"det_bbox"        : np.array(bbox_xywhn, dtype=np.float32),
				"det_score"       : score,
				"pose_bbox"       : None,
				"pose_keypoints"  : None,
				"feat_embeddings" : None,
				"track_id"        : None,
			})

	return result_dict

# endregion


# region Stage 2: pose estimation

def pose_frame(pose_estimator, class_labels, det_comp, image_bgr):
	"""Estimate poses for one frame's person-like detections.

	Only Person, FourierGR1T2 and AgilityDigit are sent through ViTPose; every
	other class keeps `pose_bbox` and `pose_keypoints` as None. The detection
	boxes arrive as cx, cy, w, h **normalized** and are converted with
	`bbox_cxcywh_norm_to_xywh` to top-left x, y, w, h in **absolute pixels**,
	which is what the pose adapter expects.

	`det_comp` is deep-copied on entry: the split below re-appends the caller's
	own instance dicts and then assigns `pose_bbox` / `pose_keypoints` onto them,
	so without the copy this stage would rewrite the detection stage's output
	in place. That is invisible when each stage reloads a fresh pickle and fatal
	when both run in one process.

	Args:
		pose_estimator (PoseEstimaterAdapter): Already-constructed pose model.
		class_labels (ClassLabels): Class label registry; resolves the three
			class ids routed through pose estimation.
		det_comp (dict): The detection stage's frame comp. Not modified.
		image_bgr (numpy.ndarray): (H, W, 3) uint8 BGR frame the boxes refer to.
	Returns:
		dict: The same frame comp with `pose_bbox` (list of 4 floats, absolute
		pixels) and `pose_keypoints` (list of 17 dicts holding `name`, `x`, `y`,
		`score`, with x and y in absolute pixels of the full frame) filled in for
		the pose classes. Its `instances` are **reordered**: the non-pose classes
		come first, then the pose classes. No stage depends on that order, but a
		positional comparison against the pickle output does.
	"""
	# Work on our own copy; the split below hands the caller's instance dicts to
	# the mutating assignment at the end of this function.
	detection_comp = deepcopy(det_comp)

	img_h = detection_comp["height"]
	img_w = detection_comp["width"]

	# Split detection_comp
	pose_est_need_comp  = deepcopy(detection_comp)  # will be updated with pose estimation results and saved to pickle file for each image/frame
	pose_remain_comp    = deepcopy(detection_comp)  # will be saved to pickle file for each image/frame if no person detected, where the pose estimation results will be empty
	pose_est_need_comp["instances"]  = []
	pose_remain_comp["instances"]    = []

	for instance in detection_comp["instances"]:
		if instance["class_id"] in [
			class_labels.get_id_by_name('Person'),
			class_labels.get_id_by_name('FourierGR1T2'),
			class_labels.get_id_by_name('AgilityDigit'),
		]:
			pose_est_need_comp["instances"].append(instance)  # add person/humannoid instances to pose_est_need_comp
		else:
			pose_remain_comp["instances"].append(instance)  # add non-person/non-humannoid instances to pose_remain_comp
	# first we add the non-person/non-humanboid instances to the final comp,
	pose_final_comp = deepcopy(pose_remain_comp)

	# Get boxes out of the detection_comp and convert to numpy array for further processing
	bbox_xywhs = []
	for instance in pose_est_need_comp["instances"]:
		if instance["class_id"] in [
			class_labels.get_id_by_name('Person'),
			class_labels.get_id_by_name('FourierGR1T2'),
			class_labels.get_id_by_name('AgilityDigit'),
		]:
			bbox_xywhs.append(bbox_cxcywh_norm_to_xywh(np.array(instance["det_bbox"], dtype=np.float32), img_h, img_w))
	bbox_xywhs = np.array(bbox_xywhs)

	# If no person detected, return the original detection_comp without pose estimation results.
	if len(bbox_xywhs) > 0:

		pose_results = pose_estimator.forward(image=image_bgr, bbox_xywh=bbox_xywhs)

		# Keypoints and scores for each detected person
		instances    = []
		for i, (person_pose, instance) in enumerate(zip(pose_results, pose_est_need_comp["instances"])):
			instance["pose_bbox"] = person_pose["pose_bbox"] # (x_top, y_left, ...) in absolute pixel values
			instance["pose_keypoints"] = person_pose["pose_keypoints"] # list of keypoints, each with [x, y, confidence]
			instances.append(instance)

		# Update the pose_final_comp with the pose estimation results
		pose_final_comp["instances"] += instances

	return pose_final_comp

# endregion


# region Stage 3: ReID features

def load_feature_extraction_model(config):
	"""Load the feature extraction model based on the configuration.
	Args:
		config (dict): Configuration dictionary containing the model parameters.
	Returns:
		KPRFeatureExtractor: An instance of the KPRFeatureExtractor class initialized with the specified configuration.
	"""
	cfg = get_default_config()
	cfg.model.load_weights             = config['reidentifier']['feature_extractor']['model']
	cfg.model.backbone_pretrained_path = config['reidentifier']['feature_extractor']['backbone_pretrained_folder_path']
	cfg.use_gpu                        = torch.cuda.is_available() # already done in build_config(...), but can be overwritten here

	kpr_cfg         = build_config(config=cfg)
	kpr_cfg.use_gpu = torch.cuda.is_available() # already done in build_config(...), but can be overwritten here

	extractor       = KPRFeatureExtractor(kpr_cfg)
	return extractor


def feature_frame(extractor, pose_comp, image_bgr):
	"""Extract KPR re-identification features for one frame's posed instances.

	Only instances carrying non-None `pose_keypoints` are featurized; the rest
	keep `feat_embeddings` as None. Each crop comes from
	`bbox_cxcywh_norm_to_xyxy(det_bbox, ...).astype(int)`, and the keypoints are
	shifted into crop-local coordinates and clamped for the KPR sample only --
	the `pose_keypoints` stored on the instance stay in full-frame pixels.

	`pose_comp` is deep-copied on entry. This stage both rebinds `instances` on
	the comp and assigns `feat_embeddings` onto the instance dicts it was handed,
	so without the copy it would rewrite the pose stage's output in place --
	harmless when every stage reloads a fresh pickle, not when they share a process.

	Args:
		extractor (KPRFeatureExtractor): Already-constructed feature extractor.
		pose_comp (dict): The pose stage's frame comp. Not modified.
		image_bgr (numpy.ndarray): (H, W, 3) uint8 BGR frame the boxes refer to.
	Returns:
		dict: The same frame comp with `feat_embeddings` filled in as
		`{"embedding": numpy.ndarray (6, 512) float32, "visibility_score":
		numpy.ndarray (6,) float32}` for every featurized instance. The tracking
		stage concatenates those two into (6, 513). `instances` is **reordered**:
		the unfeaturized instances come first, then the featurized ones. No stage
		depends on that order, but a positional comparison against the pickle
		output does.
	"""
	# Work on our own copy; the loop below rebinds `instances` and writes
	# `feat_embeddings` onto the caller's instance dicts.
	comp = deepcopy(pose_comp)

	img = image_bgr
	instances_with_feats   = []
	instance_without_feats = []
	samples = []

	# NOTE: Extract features for each pose in the current image/frame
	for instance in comp['instances']:
		if 'pose_keypoints' in instance and instance['pose_keypoints'] is not None:
			# Extract the bounding box and keypoints for the current pose
			img_width, img_height = comp['width'], comp['height']
			det_bbox_xyxy  = bbox_cxcywh_norm_to_xyxy(
				np.array(instance['det_bbox'], dtype=np.float32),
				float(img_height),
				float(img_width)
			).astype(int)  # (xc, yc, w, h) normalized -> (x1, y1, x2, y2)
			pose_keypoints = instance['pose_keypoints']  # list of keypoints, each with [x, y, confidence]

			# crop using array slicing: [ymin:ymax, xmin:xmax]
			img_cropped = img[det_bbox_xyxy[1]:det_bbox_xyxy[3], det_bbox_xyxy[0]:det_bbox_xyxy[2]]

			# Initialize lists to hold keypoints
			keypoints_xyc = []
			negative_kps  = []

			# Process each keypoint
			for keypoint in pose_keypoints:
				h, w = img_cropped.shape[:2]
				x = max(0, min(int(keypoint['x'] - det_bbox_xyxy[0]), w - 1))  # Adjust x coordinate to be relative to the cropped image
				y = max(0, min(int(keypoint['y'] - det_bbox_xyxy[1]), h - 1))  # Adjust y coordinate to be relative to the cropped image
				confidence = keypoint['score']
				keypoints_xyc.append([x, y, confidence])

			# Create the sample dictionary
			sample = {
				"image"        : img_cropped, # (H, W, 3) uint8 BGR
				"keypoints_xyc": keypoints_xyc, # (17, 3)
				"negative_kps" : negative_kps, # (M, 17, 3) or (0,)
			}
			samples.append(sample)

			instances_with_feats.append(instance)  # Add the instance with valid pose/keypoints to the list for feature extraction
		else:
			instance_without_feats.append(instance)  # If no valid pose/keypoints, keep the original instance without features

	# Add instances without features back to the comp['instances']
	comp['instances'] = instance_without_feats

	# Extract features for instances with valid pose/keypoints
	if len(samples) > 0:
		_, embeddings, visibility_scores, _ = extractor(samples)
		np_embeddings  = embeddings.cpu().detach().numpy()
		np_vis_scrores = visibility_scores.cpu().detach().numpy()

		for instance, np_embedding, np_vis_score in zip(instances_with_feats, np_embeddings, np_vis_scrores):
			instance['feat_embeddings'] = {
				'embedding'       : np_embedding, # (6, 512) ndarray
				'visibility_score': np_vis_score, # (6) ndarray
			}
			comp['instances'].append(instance)  # Add the instance with features back to the comp['instances'] list

	return comp

# endregion


# region Stage 4: single-camera tracking

def to_namespace(data):
	"""Recursively convert a nested dict/list structure to a SimpleNamespace.
	Args:
		data (dict or list): The input data structure to convert.
	Returns:
		SimpleNamespace or list: The converted data structure, where dicts are
		converted to SimpleNamespace and lists are converted recursively.
		Non-dict/list values are returned as-is. This allows for easy attribute
		access to configuration parameters.
	Example:
		>>> config_dict = {'tracker': {'type': 'BoTSORT', 'max_age': 30}, 'data': {'root': '/data'}}
		>>> config_ns = to_namespace(config_dict)
		>>> print(config_ns.tracker.type)  # Output: 'BoTSORT'
		>>> print(config_ns.data.root)    # Output: '/data'
	"""
	if isinstance(data, dict):
		return SimpleNamespace(**{k: to_namespace(v) for k, v in data.items()})
	elif isinstance(data, list):
		return [to_namespace(item) for item in data]
	return data


def make_trackers(config, class_labels):
	"""Build one BoT-SORT tracker per class label, sharing the `tracker` config block.

	`BoTSORT.__init__` calls `BaseTrack.clear_count()` unconditionally, so every
	call to this function zeroes the process-global track-id counter. That is why
	a caller holding several cameras' tracker sets at once must build **all** of
	them before the first frame is tracked: building one lazily, mid-run, would
	zero the counter underneath the cameras already running.

	Args:
		config (dict): Configuration dictionary; the whole `tracker` block is
			converted with `to_namespace` and handed to each tracker.
		class_labels (ClassLabels): Class label registry; one tracker per entry.
	Returns:
		dict: `{class_id: BoTSORT}`. Stateful -- the same dict must be reused for
		the same camera across frames, and never shared between cameras.
	"""
	# create trackers for all class labels
	trackers = {}
	for class_label in class_labels.class_labels:
		trackers[class_label['id']] = BoTSORT(args=to_namespace(config['tracker']))
	return trackers


def track_non_person_instances(config, feat_comp, comp, class_labels, trackers, image_height, image_width):
	"""Track non-person instances for one frame, without re-identification.

	Gathers all non-person detections from feat_comp into per-class tracker
	inputs, updates each class's BoT-SORT tracker via update_no_reid, and
	appends the resulting tracklets to comp["instances"] with their track id
	and normalized track bbox (detection/pose/embedding fields set to None).
	Person instances are skipped; they are handled by a separate
	re-identification-based tracking step.

	Args:
		config (dict): Configuration dictionary containing parameters for tracking.
		feat_comp (dict): Per-frame compression containing detected "instances",
			each with "class_id", "det_bbox" (cx, cy, w, h, normalized) and "det_score".
		comp (dict): Output compression for the frame; tracked instances are
			appended to its "instances" list.
		class_labels (ClassLabels): Class label registry, used to identify the
			'Person' class to skip.
		trackers (dict): Mapping from class id to its BoTSORT tracker instance.
		image_height (int): Frame height in pixels, used for bbox (de)normalization.
		image_width (int): Frame width in pixels, used for bbox (de)normalization.
	Returns:
		dict: The comp dictionary with tracked non-person instances appended.
	"""
	# create input
	detection_inputs = {}
	for class_label in class_labels.class_labels:
		detection_inputs[class_label['id']] = []

	# NOTE: load detections for non-person classes into tracker inputs
	for instance in feat_comp["instances"]:
		if instance["class_id"] in [
			class_labels.get_id_by_name('Person'),
			class_labels.get_id_by_name('FourierGR1T2'),
			]:
			continue 
		
		class_id   = instance["class_id"]
		bbox_xyxy  = bbox_cxcywh_norm_to_xyxy(instance["det_bbox"], image_height, image_width)
		score      = instance["det_score"]
		detection_inputs[class_id].append([
			None, 					 
			bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3], 
			score,
			class_id
		])

	# NOTE: Tracking without re-identification for each class separately
	for class_id, detection_result_input in detection_inputs.items():
		if class_id in [
			class_labels.get_id_by_name('Person'),
			class_labels.get_id_by_name('FourierGR1T2'),
			]:
			continue 

		# tracking process
		tracker_result = trackers[class_id].update_no_reid(np.array(detection_result_input))

		# Add the comp
		for tracklet in tracker_result:
			bbox_tlbr   = tracklet.tlbr
			track_id    = tracklet.track_id
			class_id_re = tracklet.obj_cls if tracklet.obj_cls is not None else class_id  # Use the original class_id if the tracker doesn't provide one
			comp["instances"].append({
					"class_id"        : class_id_re,
					"det_bbox"        : None,
					"det_score"       : None,
					"pose_bbox"       : None,
					"pose_keypoints"  : None,
					"feat_embeddings" : {
						'embedding'       : None,
						'visibility_score': None,
					},
					"track_score"     : tracklet.score if hasattr(tracklet, 'score') else None,  # Optional track confidence score
					"track_id"        : track_id,
					"track_bbox"      : bbox_xyxy_to_cxcywh_norm(bbox_tlbr, image_height, image_width),
				})

	return comp


def track_person_instances(config, feat_comp, comp, class_labels, trackers, image_height, image_width):
	"""Track person instances for one frame, using pose and re-identification.

	Gathers all person detections from feat_comp into per-class inputs, bundling
	each detection with its pose keypoints and re-identification features
	(embedding concatenated with per-part visibility scores). Updates each
	person class's BoT-SORT tracker via update (which fuses motion, pose, and
	appearance), and appends the resulting tracklets to comp["instances"] with
	their track id, pose keypoints, feature embeddings, and normalized track
	bbox. Non-person instances are skipped; they are handled by
	track_non_person_instances.

	Args:
		config (dict): Configuration dictionary containing parameters for tracking.
		feat_comp (dict): Per-frame compression containing detected "instances",
			each with "class_id", "det_bbox" (cx, cy, w, h, normalized),
			"det_score", "pose_keypoints" (list of {x, y, score}), and
			"feat_embeddings" ({embedding, visibility_score}).
		comp (dict): Output compression for the frame; tracked instances are
			appended to its "instances" list.
		class_labels (ClassLabels): Class label registry, used to identify the
			'Person' class to track.
		trackers (dict): Mapping from class id to its BoTSORT tracker instance.
		image_height (int): Frame height in pixels, used for bbox (de)normalization.
		image_width (int): Frame width in pixels, used for bbox (de)normalization.
	Returns:
		dict: The comp dictionary with tracked person instances appended.
	"""
	# create input
	detection_results       = {}
	pose_keypoints_results  = {}
	feat_embeddings_results = {}
	for class_label in class_labels.class_labels:
		detection_results[class_label['id']]       = []
		pose_keypoints_results[class_label['id'] ] = []
		feat_embeddings_results[class_label['id']] = []

	# NOTE: Load input for tracking
	for instance in feat_comp["instances"]:
		if not instance["class_id"] in [
			class_labels.get_id_by_name('Person'),
			class_labels.get_id_by_name('FourierGR1T2'),
			]:
			continue 
			
		# get basic information
		class_id    = instance["class_id"]

		# load detection results
		bbox_xyxy  = bbox_cxcywh_norm_to_xyxy(instance["det_bbox"], image_height, image_width)
		score      = instance["det_score"]
		detection_results[class_id].append([
			None, 					 
			bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3], 
			score,
			class_id
		])

		# load keypoints of pose estimation results
		pose_keypoints = instance['pose_keypoints']
		for i in range(len(pose_keypoints)):
			x = pose_keypoints[i]['x']
			y = pose_keypoints[i]['y']
			confidence = pose_keypoints[i]['score']
			pose_keypoints[i] = [x, y, confidence]
		pose_keypoints_results[class_id].append(pose_keypoints)

		# load features of re-identification results
		embedding        = np.array(instance['feat_embeddings']['embedding'], dtype=np.float32)  # (512,)
		visibility_score = np.array(instance['feat_embeddings']['visibility_score'], dtype=np.float32) # (6,)
		visibility_score_expand_arr = visibility_score[:, np.newaxis]
		feat_embedding   = np.concatenate((embedding, visibility_score_expand_arr), axis=1)
		feat_embeddings_results[class_id].append(feat_embedding) # (6, 512+1)

	# NOTE: Tracking with pose and re-identification for each class separately
	for class_id, detection_result_input in detection_results.items():
		if not class_id in [
			class_labels.get_id_by_name('Person'),
			class_labels.get_id_by_name('FourierGR1T2'),
			]:
			continue

		# tracking process
		tracker_result = trackers[class_id].update(
			np.array(detection_results[class_id]),
			np.array(feat_embeddings_results[class_id], dtype=np.float32),
			np.array(pose_keypoints_results[class_id], dtype=np.float32),
		)

		# Add the comp
		for tracklet in tracker_result:
			bbox_tlbr        = tracklet.tlbr
			track_id         = tracklet.track_id
			class_id_re      = tracklet.obj_cls if tracklet.obj_cls is not None else class_id  # Use the original class_id if the tracker doesn't provide one
			embedding        = tracklet.curr_feat
			visibility_score = tracklet.curr_viss
			
			if tracklet.pose is not None:
				pose_keypoints = []
				for keypoint in tracklet.pose:
					x, y, confidence = keypoint
					pose_keypoints.append({'x': x, 'y': y, 'score': confidence})
			else:
				pose_keypoints   = None

			comp["instances"].append({
					"class_id"        : class_id_re,
					"det_bbox"        : None,
					"det_score"       : None,
					"pose_bbox"       : None,
					"pose_keypoints"  : pose_keypoints,
					"feat_embeddings" : {
						'embedding'       : embedding,
						'visibility_score': visibility_score,
					},
					"track_score"     : tracklet.score if hasattr(tracklet, 'score') else None,  # Optional track confidence score
					"track_id"        : track_id,
					"track_bbox"      : bbox_xyxy_to_cxcywh_norm(bbox_tlbr, image_height, image_width),
				})

	return comp


def track_frame(config, class_labels, trackers, zones_none, feat_comp):
	"""Filter one frame's detections and run single-camera tracking on them.

	Drops detections below `tracker.min_confidence_det` and boxes lying inside a
	`regions.zone_camera` polygon, then routes what is left through
	`track_person_instances` (Person and FourierGR1T2, with pose and ReID) and
	`track_non_person_instances` (everything else, without ReID). AgilityDigit
	carries a pose from the pose stage but is tracked without ReID.

	`feat_comp` is deep-copied on entry. `track_person_instances` rewrites
	`instance['pose_keypoints']` **in place**, from dicts to `[x, y, conf]` lists,
	on the very instance dicts it is handed -- and the filter below re-appends the
	caller's originals rather than copies. Without the copy this stage would
	corrupt the feature stage's output the moment it ran in the same process.

	Args:
		config (dict): Configuration dictionary; reads `tracker.min_confidence_det`
			and passes the whole config down to the two tracking helpers.
		class_labels (ClassLabels): Class label registry.
		trackers (dict): `{class_id: BoTSORT}` from `make_trackers`. **Stateful**:
			it must be the same dict for the same camera on every frame, and each
			camera needs its own.
		zones_none (list): Polygons from `load_zone_camera`; a box fully inside one
			is dropped.
		feat_comp (dict): The feature stage's frame comp. Not modified.
	Returns:
		dict: The frame comp whose instance schema **changes shape** -- `track_id`,
		`track_bbox` (cx, cy, w, h **normalized**) and `track_score` appear, while
		`det_bbox`, `det_score` and `pose_bbox` are dropped to None even for
		persons. `pose_keypoints` comes back as dicts of `x`, `y`, `score`: the
		`name` key present upstream does **not** survive the round trip through
		the tracker.
	"""
	# Work on our own copy; the filter below re-appends the caller's instance
	# dicts and `track_person_instances` rewrites their keypoints in place.
	feat_comp = deepcopy(feat_comp)

	# NOTE: filter bbox
	feat_comp_filtered = deepcopy(feat_comp)
	feat_comp_filtered["instances"] = []
	for instance in feat_comp["instances"]:

		# filter out low-confidence detections
		if float(instance["det_score"]) < config['tracker']['min_confidence_det']:
			continue

		bbox_xyxy    = bbox_cxcywh_norm_to_xyxy(instance["det_bbox"], feat_comp["height"], feat_comp["width"])
		# box(xmin, ymin, xmax, ymax)
		bbox_polygon = box(
			min(max(bbox_xyxy[0], 1), feat_comp["width"] - 2),
			min(max(bbox_xyxy[1], 1), feat_comp["height"] - 2),
			min(max(bbox_xyxy[2], 1), feat_comp["width"] - 2),
			min(max(bbox_xyxy[3], 1), feat_comp["height"] - 2)
		)

		# filter out bbox in none zone
		if is_bbox_in_zone(zones_none, bbox_polygon):
			continue

		# if instance okay, we append it
		feat_comp_filtered["instances"].append(instance)

	# create new comp
	comp = deepcopy(feat_comp_filtered)
	comp["instances"] = []

	image_height, image_width = comp['height'], comp['width']

	# NOTE: Person tracking with re-identification
	# Load information for tracking with re-identification
	comp = track_person_instances(
		config,
		feat_comp_filtered,
		comp,
		class_labels,
		trackers,
		image_height,
		image_width
	)

	# NOTE: Non-Person tracking
	# Load information for tracking without re-identification
	comp = track_non_person_instances(
		config,
		feat_comp_filtered,
		comp,
		class_labels,
		trackers,
		image_height,
		image_width
	)

	return comp

# endregion


# region Stage 5: multi-camera fusion

# Code-level defaults so per-scene configs need no `mtmc:` block unless overriding.
DEFAULT_MTMC = {
	"ankle_conf_thresh"       : 0.5,
	"pose_conf_thresh"        : 0.3,
	"min_conf_keypoint_for_estimate": 8,
	"bev_group_gate"          : 1.0,
	"reid_group_thresh"       : 0.35,
	"bev_dist_thresh_by_class": [1.0, 5.0, 2.0, 2.0, 1.5, 1.0, 2.0],
	"make_world_observation_mode": "max_confidence",
	"w_bev"                   : 0.5,
	"reid_track_thresh"       : 0.4,
	"max_speed_m_s"           : 3.0,
	"new_track_min_sep"       : 1.5,
	"max_age"                 : 60,
	"alpha_pos"               : 0.6,
	"gallery_size"            : 60,
}


def load_calib_infos(config):
	"""Return ``{camera_id: {'intrinsicMatrix', 'extrinsicMatrix', ...}}`` for the scene.

	Projection uses ``intrinsicMatrix`` (3x3 K) and ``extrinsicMatrix`` (3x4
	[R|t]); the ground homography is derived from ``P = K @ [R|t]`` downstream.
	Other keys (e.g. ``translationToGlobalCoordinates``, absent in
	Warehouse_027) are intentionally not required.
	"""
	calib_path = os.path.join(
		# config["data"]["root"], 
		# config["data"]["subset"],
		# config["data"]["scene_name"], 
		config["data"]["calibration"]["file"],
	)
	with open(calib_path, "r") as f:
		scene_calib = json.load(f)

	calib_dict = {}
	for sensor in scene_calib["sensors"]:
		if sensor.get("type") == "camera":
			calib_dict[sensor["id"]] = {
				"intrinsicMatrix": sensor["intrinsicMatrix"],
				"extrinsicMatrix": sensor["extrinsicMatrix"],
				"scaleFactor": sensor.get("scaleFactor"),
				"translationToGlobalCoordinates": sensor.get("translationToGlobalCoordinates"),
			}
	return calib_dict


def observations_from_comp(comp, camera_index, camera_name, homography_matrix, camera_matrix,
						   class_labels, mtmc_cfg, zones_bev = None, height_sampler = None):
	"""Project one camera's single-camera-tracking comp into ``CamObservation``s.

	Bbox is read from ``track_bbox`` (normalized cx, cy, w, h), never from
	``det_bbox`` -- the tracking stage sets the latter to None. ``class_id`` of
	``None`` means Person: only Person detections enter the ReID path, so the
	single-camera writer may leave the field unset. An instance whose ground
	anchor cannot be found, whose projection is invalid, or which falls outside
	the camera's BEV zone is **dropped silently**; the returned list is therefore
	shorter than ``comp["instances"]``.
	"""
	person_id                      = class_labels.get_id_by_name("Person")
	ankle_conf_thresh              = mtmc_cfg["ankle_conf_thresh"]
	pose_conf_thresh               = mtmc_cfg["pose_conf_thresh"]
	min_conf_keypoint_for_estimate = mtmc_cfg["min_conf_keypoint_for_estimate"]
	projection_type                = mtmc_cfg.get("projection_type",{})
	bbox_ratio_filter              = mtmc_cfg.get("bbox_ratio_filter", {})
	skeleton_filter				   = mtmc_cfg.get("skeleton_filter", [])

	observations = []
	img_h, img_w = comp["height"], comp["width"]
	for instance in comp["instances"]:
		class_id = instance.get("class_id")
		# Persons are written with class_id=None upstream (only Person
		# detections enter the ReID path); map them to the Person id. int()
		# guards against numpy-typed ids from the tracker.
		class_id = person_id if class_id is None else int(class_id)

		if not class_labels.get_class_label(key="id", value=class_id):
			continue

		# NOTE: Person height sample from THIS frame only. Computed from the
		# ankle-midpoint ground point, deliberately independent of the
		# tracking anchor below (whose top_bottom branch derives from the
		# class-constant height). Keys off the RESOLVED class id.
		h_sample = None
		if height_sampler is not None and class_id == person_id:
			h_sample = height_sampler.compute_sample(
				instance, camera_name, camera_matrix, homography_matrix,
				img_w, img_h,
				zone_polygons=(zones_bev or {}).get(camera_name),
				zone_dist_thresh=mtmc_cfg.get("distance_to_polygon_thresh"))

		# NOTE: Heuristically determine the ground anchor pixel on image for this instance.
		# find the pixel location on the image of the ground contact point.
		anchor = ground_anchor_px(instance, img_w, img_h,
									ankle_conf_thresh, pose_conf_thresh, min_conf_keypoint_for_estimate,
									camera_matrix, projection_type, bbox_ratio_filter, skeleton_filter)
		if anchor is None:
			continue

		# NOTE: Project the anchor pixel to world coordinates using the camera's homography; skip if invalid.
		# find BEV location of bbox
		world_xy, valid = image_to_world(homography_matrix, anchor)
		if not valid:
			continue

		# NOTE: filter out any bounding box that is out of Camera BEV zone.
		if zones_bev is not None and camera_name in zones_bev and len(zones_bev[camera_name]) > 0:
			if not is_point_in_zone(zones_bev[camera_name], world_xy, mtmc_cfg["distance_to_polygon_thresh"]):
				continue

		# create a CamObservation with the projected world coordinates, original bbox, and any available embedding/visibility; skip if no valid anchor or projection.
		feat = instance.get("feat_embeddings") or {}
		observations.append(CamObservation(
			cam_idx        = camera_index,
			class_id       = class_id,
			local_track_id = instance.get("track_id"),  # avoid ID clashes across classes
			world_xy       = world_xy,
			confidence     = instance.get("track_score"),  # optional confidence score for the observation
			bbox_xywhn     = instance.get("track_bbox"),
			embedding      = feat.get("embedding"),
			visibility     = feat.get("visibility_score"),
			height_sample  = h_sample,
			ground_pixel   = np.asarray(anchor, dtype=np.float64).copy(),
			pose_keypoints = instance.get("pose_keypoints"),
			image_size     = (int(img_h), int(img_w)),
		))

	return observations


def write_submission(result, scene_id, result_path, class_labels,
					 height_estimation_enabled=False):
	"""Write all global tracks as Track 1 rows, sorted by (frame, class, object).
	Format:
	``<scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>``
	where the four ids are integers and the seven floats are 2-decimal with a single space separator (no trailing newline, no comment markers).

	With ``height_estimation_enabled`` the per-frame estimated person height
	(``track.height_by_frame``, current frame only) replaces the class-constant
	height, with ``z = h/2`` following it; frames without an accepted sample
	keep the ``lift_to_3d`` default. Only person tracks carry samples, so all
	other classes are unaffected.
	"""
	rows = []
	index_track_id    = 0  # global track id across all classes
	index_max_current = 0  
	for class_id, tracks in result.items():

		if not class_labels.get_class_label(key="id", value=class_id):
			logger.warning(f"No definition for class_id {class_id}; skipping its tracks.")
			continue

		for track in tracks:
			for frame_id, (x, y) in track.frames.items():
				# create the 3D box parameters for this frame's observation
				location_x, location_y, location_z, width, length, height, _ = lift_to_3d(class_id, (x, y))
				# per-frame yaw from the track's fitted heading (lift_to_3d yaw is a 0.0 placeholder)
				yaw = track.rotations.get(frame_id, {}).get('yaw', 0.0)

				# Per-frame estimated height (persons only carry entries); a frame
				# with no accepted sample keeps the lift_to_3d class default.
				if height_estimation_enabled:
					height     = track.height_by_frame.get(frame_id, height)
					location_z = height / 2.0

				# global track id 
				current_track_id  = index_track_id + track.object_id
				index_max_current = max(index_max_current, current_track_id)

				# DEBUG: for check number of object create, every class has track_id start from 1
				# current_track_id  = track.object_id

				row = {
					"scene_id"         : scene_id,
					"class_id"         : class_id,
					"object_id"        : current_track_id,
					"frame_id"         : frame_id,
					"location_x"       : location_x,
					"location_y"       : location_y,
					"location_z"       : location_z,
					"width"            : width,
					"length"           : length,
					"height"           : height,
					"yaw"              : yaw
				}

				# append the formatted row
				rows.append(row)
				# rows.append((row['frame_id'], row['class_id'], row['object_id'], row['frame_id'], row))
				# rows.append((frame_id, class_id, current_track_id, frame_id,
							#  format_row(scene_id, class_id, current_track_id, frame_id,
							# 			location_x, location_y, location_z, width, length, height, yaw)))

		index_track_id = index_max_current
	
	# sort by <scene_id> <class_id> <object_id> <frame_id> 
	# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
	rows.sort(key=lambda r: (r['scene_id'], r['class_id'], r['object_id'], r['frame_id']))
	
	from tools.post_process import _row_sort_key, _remap_continuous, _OBJECT_ID_KEY
	# rows = _do_(rows, result_path, scene_id, skip_person_height_hardcode=height_estimation_enabled)
	rows.sort(key=_row_sort_key)
	rows = _remap_continuous(rows, _OBJECT_ID_KEY)

	# write results to the specified result path, creating directories as needed
	os.makedirs(os.path.dirname(result_path), exist_ok=True)
	with open(result_path, "w") as f:
		for row in rows:
			line = format_row(
				row["scene_id"], row["class_id"], row["object_id"], row["frame_id"],
				row["location_x"], row["location_y"], row["location_z"],
				row["width"], row["length"], row["height"], row["yaw"]
			)
			f.write(line + "\n")

	return len(rows)

# endregion

# region Out-of-memory store

# The four intermediate trees, in pipeline order. These are `data_writer` keys,
# and they are the stage boundaries: each one is written by the stage above it
# and read back by the stage below it.
COMP_TREES = ("dets_comp", "poses_comp", "feats_comp", "mots_single_comp")

# Pickle protocol per tree, matching the stage that produces it in the
# five-process path. The protocols are NOT uniform there: detection passes
# `protocol=pickle.HIGHEST_PROTOCOL` explicitly, while the pose, feature and
# tracking stages call bare `pickle.dump(comp, f)` and so get
# `pickle.DEFAULT_PROTOCOL`. On this interpreter that is 5 and 4 respectively,
# and the baseline trees on disk carry exactly those two headers. Writing all
# four at one protocol round-trips correctly but changes the bytes, which would
# fail a file-level diff against the baseline for protocol reasons alone.
COMP_PROTOCOL = {
	"dets_comp"        : pickle.HIGHEST_PROTOCOL,  # run_object_detection.py:282
	"poses_comp"       : pickle.DEFAULT_PROTOCOL,  # run_pose_estimation.py:287
	"feats_comp"       : pickle.DEFAULT_PROTOCOL,  # run_feature_extractor.py:245
	"mots_single_comp" : pickle.DEFAULT_PROTOCOL,  # run_single_camera_tracking.py:589
}


def comp_dir(config, tree_key, camera_name, root=None):
	"""Folder holding one stage's per-frame comps for one camera.

	Mirrors the path the five-process detection stage builds, key for key, so the
	merged run's trees land exactly where the split run's do and can be diffed
	against them. Note the directory component is the camera **stem**, not the
	`camera_name` recorded inside the comp, which carries the `.mp4` extension.
	`os.makedirs` on the result belongs at setup, once per (tree, camera) -- never
	per frame.

	Args:
		config (dict): Configuration dictionary; reads `data_writer.root`,
			`data_writer[tree_key]` and `data.scene_name`.
		tree_key (str): One of `COMP_TREES`.
		camera_name (str): Camera stem, e.g. "Camera_0000".
		root (str): Store root overriding `data_writer.root`. Used by the
			fusion-less input modes so a debug run cannot overwrite a scene's
			pickles; None means the configured root.
	Returns:
		str: e.g. "<root>/<dets_comp>/Warehouse_025/Camera_0000".
	"""
	return os.path.join(
		root if root is not None else config["data_writer"]["root"],
		config["data_writer"][tree_key],
		config["data"]["scene_name"],
		camera_name,
	)


def write_comp(folder, frame_index, comp, protocol=pickle.HIGHEST_PROTOCOL):
	"""Pickle one frame comp into the store as `%08d.pkl`.

	Callers pass `COMP_PROTOCOL[tree_key]`, not the default: the five-process
	path does not write all four trees at the same protocol, and a mismatch
	round-trips correctly while changing the bytes on disk, which breaks a
	file-level diff against a baseline tree.

	Args:
		folder (str): Directory from `comp_dir`; must already exist.
		frame_index (int): The timestep, i.e. the comp's `index`.
		comp (dict): The frame comp to store.
		protocol (int): Pickle protocol; see `COMP_PROTOCOL`. The default matches
			the detection stage, which is the one tree written at
			`pickle.HIGHEST_PROTOCOL`.
	"""
	with open(os.path.join(folder, f"{frame_index:08d}.pkl"), "wb") as f:
		pickle.dump(comp, f, protocol=protocol)


def read_comp(folder, frame_index):
	"""Load one frame comp back out of the store.

	This returns a **fresh object graph** with no aliasing back to whatever was
	written, and that is the point rather than an optimization to skip when the
	object is still in a local. Several stages mutate their input in place; the
	read-back is what makes that unreachable, for the same reason the five
	separate processes are safe from it.

	Args:
		folder (str): Directory from `comp_dir`.
		frame_index (int): The timestep.
	Returns:
		dict: The stored frame comp.
	"""
	with open(os.path.join(folder, f"{frame_index:08d}.pkl"), "rb") as f:
		return pickle.load(f)


def make_store_dirs(config, list_of_cameras, root=None):
	"""Create every (tree, camera) store folder once, before the first timestep.

	Four trees x N cameras `makedirs` calls in total. Doing this per frame instead
	would add four redundant syscalls per camera per timestep -- 360,000 of them
	over a 10-camera, 9000-frame scene -- which is the easiest way to make the
	store look slow.

	Args:
		config (dict): Configuration dictionary.
		list_of_cameras (list): Camera stems in scene order.
		root (str): Store root overriding `data_writer.root`, or None.
	Returns:
		dict: `{tree_key: {camera_name: folder}}` for every key in `COMP_TREES`.
	"""
	dirs = {}
	for tree_key in COMP_TREES:
		dirs[tree_key] = {}
		for camera_name in list_of_cameras:
			folder = comp_dir(config, tree_key, camera_name, root)
			os.makedirs(folder, exist_ok=True)
			dirs[tree_key][camera_name] = folder
	logger.info(f"Comp store: {len(COMP_TREES)} trees x {len(list_of_cameras)} cameras under "
				f"{root if root is not None else config['data_writer']['root']}")
	return dirs


def verify_store_is_channel():
	"""Startup self-check that `write_comp`/`read_comp` really go through disk.

	A pipeline that writes each comp and then forwards the in-memory local instead
	of the stored one produces identical output, so every comparison against a
	baseline passes while the store is decorative. This is the check that tells
	the two apart: poison the written object between the write and the read, and
	confirm the poison does not survive the round-trip.

	It runs on a throwaway directory rather than a real tree, so it cannot leave a
	sentinel `.pkl` behind for a whole-tree diff to trip over.

	Raises:
		AssertionError: If the read-back reflects a mutation made after the write,
			or aliases the written object.
	"""
	with tempfile.TemporaryDirectory(prefix="run_pipeline_selfcheck_") as folder:
		comp = {"index": 0, "instances": [{"class_id": 0, "track_id": None}]}
		write_comp(folder, 0, comp)
		comp["instances"] = []                    # poison the local, post-write
		restored = read_comp(folder, 0)
		if not restored["instances"]:
			raise AssertionError(
				"store round-trip returned the in-memory object, not the stored copy")
		if restored["instances"] is comp["instances"]:
			raise AssertionError("read_comp aliased the written object graph")

# endregion


# region Scene layout

def camera_file_name(config, camera_name):
	"""Camera identity as the detection stage records it, i.e. **with** extension.

	`frame_comp["camera_name"]` is the source video's basename, so it carries the
	`.mp4` suffix, while every directory in the pipeline uses the stem. Nothing
	downstream reads the field, but it is preserved verbatim so the comp stays
	comparable with the pickle path.

	Args:
		config (dict): Configuration dictionary; the suffix comes from `data.file`.
		camera_name (str): Camera stem, e.g. "Camera_0000".
	Returns:
		str: e.g. "Camera_0000.mp4".
	"""
	extension = os.path.splitext(config["data"]["file"])[1] or ".mp4"
	return f"{camera_name}{extension}"


def scene_image_root(config):
	"""Root folder holding one sub-folder of extracted frames per camera."""
	return os.path.join(
		config["data_writer"]["root"],
		config["data_writer"]["images"],
		config["data"]["scene_name"],
	)


def list_scene_cameras(config, only=None):
	"""Camera stems for the scene, sorted the way every stage sorts them.

	The order defines `cam_idx`, which BEV fusion uses to refuse to merge two
	observations from the same camera, so it must stay
	`sorted(..., key=get_index_file)`.

	Args:
		config (dict): Configuration dictionary.
		only (list): Optional subset of camera stems to keep.
	Returns:
		list: Camera stems, e.g. ["Camera_0000", ..., "Camera_0009"].
	"""
	image_root = scene_image_root(config)
	cameras    = [d for d in os.listdir(image_root)
				  if os.path.isdir(os.path.join(image_root, d))]
	if only is not None:
		wanted   = set(only)
		cameras  = [c for c in cameras if c in wanted]
		missing  = wanted - set(cameras)
		if missing:
			logger.warning(f"--cameras named cameras with no image folder: {sorted(missing)}")
	return sorted(cameras, key=get_index_file)


def list_camera_frames(camera_dir):
	"""Sorted frame paths for one camera folder, in `get_index_file` order."""
	return sorted(glob.glob(os.path.join(camera_dir, "*.jpg")), key=get_index_file)

# endregion


# region Input modes

def iter_timesteps(config, args, list_of_cameras):
	"""Yield `(frame_index, {camera_name: image_bgr})` one timestep at a time.

	A camera **absent from the yielded dict has no image at `t`** and is simply
	not processed for that timestep, mirroring the `os.path.exists` skip the
	pickle-based fusion stage does. A timestep where every camera is absent is
	still yielded with an empty dict, so fusion keeps ageing its tracks exactly
	as `range(num_frames)` made it do before.

	Frames are addressed by **position** in each camera's sorted file list, not
	by the number in the filename -- that is the same `enumerate` position the
	detection stage used as its frame index and pickle name.

	Args:
		config (dict): Configuration dictionary; reads `data.num_frames`.
		args (argparse.Namespace): Parsed CLI arguments.
		list_of_cameras (list): Camera stems in scene order.
	Yields:
		tuple: `(frame_index, {camera_name: numpy.ndarray})`, images as (H, W, 3)
		uint8 BGR.
	"""
	image_root       = scene_image_root(config)
	frames_by_camera = {cam: list_camera_frames(os.path.join(image_root, cam))
						for cam in list_of_cameras}

	num_frames = config["data"]["num_frames"]
	if args.max_timesteps is not None:
		num_frames = min(num_frames, args.max_timesteps)

	for frame_index in tqdm(range(num_frames), desc=f"Pipeline {config['data']['scene_name']}"):
		frames_t = {}
		for camera_name in list_of_cameras:
			paths = frames_by_camera[camera_name]
			if frame_index >= len(paths):
				continue
			image = cv2.imread(paths[frame_index])
			if image is None:
				logger.warning(f"Unreadable frame skipped: {paths[frame_index]}")
				continue
			frames_t[camera_name] = image
		yield frame_index, frames_t


def camera_name_from_path(path, fallback=None):
	"""Camera stem implied by a file or folder path.

	`.../Camera_0001/00000042.jpg` resolves to `Camera_0001` from the parent
	folder; a folder path resolves to its own name. `fallback` (i.e.
	`--camera-name`) always wins when supplied.
	"""
	if fallback is not None:
		return fallback
	if os.path.isdir(path):
		return os.path.basename(os.path.normpath(path))
	return os.path.basename(os.path.dirname(os.path.abspath(path)))


def iter_single_camera(config, args):
	"""Yield timesteps for `--images` / `--image`: one view, one or many `t`.

	Yields:
		tuple: `(frame_index, {camera_name: image_bgr})` with exactly one camera.
	"""
	if args.image is not None:
		paths       = [args.image]
		camera_name = camera_name_from_path(args.image, args.camera_name)
	else:
		paths       = list_camera_frames(args.images)
		camera_name = camera_name_from_path(args.images, args.camera_name)
		if not paths:
			logger.error(f"No .jpg frames under {args.images}")
			return

	for frame_index, path in enumerate(paths):
		image = cv2.imread(path)
		if image is None:
			logger.warning(f"Unreadable frame skipped: {path}")
			continue
		# The timestep index is the position in the sorted list, matching the
		# detection stage; for a lone --image that is 0.
		yield frame_index, {camera_name: image}


def read_timestep_dir(args):
	"""Read `--timestep-dir` as one timestep with N views.

	Camera identity comes from each image's filename stem, and the timestep index
	from the directory name's trailing number when it has one, else 0.

	Returns:
		tuple: `(frame_index, {camera_name: image_bgr})`.
	"""
	paths = sorted(
		[p for p in glob.glob(os.path.join(args.timestep_dir, "*"))
		 if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")],
		key=get_index_file,
	)
	frames_t = {}
	for path in paths:
		image = cv2.imread(path)
		if image is None:
			logger.warning(f"Unreadable frame skipped: {path}")
			continue
		frames_t[os.path.splitext(os.path.basename(path))[0]] = image

	try:
		frame_index = get_index_file(os.path.normpath(args.timestep_dir))
	except ValueError:
		frame_index = 0
	return frame_index, frames_t

# endregion


# region Model and fusion setup

def load_models(config):
	"""Construct the detector, pose estimator and ReID extractor once.

	The pose estimator is loaded **once**, where the five-process path loaded
	ViTPose a second time. All three objects are held for the whole run on
	purpose: `BaseDetector` and `PoseEstimaterAdapter` free their weights in
	`__del__`, so dropping a reference mid-run would unload a model.

	Args:
		config (dict): Configuration dictionary. `detector.class_labels` must
			already hold the live `ClassLabels` object, which `load_detectors`
			requires.
	Returns:
		tuple: `(detector, pose_estimator, extractor)`.
	"""
	detector       = load_detectors(config)
	pose_estimator = load_pose_estimator(config)
	extractor      = load_feature_extraction_model(config)
	logger.info(f"Detector initialized: {detector}")
	return detector, pose_estimator, extractor


def make_all_trackers(config, class_labels, list_of_cameras):
	"""Build every camera's tracker set **before the first timestep**.

	`BoTSORT.__init__` calls `BaseTrack.clear_count()` unconditionally, so
	building a camera's trackers lazily on its first appearance would zero the
	process-global track-id counter underneath the cameras already running. With
	every set built up front, all of those resets happen before any `next_id()`
	call and the per-camera save/restore in `run_timestep` fully controls the
	counter.

	Args:
		config (dict): Configuration dictionary.
		class_labels (ClassLabels): Class label registry.
		list_of_cameras (list): Camera stems in scene order.
	Returns:
		dict: `{camera_name: {class_id: BoTSORT}}`.
	"""
	trackers_by_camera = {cam: make_trackers(config, class_labels) for cam in list_of_cameras}
	BaseTrack.clear_count()
	return trackers_by_camera


def setup_fusion(config, class_labels, list_of_cameras):
	"""Build everything the fusion stage needs, once, before the first timestep.

	Nothing this returns may be rebuilt per frame: the homographies and zones are
	pure functions of the calibration, and the tracker is the run's accumulating
	state.

	Args:
		config (dict): Configuration dictionary.
		class_labels (ClassLabels): Class label registry.
		list_of_cameras (list): Camera stems in scene order; `cam_idx` is the
			position in this list.
	Returns:
		tuple: `(bev_tracker, homographies, camera_matrices, zones_bev,
		mtmc_cfg, height_sampler)`.
	"""
	calib_dict = load_calib_infos(config)
	mtmc_cfg   = {**DEFAULT_MTMC, **(config.get("mtmc") or {})}
	logger.info(f"Config MTMC params for scene {config['data']['scene_name']}: {mtmc_cfg}")

	# Per-frame person height estimation (off by default).
	height_cfg     = resolve_height_estimation_cfg(mtmc_cfg.get("height_estimation"))
	height_sampler = None
	if height_cfg["enabled"]:
		height_sampler = PersonHeightSampler(
			height_cfg, mtmc_cfg["ankle_conf_thresh"], mtmc_cfg["pose_conf_thresh"])
		logger.info(f"Person height estimation enabled: {height_cfg}")

	homographies, camera_matrices = load_homographies(calib_dict, list_of_cameras)
	logger.info(f"Scene {config['data']['scene_name']}: {len(list_of_cameras)} cameras, "
				f"{len(homographies)} with calibration.")

	# Load the BEV zones for each camera, if specified in the config.
	zones_bev = {}
	zone_roi  = []
	for camera_name in list_of_cameras:
		zone_bev_path = os.path.join(config["regions"]["zone_bev"], f"{camera_name}.json")
		img_bev_path  = os.path.join(config["regions"]["zone_bev"], f"{camera_name}.jpg")
		if camera_name in calib_dict:
			camera_calib_dict      = calib_dict[camera_name]
			if len(zone_roi) == 0:
				# Ported verbatim from the five-process path, where it is also
				# built once and then never read.
				zone_roi = load_zone_bev(
					os.path.join(config["regions"]["zone_bev"], "map.json"),
					os.path.join(config["regions"]["zone_bev"], "map.jpg"),
					"ROI", camera_calib_dict)

	# Only Person and FourierGR1T2 are re-identified in the current MTMC pipeline.
	reid_class_ids = set()
	if class_labels is not None:
		for _name in ("Person", "FourierGR1T2"):
			try:
				_cid = class_labels.get_id_by_name(_name)
			except Exception:
				_cid = None
			if _cid is not None:
				reid_class_ids.add(_cid)

	tracker_cls = BEVFusionTrackerNOR
	bev_tracker = tracker_cls(class_labels, reid_class_ids, config["data"]["scene_id"],
							  config["data"]["frame_rate"], mtmc_cfg)

	return bev_tracker, homographies, camera_matrices, zones_bev, mtmc_cfg, height_sampler

# endregion


def run_timestep(config, class_labels, detector, pose_estimator, extractor,
				 trackers_by_camera, track_count_by_camera, zones_none_by_camera,
				 comp_dirs, list_of_cameras, frame_index, frames_t,
				 homographies, camera_matrices, mtmc_cfg, zones_bev, height_sampler):
	"""Run stages 1-4 for every camera present at `t` and project the results.

	Cameras are visited in `list_of_cameras` order, and each one's `cam_idx` is
	its **position in that list**, not the order it happened to be processed in:
	BEV fusion refuses to merge two observations sharing a `cam_idx`.

	**Every stage boundary goes through the store.** Each stage's comp is written
	to its tree and the local is then rebound from `read_comp`, so the next stage
	sees the file rather than the object that produced it. The rebinding is not
	bookkeeping: `pose_frame` and `track_frame` both mutate instances of their
	input in place, and the fresh graph `read_comp` returns is what makes those
	mutations unreachable -- exactly as reloading a pickle does in the five-process
	path. Keeping the written object and carrying on with it would reinstate every
	aliasing hazard while looking like it had fixed them.

	The `BaseTrack._count` save/restore around each camera's tracking call is
	**required, not defensive**. The counter is global class state on
	`BaseTrack`, and a camera-major run gets a counter that starts at 0 for every
	camera because `BoTSORT.__init__` clears it. Holding all cameras' trackers at
	once removes those mid-run resets, so without this guard each camera would
	continue the previous camera's numbering and every `track_id` would change.

	Args:
		config (dict): Configuration dictionary.
		class_labels (ClassLabels): Class label registry.
		detector (BaseDetector): Shared detector.
		pose_estimator (PoseEstimaterAdapter): Shared pose model.
		extractor (KPRFeatureExtractor): Shared ReID extractor.
		trackers_by_camera (dict): `{camera_name: {class_id: BoTSORT}}`, built once.
		track_count_by_camera (dict): `{camera_name: int}`, this camera's private
			view of `BaseTrack._count`. Mutated in place.
		zones_none_by_camera (dict): `{camera_name: polygons}` for the box filter.
		comp_dirs (dict): `{tree_key: {camera_name: folder}}` from
			`make_store_dirs`; the folders already exist, so nothing here calls
			`makedirs`.
		list_of_cameras (list): Camera stems in scene order.
		frame_index (int): The timestep.
		frames_t (dict): `{camera_name: image_bgr}` for this timestep. A camera
			missing from it has no image at `t`.
		homographies (dict): Per-camera ground homography.
		camera_matrices (dict): Per-camera calibration entry.
		mtmc_cfg (dict): Merged MTMC parameters.
		zones_bev (dict): Per-camera BEV zones.
		height_sampler (PersonHeightSampler): Or None.
	Returns:
		tuple: `(observations, stage_comps)`. `observations` is the concatenated
		list of `CamObservation`s for this timestep, empty for any camera without
		a homography. `stage_comps` is `[(camera_name, feat_comp, track_comp)]` in
		visit order, which the fusion-less input modes print instead of writing a
		submission.
	"""
	observations = []
	stage_comps  = []
	for camera_index, camera_name in enumerate(list_of_cameras):
		image = frames_t.get(camera_name)
		if image is None:
			continue

		# --- stage 1: detection ---
		det_comp = detect_frame(detector, config, camera_file_name(config, camera_name),
								frame_index, image)
		if det_comp is None:
			continue
		# Boundary 1 -> dets_comp. Write, then rebind the local from the store: the
		# stored copy is what stage 2 consumes, and the object just written is
		# dropped here rather than carried forward.
		write_comp(comp_dirs["dets_comp"][camera_name], frame_index, det_comp,
				   COMP_PROTOCOL["dets_comp"])
		det_comp = read_comp(comp_dirs["dets_comp"][camera_name], frame_index)

		# --- stage 2: pose estimation ---
		# The five-process path wrapped its whole camera loop in no_grad; here the
		# equivalent scope is the call itself.
		with torch.no_grad():
			pose_comp = pose_frame(pose_estimator, class_labels, det_comp, image)
		# Boundary 2 -> poses_comp.
		write_comp(comp_dirs["poses_comp"][camera_name], frame_index, pose_comp,
				   COMP_PROTOCOL["poses_comp"])
		pose_comp = read_comp(comp_dirs["poses_comp"][camera_name], frame_index)

		# --- stage 3: ReID features ---
		feat_comp = feature_frame(extractor, pose_comp, image)

		# Boundary 3 -> feats_comp.
		write_comp(comp_dirs["feats_comp"][camera_name], frame_index, feat_comp,
				   COMP_PROTOCOL["feats_comp"])
		feat_comp = read_comp(comp_dirs["feats_comp"][camera_name], frame_index)

		# --- stage 4: single-camera tracking ---
		# The counter guard: give this camera the counter it would have had in a
		# camera-major run, and hand it back untouched by the other cameras.
		BaseTrack._count = track_count_by_camera[camera_name]
		comp = track_frame(config, class_labels, trackers_by_camera[camera_name],
						   zones_none_by_camera[camera_name], feat_comp)
		track_count_by_camera[camera_name] = BaseTrack._count

		# Boundary 4 -> mots_single_comp. The instance schema changes shape here,
		# and this is the file the fusion stage reads in the five-process path.
		write_comp(comp_dirs["mots_single_comp"][camera_name], frame_index, comp,
				   COMP_PROTOCOL["mots_single_comp"])
		comp = read_comp(comp_dirs["mots_single_comp"][camera_name], frame_index)

		stage_comps.append((camera_name, feat_comp, comp))

		# --- stage 5 input: project to the shared ground plane ---
		homography_matrix = homographies.get(camera_name)
		if homography_matrix is None:
			continue
		observations += observations_from_comp(
			comp, camera_index, camera_name, homography_matrix,
			camera_matrices.get(camera_name), class_labels, mtmc_cfg,
			zones_bev, height_sampler,
		)

	return observations, stage_comps


def print_stage_comps(frame_index, stage_comps):
	"""Print one timestep's per-camera comps, for the input modes that skip fusion.

	The feature-stage instances are what the plan's cheapest regression test
	compares against a baseline `feats_comp` pickle, so they are printed with the
	fields that identify them numerically; the tracked instances follow.
	"""
	print(f"--- frame {frame_index} ---")
	for camera_name, feat_comp, track_comp in stage_comps:
		print(f"  camera {camera_name}: {feat_comp['width']}x{feat_comp['height']}, "
			  f"comp index {feat_comp['index']}, camera_name field "
			  f"{feat_comp['camera_name']!r}")
		print(f"    feature-stage instances ({len(feat_comp['instances'])}):")
		for instance in feat_comp["instances"]:
			bbox      = instance["det_bbox"]
			keypoints = instance["pose_keypoints"]
			feat      = instance["feat_embeddings"]
			print(f"      class_id={instance['class_id']} "
				  f"det_bbox={[round(float(v), 6) for v in bbox]} "
				  f"det_score={float(instance['det_score']):.6f} "
				  f"keypoints={0 if keypoints is None else len(keypoints)} "
				  f"embedding={None if feat is None else feat['embedding'].shape}")
		print(f"    tracked instances ({len(track_comp['instances'])}):")
		for instance in track_comp["instances"]:
			print(f"      class_id={instance['class_id']} "
				  f"track_id={instance['track_id']} "
				  f"track_score={instance['track_score']} "
				  f"track_bbox={[round(float(v), 6) for v in instance['track_bbox']]}")


def log_state_footprint(trackers_by_camera, bev_tracker):
	"""Log the persistent state that grows with run length.

	The per-timestep working set is bounded and small. What accumulates is
	BoT-SORT's per-camera track lists -- `removed_stracks` in particular is only
	ever extended, never pruned, and under frame-major execution every camera's
	tracker stays live for the whole run -- plus the fusion tracker's per-track
	frame history, which is the same state the five-process path already held.
	"""
	tracked = lost = removed = features = 0
	for by_class in trackers_by_camera.values():
		for tracker in by_class.values():
			tracked += len(tracker.tracked_stracks)
			lost    += len(tracker.lost_stracks)
			removed += len(tracker.removed_stracks)
			for bucket in (tracker.tracked_stracks, tracker.lost_stracks, tracker.removed_stracks):
				for strack in bucket:
					features += len(getattr(strack, "features", ()) or ())
	logger.info(f"BoTSORT state: tracked={tracked} lost={lost} removed={removed} "
				f"feature-deque entries={features}")

	tracks = getattr(bev_tracker, "tracks", None)
	if tracks:
		flat    = [t for v in (tracks.values() if isinstance(tracks, dict) else [tracks]) for t in v]
		frames  = sum(len(getattr(t, "frames", ()) or ()) for t in flat)
		history = sum(len(getattr(t, "history", ()) or ()) for t in flat)
		logger.info(f"BEV fusion state: tracks={len(flat)} frame entries={frames} "
					f"history entries={history}")


def main(args, config):
	"""Run the merged pipeline for one of the four input modes.

	The scene mode and `--timestep-dir` run fusion and write the submission;
	`--images` and `--image` supply a single view, so fusion is skipped with an
	explicit warning and no file is written -- never a zero-row `.txt`.

	All four modes drive the same store: the four comp trees are written
	unconditionally and are the channel between stages. Only their root differs --
	the fusion-less modes use a temporary directory removed on exit, so a debug run
	cannot overwrite a scene's pickles.

	Args:
		args (argparse.Namespace): Parsed CLI arguments.
		config (dict): Configuration already returned by `pin_cuda_visible_devices`,
			so `CUDA_VISIBLE_DEVICES` is set and the pose estimator's device index
			has been rewritten to the local ordinal.
	"""
	# `load_detectors` requires the live ClassLabels object on the detector block.
	class_labels = load_class_labels(config)
	config["detector"]["class_labels"] = class_labels

	fusion_runs = args.images is None and args.image is None

	# ---- resolve the camera list and the timestep source ----
	if args.timestep_dir is not None:
		frame_index, frames_t = read_timestep_dir(args)
		list_of_cameras       = sorted(frames_t.keys(), key=get_index_file)
		timesteps             = iter([(frame_index, frames_t)])
	elif args.images is not None or args.image is not None:
		timesteps       = iter_single_camera(config, args)
		probe           = args.image if args.image is not None else args.images
		list_of_cameras = [camera_name_from_path(probe, args.camera_name)]
	else:
		list_of_cameras = list_scene_cameras(config, args.cameras)
		timesteps       = iter_timesteps(config, args, list_of_cameras)

	if not list_of_cameras:
		logger.error("No cameras resolved from the given input; nothing to do.")
		return
	logger.info(f"Processing scene: {config['data']['scene_name']} "
				f"with {len(list_of_cameras)} cameras: {list_of_cameras}")

	# ---- models, trackers, zones ----
	detector, pose_estimator, extractor = load_models(config)

	trackers_by_camera    = make_all_trackers(config, class_labels, list_of_cameras)
	track_count_by_camera = {cam: 0 for cam in list_of_cameras}
	zones_none_by_camera  = {}
	# zones_none_by_camera  = {
	# 	cam: load_zone_camera(os.path.join(config["regions"]["zone_camera"], f"{cam}.json"))
	# 	for cam in list_of_cameras
	# }

	# ---- the out-of-memory store: every stage boundary is a file ----
	# The two fusion-less modes are debug modes and get a throwaway store root:
	# writing them under `data_writer.root` would silently overwrite the scene's
	# pickles for whatever frames they happen to touch. Everything else about the
	# store is identical between the modes, so there is only one execution
	# contract -- the destination differs, the channel does not.
	verify_store_is_channel()
	store_tempdir = (None if fusion_runs
					 else tempfile.TemporaryDirectory(prefix="run_pipeline_store_"))
	comp_dirs     = make_store_dirs(config, list_of_cameras,
									None if store_tempdir is None else store_tempdir.name)

	# ---- fusion setup, and the reasons it may be skipped ----
	bev_tracker = homographies = camera_matrices = zones_bev = height_sampler = None
	mtmc_cfg    = {**DEFAULT_MTMC, **(config.get("mtmc") or {})}
	if fusion_runs:
		(bev_tracker, homographies, camera_matrices,
		 zones_bev, mtmc_cfg, height_sampler) = setup_fusion(config, class_labels, list_of_cameras)
		uncalibrated = [c for c in list_of_cameras if homographies.get(c) is None]
		if uncalibrated:
			logger.warning(
				f"no calibration entry for {uncalibrated} in "
				f"{config['data']['calibration']['file']}; those cameras contribute "
				f"no observations")
		if args.timestep_dir is not None:
			logger.warning(
				"single-timestep mode: every row will carry yaw = 0.00000000 "
				"(find_yaw needs at least two frames) and object_ids are not "
				"persistent -- this is not a scene submission")
	else:
		logger.warning(
			"multi-camera fusion skipped: 1 camera view supplied; "
			"BEVFusionTracker._cluster merges only across distinct cameras. "
			"No submission file will be written, and the four comp trees go to a "
			"temporary store discarded on exit -- the scene's pickles under "
			"data_writer.root are left untouched.")
		homographies = camera_matrices = {}
		zones_bev    = {}

	# ---- the loop ----
	try:
		for frame_index, frames_t in timesteps:
			observations, stage_comps = run_timestep(
				config, class_labels, detector, pose_estimator, extractor,
				trackers_by_camera, track_count_by_camera, zones_none_by_camera,
				comp_dirs, list_of_cameras, frame_index, frames_t,
				homographies, camera_matrices, mtmc_cfg, zones_bev, height_sampler,
			)
			if fusion_runs:
				bev_tracker.update(bev_tracker.group_observations(observations), frame_index)
			else:
				print_stage_comps(frame_index, stage_comps)
	finally:
		if store_tempdir is not None:
			store_tempdir.cleanup()

	if not fusion_runs:
		log_state_footprint(trackers_by_camera, bev_tracker)
		return

	# ---- finalize and write the submission ----
	result = bev_tracker.finalize()
	if height_sampler is not None:
		height_sampler.log_summary()
	log_state_footprint(trackers_by_camera, bev_tracker)

	result_path = os.path.join(
		config["data_writer"]["root"],
		config["data_writer"]["mots_multi"],
		f"{config['data']['scene_name']}.txt",
	)
	num_rows   = write_submission(result, config["data"]["scene_id"], result_path, class_labels,
								  height_estimation_enabled=height_sampler is not None)
	num_tracks = sum(len(v) for v in result.values())
	logger.info(f"Wrote {num_rows} rows for {num_tracks} global tracks -> {result_path}")


if __name__ == "__main__":
	main(_ARGS, _CONFIG)
