from __future__ import annotations

from copy import deepcopy
from typing import Optional, Union

import cv2
import numpy as np
import torch

from loguru import logger
from PIL import Image

from transformers import AutoProcessor, VitPoseForPoseEstimation

__all__ = [
	"PoseEstimaterAdapter"
]


# MARK: - ViTPose

class PoseEstimaterAdapter:
	"""ViTPose pose estimator adapter.

	Create the ViTPose model based on the provided configuration and process
	input image with bounding boxes to return pose estimation results in the
	format of `result_dict["instances"]`.

	Attributes:
		name (str):
			Name of the pose estimator model.
		model (VitPoseForPoseEstimation):
			Pose estimation model.
		image_processor (AutoProcessor):
			Image processor for preprocessing and postprocessing.
		weights (str):
			Path to the pretrained weights/model folder.
		device (str):
			Cuda device, i.e. 0 or 0,1,2,3 or cpu.
		dataset_index (int):
			Index of the dataset expert to use for ViTPose+ (MoE) models.
			Default: `0` (COCO).
	"""

	# MARK: Magic Functions

	def __init__(
			self,
			name         : str           = "vitpose-plus-huge",
			model        : Optional[str] = None,
			device       : Optional[str] = None,
			dataset_index: int           = 0,
			*args, **kwargs
	):
		super().__init__()
		self.name            = name
		self.weights         = model
		self.device          = "cpu" if device in [None, "cpu"] else f"cuda:{device}"
		self.dataset_index   = dataset_index
		self.model           = None
		self.image_processor = None

		# NOTE: Load model
		self.init_model()

	def __del__(self):
		self.clear_model_memory()

	def __str__(self):
		return (
			f"{self.__class__.__name__}(\n"
			f"    name          = {self.name},\n"
			f"    weights       = {self.weights},\n"
			f"    device        = {self.device},\n"
			f"    dataset_index = {self.dataset_index},\n"
			f"    model_loaded  = {self.model is not None}\n"
			f")"
		)

	# MARK: Configure

	def init_model(self):
		"""Create and load model from weights."""
		# NOTE: load image processor and model
		self.image_processor = AutoProcessor.from_pretrained(self.weights)
		self.model           = VitPoseForPoseEstimation.from_pretrained(
			self.weights, device_map=self.device
		)
		self.model.eval()

	# MARK: Pose Estimation

	def forward(
			self,
			image    : Union[np.ndarray, Image.Image],
			bbox_xywh: np.ndarray
	) -> list:
		"""Estimate poses for each bounding box in the image.

		Args:
			image (np.ndarray, PIL.Image.Image):
				Image of shape [H, W, C] in BGR format (cv2), or a PIL image
				in RGB format.
			bbox_xywh (np.ndarray):
				Bounding boxes of shape [N, 4] in COCO (x_top_left, y_top_left,
				width, height) format, in absolute pixel values.

		Returns:
			pose_results (list):
				List of dicts, one for each bounding box, in the format of
				`result_dict["instances"]`:
					"pose_bbox"     : (x_top_left, y_top_left, width, height)
					                  in absolute pixel values.
					"pose_keypoints": list of {"name", "x", "y", "score"}.
		"""
		# NOTE: Safety check
		if self.model is None:
			logger.error("Model has not been defined yet!")
			raise NotImplementedError

		# If no bounding box provided, return empty results
		bbox_xywh = np.array(bbox_xywh, dtype=np.float32).reshape(-1, 4)
		if len(bbox_xywh) == 0:
			return []

		# NOTE: Preprocess
		inputs = self.preprocess(image=image, bbox_xywh=bbox_xywh)

		# NOTE: Forward
		with torch.no_grad():
			dataset_index = torch.tensor([self.dataset_index], device=self.model.device)  # must be a tensor of shape (batch_size,)
			outputs       = self.model(**inputs, dataset_index=dataset_index)
		
		# NOTE: Postprocess
		pose_results = self.postprocess(outputs=outputs, bbox_xywh=bbox_xywh)

		return pose_results

	def preprocess(
			self,
			image    : Union[np.ndarray, Image.Image],
			bbox_xywh: np.ndarray
	):
		"""Preprocess the input image to model's input.

		Args:
			image (np.ndarray, PIL.Image.Image):
				Image of shape [H, W, C] in BGR format (cv2), or a PIL image
				in RGB format.
			bbox_xywh (np.ndarray):
				Bounding boxes of shape [N, 4] in COCO (x_top_left, y_top_left,
				width, height) format, in absolute pixel values.

		Returns:
			inputs (dict):
				Model's input.
		"""
		# Convert from BGR of CV2 format -> RGB of PIL format
		if isinstance(image, np.ndarray):
			image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

		inputs = self.image_processor(
			image, boxes=[bbox_xywh], return_tensors="pt"
		).to(self.model.device)
		return inputs

	def postprocess(self, outputs, bbox_xywh: np.ndarray) -> list:
		"""Postprocess the prediction.

		Args:
			outputs:
				Raw outputs from the model.
			bbox_xywh (np.ndarray):
				Bounding boxes of shape [N, 4] in COCO (x_top_left, y_top_left,
				width, height) format, in absolute pixel values.

		Returns:
			pose_results (list):
				List of dicts, one for each bounding box, in the format of
				`result_dict["instances"]` ("pose_bbox", "pose_keypoints").
		"""
		# Post-process the pose estimation results to absolute pixel coordinates
		image_pose_result = self.image_processor.post_process_pose_estimation(
			outputs, boxes=[bbox_xywh]
		)[0]

		# NOTE: Create pose result dicts
		pose_results = []
		for person_pose in image_pose_result:
			pose_keypoints = []
			for keypoint, label, score in zip(
				person_pose["keypoints"], person_pose["labels"], person_pose["scores"], strict=True
			):
				keypoint_name = self.model.config.id2label[label.item()]
				x, y          = keypoint
				pose_keypoints.append({
					"name" : keypoint_name,
					"x"    : x.item(),
					"y"    : y.item(),
					"score": score.item()
				})
			pose_results.append({
				"pose_bbox"     : person_pose["bbox"].numpy().tolist(),  # (x_top, y_left, ...) in absolute pixel values
				"pose_keypoints": pose_keypoints,
			})

		return pose_results

	# MARK: Utils

	def clear_model_memory(self):
		"""Free the memory of model

		Returns:
			None
		"""
		if self.model is not None:
			del self.model
			torch.cuda.empty_cache()
