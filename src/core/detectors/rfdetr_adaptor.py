from __future__ import annotations

import numpy as np

from loguru import logger
from torch import Tensor

from core.utils.image import is_channel_first
from core.factory.builder import DETECTORS
from core.objects.instance import Instance
from core.detectors import BaseDetector

from rfdetr import (
	RFDETRBase,
	RFDETRLarge,
	RFDETRMedium,
	RFDETRNano,
	RFDETRSmall,
)

# `RFDETRXLarge` / `RFDETR2XLarge` are "plus" models served from the optional
# `rfdetr_plus` package; `rfdetr` re-exports them lazily and raises ImportError
# when that package is absent. Guard the import so the adapter still works for
# the standard variants when the plus package is not installed.
try:
	from rfdetr import RFDETR2XLarge, RFDETRXLarge
except ImportError:
	RFDETRXLarge  = None
	RFDETR2XLarge = None

__all__ = [
	"RFDETR_Adapter"
]


# Map a normalized variant key -> RF-DETR variant class. Keys are the size word
# only; `_resolve_variant` strips the "rfdetr"/separator noise before lookup.
RFDETR_VARIANTS = {
	"nano"   : RFDETRNano,
	"small"  : RFDETRSmall,
	"medium" : RFDETRMedium,
	"large"  : RFDETRLarge,
	"base"   : RFDETRBase,
}

# Plus-only variants. Values are `None` when `rfdetr_plus` is not installed, in
# which case `_resolve_variant` raises a clear install hint if one is requested.
RFDETR_PLUS_VARIANTS = {
	"xlarge"  : RFDETRXLarge,
	"xxlarge" : RFDETR2XLarge,
	"2xlarge" : RFDETR2XLarge,
}

# Class names that require explicit platform-model license acceptance.
RFDETR_PLUS_CLASS_NAMES = ("RFDETRXLarge", "RFDETR2XLarge")


# MARK: - RFDETR

@DETECTORS.register(name="rfdetr")
class RFDETR_Adapter(BaseDetector):

	# MARK: Magic Functions

	def __init__(self,
	             name: str = "rfdetr",
	             *args, **kwargs):
		super().__init__(name=name, *args, **kwargs)

	# MARK: Configure

	def _resolve_variant(self):
		"""Pick the RF-DETR variant class from `self.variant` (fallback `self.name`).

		Accepts values like "nano", "rfdetr-small", "RFDETRMedium", "xlarge",
		"2xlarge". Raises `ImportError` (with an install hint) when a plus-only
		variant is requested but `rfdetr_plus` is not installed, and falls back to
		`RFDETRBase` (with a warning) for unrecognized variants.
		"""
		key = (self.variant or self.name or "").lower()
		for sep in ("-", "_", " "):
			key = key.replace(sep, "")
		key = key.replace("rfdetr", "")

		if key in RFDETR_VARIANTS:
			return RFDETR_VARIANTS[key]

		if key in RFDETR_PLUS_VARIANTS:
			variant_class = RFDETR_PLUS_VARIANTS[key]
			if variant_class is None:
				raise ImportError(
					f"RF-DETR variant '{self.variant}' requires the optional 'rfdetr_plus' "
					f"package. Install it with `pip install rfdetr[plus]` "
					f"(or `pip install rfdetr_plus`)."
				)
			return variant_class

		logger.warning(
			f"Unknown RF-DETR variant '{self.variant}', falling back to RFDETRBase. "
			f"Supported variants: {sorted(list(RFDETR_VARIANTS) + list(RFDETR_PLUS_VARIANTS))}."
		)
		return RFDETRBase

	def init_model(self):
		"""Create and load model from weights."""
		# Get image size of detector (square side length), mirroring the YOLO adapters.
		self.img_size = None
		if self.shape is not None:
			if is_channel_first(np.ndarray(self.shape)):
				self.img_size = self.shape[2]
			else:
				self.img_size = self.shape[0]

		# Number of foreground classes for the detection head. The model_cfg value
		# wins when provided, otherwise it is derived from the class labels. The
		# head size must match the checkpoint in `self.weights`.
		num_classes = None
		if isinstance(self.model_cfg, dict) and self.model_cfg.get("num_classes"):
			num_classes = int(self.model_cfg["num_classes"])
		elif self.class_labels is not None:
			n = self.class_labels.num_classes(key="train_id")
			num_classes = n if n > 0 else None

		# NOTE: load model. `pretrain_weights` is passed explicitly (even when None)
		# so the published variant default is never auto-downloaded behind our back.
		variant_class = self._resolve_variant()
		model_kwargs = {
			"pretrain_weights" : self.weights,
			"device"           : str(self.device),
		}
		if self.img_size:
			# RF-DETR resolution must be divisible by `patch_size * num_windows`.
			model_kwargs["resolution"] = self.img_size
		if num_classes:
			model_kwargs["num_classes"] = num_classes
		if variant_class.__name__ in RFDETR_PLUS_CLASS_NAMES:
			# XLarge / 2XLarge are platform models gated behind a license.
			model_kwargs["accept_platform_model_license"] = True
		self.model = variant_class(**model_kwargs)

		# Optional inference-time optimization (eval mode + fused graph). `compile=False`
		# avoids torch.compile's fixed-batch-size requirement so variable last batches
		# still run. Best-effort: unoptimized inference works if this fails.
		try:
			self.model.optimize_for_inference(compile=False)
		except Exception as e:
			logger.warning(f"RF-DETR optimize_for_inference skipped: {e}")

	# MARK: Detection

	def detect(self, indexes: np.ndarray, images: np.ndarray) -> list:
		"""Detect objects in the images.

		Args:
			indexes (np.ndarray):
				Image indexes.
			images (np.ndarray):
				Images of shape [B, H, W, C].

		Returns:
			instances (list):
				List of `Instance` objects.
		"""
		# NOTE: Safety check
		if self.model is None:
			logger.error("Model has not been defined yet!")
			raise NotImplementedError

		# NOTE: Preprocess
		input_imgs = self.preprocess(images=images)
		# NOTE: Forward
		preds  = self.forward(input_imgs)
		# NOTE: Postprocess
		instances = self.postprocess(
			indexes=indexes, images=images, input_imgs=input_imgs, pred=preds
		)

		return instances

	def preprocess(self, images: np.ndarray):
		"""Preprocess the input images to model's input image.

		The pipeline feeds BGR uint8 images (OpenCV convention) while RF-DETR's
		`predict()` expects RGB. Convert here and return a list of contiguous
		HWC uint8 RGB arrays; `predict()` handles tensor conversion and resizing.

		Args:
			images (np.ndarray):
				Images of shape [B, H, W, C], BGR channel order.

		Returns:
			input_imgs (list):
				List of HWC uint8 RGB `np.ndarray`.
		"""
		input_imgs = [np.ascontiguousarray(image[:, :, ::-1]) for image in images]
		return input_imgs

	def forward(self, input_imgs):
		"""Forward pass.

		Args:
			input_imgs (list):
				List of HWC uint8 RGB images.

		Returns:
			pred (list):
				List of supervision `Detections`, one per input image.
		"""
		pred = self.model.predict(
			input_imgs,
			threshold=self.min_confidence,
			include_source_image=False,
		)
		# `predict()` returns a single `Detections` for a single image and a list
		# otherwise; normalize to a list aligned with `indexes`.
		if not isinstance(pred, list):
			pred = [pred]
		return pred

	def postprocess(
			self,
			indexes   : np.ndarray,
			images    : np.ndarray,
			input_imgs,
			pred,
			*args, **kwargs
	) -> list:
		"""Postprocess the prediction.

		RF-DETR returns boxes as absolute-pixel `xyxy`; convert them to normalized
		center `xywh` (cx, cy, w, h) to match the `Instance.bbox` convention used by
		the YOLO adapters and the downstream pipeline.

		Args:
			indexes (np.ndarray): order of images
				Image indexes.
			images (np.ndarray): raw images
				Images of shape [B, H, W, C].
			input_imgs (list): pre_processed images
				List of RGB images fed to the model.
			pred (list): results from model
				List of supervision `Detections`.

		Returns:
			instances (list):
				List of `Instances` objects.
		"""
		# NOTE: Create Detection objects
		instances = []

		for idx, (frame_index, detections) in enumerate(zip(indexes, pred)):
			inst = []
			image_height, image_width = images[idx].shape[:2]

			boxes = detections.xyxy             # (N, 4) absolute pixels
			confs = detections.confidence       # (N,)
			clses = detections.class_id         # (N,)
			if boxes is None or len(boxes) == 0:
				instances.append(inst)
				continue

			for box, conf, cls in zip(boxes, confs, clses):
				x_min, y_min, x_max, y_max = box
				cx = ((x_min + x_max) / 2.0) / image_width
				cy = ((y_min + y_max) / 2.0) / image_height
				w  = (x_max - x_min) / image_width
				h  = (y_max - y_min) / image_height
				bbox = np.array([cx, cy, w, h], dtype=np.float32)

				confident   = float(conf)
				class_id    = int(cls)
				class_label = self.class_labels.get_class_label(
					key="train_id", value=class_id
				)
				inst.append(
					Instance(
						frame_index = frame_index,
						bbox        = bbox,
						confidence  = confident,
						class_label = class_label,
						label       = class_label,
						class_id    = class_id,
						image_size  = images[idx].shape[:2]  # (H, W)
					)
				)

			instances.append(inst)
		return instances
