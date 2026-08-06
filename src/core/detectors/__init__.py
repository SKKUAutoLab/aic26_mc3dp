#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Detector classes.
"""

from __future__ import annotations

from .basedetector import *

def load_detectors(config):
	# Initialize the object detector based on the configuration
	detector_name = config['detector']['name']
	if detector_name == "yolov11":
		from .yolov11_adaptor import YOLOv11_Adapter
		detector = YOLOv11_Adapter(**config['detector'])
	elif detector_name == "yolov26":
		from .yolov26_adaptor import YOLOv26_Adapter
		detector = YOLOv26_Adapter(**config['detector'])
	elif detector_name == "rfdetr":
		from .rfdetr_adaptor import RFDETR_Adapter
		detector = RFDETR_Adapter(**config['detector'])
	else:
		raise ValueError(f"Unsupported detector: {detector_name}")
	return detector
