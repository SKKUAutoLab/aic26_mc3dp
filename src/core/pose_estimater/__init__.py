#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pose estimator classes.
"""

from __future__ import annotations

def load_pose_estimator(config):
	# Initialize the pose estimator based on the configuration
	pose_estimator_name = config['reidentifier']['pose_estimator']['name']
	if pose_estimator_name in ["vitpose-plus-huge", "vitpose-plus-base"]:
		from .vitpose_adapter import PoseEstimaterAdapter
		pose_estimator = PoseEstimaterAdapter(**config['reidentifier']['pose_estimator'])
	else:
		raise ValueError(f"Unsupported pose estimator: {pose_estimator_name}")
	return pose_estimator