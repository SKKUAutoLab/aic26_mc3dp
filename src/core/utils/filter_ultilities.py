import os
import sys
import glob
import argparse
import pickle
import random
from pathlib import Path
import shutil
from dataclasses import dataclass
from copy import deepcopy
from collections import Counter

import yaml
import numpy as np
from loguru import logger
from tqdm import tqdm
import json

from shapely.geometry import box, Polygon, Point
import cv2

# region Filter any bounding box that is completely contained within a NONE zone


def transform_polygon(polygon, homography_matrix):
	"""Transform a shapely polygon using a homography matrix."""
	if polygon.is_empty:
		return polygon

	# Convert the polygon to a numpy array of shape (N, 2)
	points = np.array(polygon.exterior.coords)

	# Convert points to homogeneous coordinates (N, 3)
	points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

	# Apply the homography matrix
	transformed_points_homogeneous = points_homogeneous @ homography_matrix.T

	# Convert back to Cartesian coordinates (world coordinates) by dividing by the last coordinate
	transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2][:, np.newaxis]

	# Create a new polygon with the transformed points
	transformed_polygon = Polygon(transformed_points)

	return transformed_polygon


def map_pixel_to_world_cartesian(points, scale, translation, map_height):
    """Inverse of the world->map.png affine: map pixel (u,v) -> Cartesian world (x,y)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    tx, ty = float(translation["x"]), float(translation["y"])
    x = pts[:, 0] / scale - tx
    y = (map_height - pts[:, 1]) / scale - ty
    return np.column_stack([x, y])


def load_zone_bev(zone_path, img_bev_path, camera_name, camera_calib_dict = None) -> list[dict]:
	"""Load NONE zones from a labelme-style JSON file as shapely geometries.

	Vertices are kept in their annotated (ring) order so that concave NONE
	zones are represented faithfully. Angular reordering is intentionally NOT
	applied: sorting vertices by angle around the centroid only reconstructs
	convex polygons and silently corrupts concave ones (see
	``convert_clockwise_points``).
	
	Args:
		zone_path: Path to the zone JSON file using the labelme "shapes" schema.
		img_bev_path: Path to the BEV image file.
		camera_name: Name of the camera for which to load zones.
			>>> camera_name = "Camera_0004"
			>>> camera_name = "ROI"  # for zones that are not camera-specific
		camera_calib_dict: Dictionary containing camera calibration parameters.
	Returns:
		One ``{"polygon": <shapely geometry>, "class_id": <group_id>}`` dict per
		shape labelled ``"NONE"``. Empty list if the file does not exist.
	"""
	zones_bev = []
	
	if os.path.exists(zone_path) and os.path.isfile(zone_path) and os.path.exists(img_bev_path) and os.path.isfile(img_bev_path):
		
		img_bev   = cv2.imread(img_bev_path)
		img_height, img_width = img_bev.shape[:2]

		with open(zone_path, "r") as f:
			zone_data = json.load(f)

		for shape in zone_data["shapes"]:
			if shape["label"] not in [camera_name, "ROI", "roi"]:
				continue

			points = deepcopy(shape["points"])

			points_are_world = False
			# convert points from bev_map image (pixel) to world coordinates
			if camera_calib_dict is not None:
				scale = camera_calib_dict.get("scaleFactor")
				trans = camera_calib_dict.get("translationToGlobalCoordinates")
				if scale and trans:
					map_height = img_height      # 1080, already in the file
					points     = map_pixel_to_world_cartesian(points, scale, trans, map_height).tolist()

					points_are_world = True

			if not points_are_world:
				continue

			# Trust the annotated vertex order so concave rings are preserved.
			polygon = Polygon(points)

			# Repair self-intersecting / degenerate rings into a valid geometry.
			if not polygon.is_valid:
				polygon = polygon.buffer(0)
			if polygon.is_empty:
				continue
			zones_bev.append({
				"polygon": polygon,
				"camera_name": camera_name,
			})
	return zones_bev


def load_zone_camera(zone_path) -> list[dict]:
	"""Load NONE zones from a labelme-style JSON file as shapely geometries.

	Vertices are kept in their annotated (ring) order so that concave NONE
	zones are represented faithfully. Angular reordering is intentionally NOT
	applied: sorting vertices by angle around the centroid only reconstructs
	convex polygons and silently corrupts concave ones (see
	``convert_clockwise_points``).

	Args:
		zone_path: Path to the zone JSON file using the labelme "shapes" schema.

	Returns:
		One ``{"polygon": <shapely geometry>, "type": <group_id>}`` dict per
		shape labelled ``"NONE"`` or ``"ROI"``. Empty list if the file does not exist.
	"""
	zone_none = []
	if os.path.exists(zone_path) and os.path.isfile(zone_path):
		with open(zone_path, "r") as f:
			zone_data = json.load(f)
		for shape in zone_data["shapes"]:
			if shape["label"] not in ["NONE", "ROI"]:
				continue
			# Trust the annotated vertex order so concave rings are preserved.
			polygon = Polygon(shape["points"])
			# Repair self-intersecting / degenerate rings into a valid geometry.
			if not polygon.is_valid:
				polygon = polygon.buffer(0)
			if polygon.is_empty:
				continue
			zone_none.append({
				"polygon": polygon,
				"type": shape["label"],
			})
	return zone_none


def convert_clockwise_points(points: list[list[float]] | np.ndarray) -> list[list[float]]:
	"""Order points clockwise by angle around their centroid.

	WARNING: convex-only. This reconstructs the ring of a convex point set but
	corrupts concave polygons. NONE zones may be concave, so ``load_zone`` does
	NOT use this and trusts the annotated vertex order instead.
	"""
	# Convert points to numpy array for easier manipulation
	points_array = np.array(points)

	# Calculate the centroid of the points
	centroid = np.mean(points_array, axis=0)

	# Calculate the angle of each point with respect to the centroid
	angles = np.arctan2(points_array[:, 1] - centroid[1], points_array[:, 0] - centroid[0])

	# Sort the points based on the angles in clockwise order
	sorted_indices = np.argsort(angles)
	sorted_points = points_array[sorted_indices]

	return sorted_points.tolist()


def is_bbox_in_zone(zones, bbox_polygon) -> bool:
	"""Check whether a bbox is completely contained in any zone.

	Works for concave zones: ``contains`` tests the true polygon footprint, so a
	bbox lying in a concave "notch" (outside the polygon but inside its convex
	hull) is correctly reported as NOT inside the zone.

	Args:
		zones: Zones as returned by :func:`load_zone_camera`.
		bbox_polygon: Axis-aligned bounding box as a shapely geometry.

	Returns:
		True if ``bbox_polygon`` lies completely inside at least one zone.
	"""
	for zone in zones:
		polygon = zone["polygon"]
		if polygon.is_empty:
			continue
		if polygon.contains(bbox_polygon):
			return True
	return False
	

def is_point_in_zone(zones, point, distance_to_polygon_thresh = 1.0) -> bool:
	"""Check whether a point is inside any zone.

	Args:
		zones: Zones as returned by :func:`load_zone_camera`.
		point: A tuple (x, y) representing the point coordinates.

	Returns:
		True if the point lies inside at least one zone.
	"""
	if not isinstance(point, Point):
		point = Point(point)

	for zone in zones:
		polygon = zone['polygon']
		
		# Skip empty polygons to avoid errors in the contains() method.
		if polygon.is_empty:
			continue

		# Check if the point is strictly inside the polygon
		if polygon.contains(point):
			# DEBUG:
			# print(f"Point {point} is inside zone: {polygon}")
			return True

		# Allow points on the boundary (distance 0) or very close to it (distance <= 1) to be considered inside the zone.
		if point.distance(polygon) <= distance_to_polygon_thresh: 
			# DEBUG:
			# print(f"Distance from Point {point} is on the boundary of zone: {polygon}")
			return True
	return False

# endregion
