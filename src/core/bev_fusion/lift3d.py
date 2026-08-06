import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class BoxDimensions:
	"""Per-class world box size in meters (scale order: width, length, height)."""
	width  : float
	length : float
	height : float


# class_id -> BoxDimensions in meters.
CLASS_DIMENSIONS = {
	# 0: BoxDimensions(0.587, 0.408, 1.720), # Person (2025 GT median, n=157725)  
	0: BoxDimensions(0.587, 0.56, 1.720), # Person (2025 GT median, n=157725)
	1: BoxDimensions(1.8731641267669, 1.12938480437337, 2.76786805165108), # Forklift (2025 GT median, n=2160)
	2: BoxDimensions(0.707, 0.477, 0.5636221), # NovaCarter (2025 GT median, n=5309)
	3: BoxDimensions(1.431, 0.652, 0.23397565), # Transporter (2025 GT median, n=4128) -- low/flat AMR
	4: BoxDimensions(0.599, 0.358, 1.6592345), # FourierGR1T2 (2025 GT median, n=1080)
	5: BoxDimensions(0.735, 0.533, 1.7572125), # AgilityDigit (2025 GT median, n=180)
	6: BoxDimensions(0.700, 1.600, 1.300), # PalletTruck (estimate; no 2025 GT equivalent)
}


def lift_to_3d(class_id: int, world_xy: np.ndarray) -> tuple:
	"""Lift a world ground point to a 3D box.

	Returns ``(x, y, z, w, l, h, yaw)`` with ``z = h/2`` (box centered vertically
	on a floor-standing object), ``(w, l, h)`` from ``CLASS_DIMENSIONS`` and
	``yaw = 0.0``.
	"""
	dims = CLASS_DIMENSIONS[class_id]
	x    = float(world_xy[0])
	y    = float(world_xy[1])
	z    = dims.height / 2.0
	return (x, y, z, dims.width, dims.length, dims.height, 0.0)


def angle_with_polyfit_degree_2(world_xys, eval_index=-1, min_speed=1e-6):
	"""Estimate heading (yaw) at one point of a world-XY trajectory window.

	Fits the window parametrically -- ``x(t)`` and ``y(t)`` as separate
	polynomials of the sample index ``t`` -- and returns the tangent direction
	``atan2(y'(t), x'(t))`` at the requested sample. ``t`` is centered on the
	evaluation point so it stays small (well-conditioned ``polyfit``) and the
	derivative there is just the linear coefficient of each fit. Degree 2 is used
	with three or more points, degree 1 with exactly two; the heading is left
	undefined below that or when the window is (near-)stationary.

	Args:
		world_xys: Ordered ground positions ``[(x, y), ...]`` for the window, in
			meters, in trajectory time order.
		eval_index: Index into ``world_xys`` at which to evaluate the heading
			(supports negative indexing); defaults to the last point.
		min_speed: Minimum fitted speed ``hypot(x'(t), y'(t))`` below which the
			heading is treated as undefined.

	Returns:
		Heading in radians in ``(-pi, pi]``, or ``None`` when fewer than two
		points are given or the window is (near-)stationary.
	"""
	points = np.asarray(world_xys, dtype=np.float64)
	n = len(points)
	degree = 2 if n >= 3 else (1 if n == 2 else None)
	if degree is None:
		return None

	# Center t on the evaluation point (t=0 there): keeps polyfit well-conditioned
	# and makes each fit's derivative at the point its linear coefficient (-2).
	center = range(n)[eval_index]  # normalizes negative indices and bounds-checks
	t = np.arange(n, dtype=np.float64) - center
	vx = np.polyfit(t, points[:, 0], degree)[-2]
	vy = np.polyfit(t, points[:, 1], degree)[-2]

	if np.hypot(vx, vy) < min_speed:
		return None
	return float(np.arctan2(vy, vx))


def format_row(scene_id, class_id, object_id, frame_id, x, y, z, width, length, height, yaw) -> str:
	"""Format one submission line.

	``<scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>``
	with the four ids as bare integers and the seven floats at 2|8 decimals,
	single-space separated (no trailing newline, no comment markers).
	"""
	return (
		f"{int(scene_id)} {int(class_id)} {int(object_id)} {int(frame_id)} "
		f"{float(x):.8f} {float(y):.8f} {float(z):.8f} {float(width):.8f} {float(length):.8f} {float(height):.8f} {float(yaw):.8f}"
	)
