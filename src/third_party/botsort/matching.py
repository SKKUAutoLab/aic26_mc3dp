import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from botsort import kalman_filter
import pdb

try:
	from cython_bbox import bbox_overlaps as bbox_ious
except ImportError:
	def bbox_ious(boxes1, boxes2):
		"""
		Calculate IoU between two sets of boxes.
		boxes1: (N, 4) ndarray of [x1, y1, x2, y2]
		boxes2: (M, 4) ndarray of [x1, y1, x2, y2]
		Returns: (N, M) ndarray of IoUs
		"""
		N = boxes1.shape[0]
		M = boxes2.shape[0]
		ious = np.zeros((N, M), dtype=np.float64)

		for i in range(N):
			box1 = boxes1[i]
			x1_1, y1_1, x2_1, y2_1 = box1
			area1 = (x2_1 - x1_1) * (y2_1 - y1_1)

			for j in range(M):
				box2 = boxes2[j]
				x1_2, y1_2, x2_2, y2_2 = box2
				area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

				# Calculate intersection
				xx1 = max(x1_1, x1_2)
				yy1 = max(y1_1, y1_2)
				xx2 = min(x2_1, x2_2)
				yy2 = min(y2_1, y2_2)

				w = max(0, xx2 - xx1)
				h = max(0, yy2 - yy1)
				inter = w * h

				# Calculate union
				union = area1 + area2 - inter
				ious[i, j] = inter / union if union > 0 else 0.0

		return ious

def merge_matches(m1, m2, shape):
	O,P,Q = shape
	m1 = np.asarray(m1)
	m2 = np.asarray(m2)

	M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
	M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

	mask = M1*M2
	match = mask.nonzero()
	match = list(zip(match[0], match[1]))
	unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
	unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

	return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
	matched_cost = cost_matrix[tuple(zip(*indices))]
	matched_mask = (matched_cost <= thresh)

	matches = indices[matched_mask]
	unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
	unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

	return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
	if cost_matrix.size == 0:
		return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
	matches, unmatched_a, unmatched_b = [], [], []
	cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
	for ix, mx in enumerate(x):
		if mx >= 0:
			matches.append([ix, mx])
	unmatched_a = np.where(x < 0)[0]
	unmatched_b = np.where(y < 0)[0]
	matches = np.asarray(matches)
	return matches, unmatched_a, unmatched_b


def tlbr_expand(tlbr, scale=1.2):
	w = tlbr[2] - tlbr[0]
	h = tlbr[3] - tlbr[1]

	half_scale = 0.5 * scale

	tlbr[0] -= half_scale * w
	tlbr[1] -= half_scale * h
	tlbr[2] += half_scale * w
	tlbr[3] += half_scale * h

	return tlbr


IOU_METRICS = ("iou", "giou", "diou", "ciou")


def iou_family(atlbrs, btlbrs, metric="iou"):
	"""
	Compute an IoU-family overlap (similarity) matrix between two sets of boxes.

	Supports plain IoU plus the Generalized (giou), Distance (diou) and
	Complete (ciou) variants, each of which adds a geometric penalty term to
	the raw IoU. Plain IoU lies in [0, 1]; the variants lie in [-1, 1].

	:type atlbrs: list[tlbr] | np.ndarray, (N, 4) of [x1, y1, x2, y2]
	:type btlbrs: list[tlbr] | np.ndarray, (M, 4) of [x1, y1, x2, y2]
	:param metric: one of IOU_METRICS
	:rtype overlaps np.ndarray, (N, M)
	"""
	if len(atlbrs) == 0 or len(btlbrs) == 0:
		return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)

	if metric == "iou":
		# Fast (optionally cython) path; the variants need the geometry below.
		return bbox_ious(
			np.ascontiguousarray(atlbrs, dtype=np.float64),
			np.ascontiguousarray(btlbrs, dtype=np.float64),
		)

	a = np.ascontiguousarray(atlbrs, dtype=np.float64).reshape(-1, 4)
	b = np.ascontiguousarray(btlbrs, dtype=np.float64).reshape(-1, 4)

	# Broadcast to all pairs: a -> (N, 1), b -> (1, M).
	ax1, ay1, ax2, ay2 = (a[:, None, k] for k in range(4))
	bx1, by1, bx2, by2 = (b[None, :, k] for k in range(4))

	area_a = (ax2 - ax1) * (ay2 - ay1)
	area_b = (bx2 - bx1) * (by2 - by1)

	inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0.0, None)
	inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0.0, None)
	inter = inter_w * inter_h
	union = area_a + area_b - inter
	iou = inter / np.maximum(union, 1e-9)

	# Smallest box enclosing each pair, shared by the giou/diou/ciou penalties.
	enc_x1 = np.minimum(ax1, bx1)
	enc_y1 = np.minimum(ay1, by1)
	enc_x2 = np.maximum(ax2, bx2)
	enc_y2 = np.maximum(ay2, by2)
	if metric == "giou":
		enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
		return iou - (enc_area - union) / np.maximum(enc_area, 1e-9)

	# diou/ciou penalise the squared centre distance over the enclosing diagonal.
	center_dist_sq = ((ax1 + ax2 - bx1 - bx2) * 0.5) ** 2 \
		+ ((ay1 + ay2 - by1 - by2) * 0.5) ** 2
	enc_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
	diou = iou - center_dist_sq / np.maximum(enc_diag_sq, 1e-9)
	if metric == "diou":
		return diou

	# ciou adds an aspect-ratio consistency term on top of diou.
	a_w = np.maximum(ax2 - ax1, 1e-9)
	a_h = np.maximum(ay2 - ay1, 1e-9)
	b_w = np.maximum(bx2 - bx1, 1e-9)
	b_h = np.maximum(by2 - by1, 1e-9)
	v = (4.0 / np.pi ** 2) * (np.arctan(a_w / a_h) - np.arctan(b_w / b_h)) ** 2
	alpha = v / np.maximum(1.0 - iou + v, 1e-9)
	return diou - alpha * v


def iou_distance(atracks: list, btracks: list, metric: str = "iou"):
	"""
	Compute matching cost based on an IoU-family overlap metric.

	:type atracks: list[STrack] | np.ndarray
	:type btracks: list[STrack] | np.ndarray
	:param metric: overlap measure, one of IOU_METRICS ("iou", "giou",
		"diou", "ciou"). Cost is 1 - overlap, so a perfect match is 0.

	:rtype cost_matrix np.ndarray
	"""
	if metric not in IOU_METRICS:
		raise ValueError(f"Unsupported metric '{metric}', expected one of {IOU_METRICS}")

	if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
		atlbrs = atracks
		btlbrs = btracks
	else:
		atlbrs = [track.tlbr for track in atracks]
		btlbrs = [track.tlbr for track in btracks]

	_ious = iou_family(atlbrs, btlbrs, metric)
	cost_matrix = 1 - _ious

	return cost_matrix


def v_iou_distance(atracks, btracks):
	"""
	Compute cost based on IoU
	:type atracks: list[STrack]
	:type btracks: list[STrack]

	:rtype cost_matrix np.ndarray
	"""

	if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
		atlbrs = atracks
		btlbrs = btracks
	else:
		atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
		btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
	_ious = iou_family(atlbrs, btlbrs)
	cost_matrix = 1 - _ious

	return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
	"""
	:param tracks: list[STrack]
	:param detections: list[BaseTrack]
	:param metric:
	:return: cost_matrix np.ndarray
	"""

	cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
	if cost_matrix.size == 0:
		return cost_matrix
	det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
	track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)

	cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
	return cost_matrix


def cosine_dist(a, b):
	a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
	b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
	return 1.0 - np.matmul(a_norm, b_norm.T)  # shape: (N, M)


def embedding_distance_kpr(tracks, detections, metric='cosine'):
	"""
	:param tracks: list[STrack]
	:param detections: list[BaseTrack]
	:param metric:
	:return: cost_matrix np.ndarray
	"""

	cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
	if cost_matrix.size == 0:
		return cost_matrix

	det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
	# track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
	track_features = np.asarray([track.curr_feat for track in tracks], dtype=np.float64)

	det_viss = np.asarray([track.curr_viss for track in detections], dtype=np.float64)
	# track_viss = np.asarray([track.smooth_viss for track in tracks], dtype=np.float64)
	track_viss = np.asarray([track.curr_viss for track in tracks], dtype=np.float64)

	body_part_dists = [] # for 6 body parts
	for p in range(6):
		part_dist = cosine_dist(track_features[:, p, :], det_features[:, p, :])  # shape: (N, M)
		body_part_dists.append(part_dist)

	body_part_dists = np.stack(body_part_dists, axis=0)  # shape: (6, N, M)

	# Validity mask: shape (6, N, M)
	valid_mask = np.sqrt(track_viss.T[:, :, None] * det_viss.T[:, None, :])

	masked_dists = body_part_dists * valid_mask
	mean_dists = np.sum(masked_dists, axis=0) / (np.sum(valid_mask, axis=0) + 1e-8)

	return mean_dists


def centroid_distance(tracks, detections, metric='euclidean'): 
	"""
	:param atracks: list[Locations]
	:param btracks: list[Locations]
	:param metric:
	:return: cost_matrix np.ndarray
	"""

	cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
	if cost_matrix.size == 0:
		return cost_matrix
	det_centroids = np.asarray([track.centroid for track in detections], dtype=np.float)
	track_centroids = np.asarray([track.centroid for track in tracks], dtype=np.float)
	cost_matrix = cdist(track_centroids, det_centroids, metric)
	return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
	if cost_matrix.size == 0:
		return cost_matrix
	gating_dim = 2 if only_position else 4
	gating_threshold = kalman_filter.chi2inv95[gating_dim]
	# measurements = np.asarray([det.to_xyah() for det in detections])
	measurements = np.asarray([det.to_xywh() for det in detections])
	for row, track in enumerate(tracks):
		gating_distance = kf.gating_distance(
			track.mean, track.covariance, measurements, only_position)
		cost_matrix[row, gating_distance > gating_threshold] = np.inf
	return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
	if cost_matrix.size == 0:
		return cost_matrix
	gating_dim = 2 if only_position else 4
	gating_threshold = kalman_filter.chi2inv95[gating_dim]
	# measurements = np.asarray([det.to_xyah() for det in detections])
	measurements = np.asarray([det.to_xywh() for det in detections])
	for row, track in enumerate(tracks):
		gating_distance = kf.gating_distance(
			track.mean, track.covariance, measurements, only_position, metric='maha')
		cost_matrix[row, gating_distance > gating_threshold] = np.inf
		cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
	return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
	if cost_matrix.size == 0:
		return cost_matrix
	reid_sim = 1 - cost_matrix
	iou_dist = iou_distance(tracks, detections)
	iou_sim = 1 - iou_dist
	fuse_sim = reid_sim * (1 + iou_sim) / 2
	det_scores = np.array([det.score for det in detections])
	det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
	#fuse_sim = fuse_sim * (1 + det_scores) / 2
	fuse_cost = 1 - fuse_sim
	return fuse_cost


def fuse_score(cost_matrix, detections):
	if cost_matrix.size == 0:
		return cost_matrix
	iou_sim = 1 - cost_matrix
	det_scores = np.array([det.score for det in detections])
	det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
	fuse_sim = iou_sim * det_scores
	fuse_cost = 1 - fuse_sim
	return fuse_cost
