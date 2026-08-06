import math

import numpy as np

__all__ = [
	"part_distance", 
	"gallery_distance"
]

_EPS = 1e-12

def part_distance(emb_a, emb_b, vis_a, vis_b) -> float:
	"""Visibility-weighted mean per-part cosine distance between two persons.

	For each part p: ``d_p = 1 - cos(a_p, b_p)`` weighted by
	``w_p = vis_a_p * vis_b_p``; returns ``sum(w_p * d_p) / sum(w_p)``.
	Returns ``inf`` when the total overlapping visibility weight is ~0
	(no commonly-visible part), or when either embedding is missing.
	"""
	if emb_a is None or emb_b is None or vis_a is None or vis_b is None:
		return math.inf

	a  = np.asarray(emb_a, dtype=np.float64)
	b  = np.asarray(emb_b, dtype=np.float64)
	va = np.asarray(vis_a, dtype=np.float64).reshape(-1)
	vb = np.asarray(vis_b, dtype=np.float64).reshape(-1)

	na = np.linalg.norm(a, axis=1)
	nb = np.linalg.norm(b, axis=1)
	norm_ok = (na > _EPS) & (nb > _EPS)

	denom = np.where(norm_ok, na * nb, 1.0)
	cos   = np.where(norm_ok, np.sum(a * b, axis=1) / denom, 0.0)
	d_part = 1.0 - cos

	w = va * vb
	w = np.where(norm_ok, w, 0.0)
	total = float(w.sum())
	if total < _EPS:
		return math.inf
	return float(np.sum(w * d_part) / total)


def gallery_distance(emb, vis, gallery: list, reduce: str = "min") -> float:
	"""Distance from ``(emb, vis)`` to a track's gallery of ``(embedding, visibility)``.

	Reduced by ``min`` (default) or ``mean`` over the finite per-entry distances.
	Returns ``inf`` for an empty gallery or one with no commonly-visible parts.
	"""
	dists = [part_distance(emb, g_emb, vis, g_vis) for g_emb, g_vis in gallery]
	finite = [d for d in dists if math.isfinite(d)]
	if not finite:
		return math.inf
	if reduce == "mean":
		return float(sum(finite) / len(finite))
	return float(min(finite))
