import collections
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment

from core.bev_fusion.lift3d import angle_with_polyfit_degree_2
from core.bev_fusion.reid import gallery_distance, part_distance

__all__ = ["BEVFusionTrackerNOR"]

_EPS = 1e-12


@dataclass
class WorldObservation:
	"""A cross-camera group of observations of one object in a single frame."""

	class_id   : int
	world_xy   : np.ndarray # (2,) mean over member cameras
	members    : list # list[CamObservation], <=1 per camera
	embedding  : Optional[np.ndarray] = None # (6, 512) visibility-weighted fuse | None
	visibility : Optional[np.ndarray] = None # (6,) | None


class GlobalTrack:
	"""A persistent cross-camera track with a single (scene, class) object_id."""

	def __init__(self, object_id, class_id, world_xy, gallery_size):
		"""Initialize a fresh track at its first sighting."""
		self.object_id     = object_id
		self.class_id      = class_id
		self.world_xy      = np.asarray(world_xy, dtype=np.float64).copy()
		self.velocity      = np.zeros(2, dtype=np.float64)
		self.members       = {} # {cam_idx: local_track_id}, last sighting
		self.gallery       = collections.deque(maxlen=gallery_size)
		self.last_frame    = -1
		self.age_missing   = 0
		self.active        = True
		self.history       = [] # list[(frame_id, x, y)] -- real sightings
		self.frames        = {} # {frame_id: (x, y)} -- filled by finalize
		self.height_by_frame = {} # {frame_id: median of that frame's accepted samples}
		self.rotations     = {} # {frame_id: {'pitch': pitch_radian, 'roll': roll_radian, 'yaw': yaw_radian}}
		self.last_rotation = {'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0} # yaw carried to frames lacking a fitted heading

	def __repr__(self):
		return f"GlobalTrack(object_id={self.object_id}, class_id={self.class_id}, world_xy={self.world_xy}, velocity={self.velocity}, members={self.members}, last_frame={self.last_frame}, age_missing={self.age_missing}, active={self.active})"

	def find_yaw(self, half_window=45, min_speed=1e-6):
		"""Fill ``self.rotations`` with a per-frame rotation from the trajectory."""
		ordered = sorted(self.frames.items())  # [(frame_id, (x, y)), ...] in time order
		positions = [xy for _, xy in ordered]
		for i, (frame_id, _) in enumerate(ordered):
			frame_start = max(0, i - half_window)
			# frame_end   = min(len(positions), i + half_window + 1)
			frame_end   = min(len(positions), i) # only use past positions to avoid future leakage
			window = positions[frame_start:frame_end]
			yaw = angle_with_polyfit_degree_2(window, eval_index=i - frame_start, min_speed=min_speed)
			if yaw is None:
				yaw = self.last_rotation['yaw']  # inherit the previous frame_id's yaw
			# Merge into any existing rotation for this frame: keep pitch/roll when
			# already set (e.g. by another pass), default them only when absent.
			entry = self.rotations.setdefault(frame_id, {})
			entry.setdefault('pitch', 0.0)
			entry.setdefault('roll', 0.0)
			entry['yaw'] = yaw
			self.last_rotation['yaw'] = yaw  # carry-forward source for the next frame
		return self.rotations


def reduce_height_samples(tr, wo, frame_id):
	"""Median-reduce this frame's accepted member height samples onto ``tr``."""
	heights = [m.height_sample for m in wo.members if m.height_sample is not None]
	if heights:
		tr.height_by_frame[frame_id] = float(np.median(heights))


class BEVFusionTrackerNOR:

	def __init__(self, class_labels, reid_class_ids, scene_id: int, fps: int, cfg: dict):
		"""Configure the tracker from a scene's class labels and MTMC config."""
		self.scene_id       = scene_id
		self.fps            = max(1, int(fps))
		self.cfg            = cfg
		self.class_labels   = class_labels
		self.reid_class_ids = reid_class_ids

		self.bev_group_gate      = cfg["bev_group_gate"]
		self.reid_group_thresh   = cfg["reid_group_thresh"]
		self.bev_thresh_by_class = list(cfg["bev_dist_thresh_by_class"])
		self.w_bev               = cfg["w_bev"]
		self.reid_track_thresh   = cfg["reid_track_thresh"]
		self.max_speed_m_s       = cfg["max_speed_m_s"]
		self.new_track_min_sep   = cfg["new_track_min_sep"]
		self.max_age             = cfg["max_age"]
		self.alpha_pos           = cfg["alpha_pos"]
		self.gallery_size        = cfg["gallery_size"]
		self.make_world_observation_mode = cfg.get("make_world_observation_mode", "max_confidence")
		self.interpolating_missing_frames_max_gap = cfg.get("interpolating_missing_frames_max_gap", 0)

		self.reclaim_ghost_min_age = cfg.get("reclaim_ghost_min_age", self.max_age)
		self.force_match_track = cfg.get("force_match_track", False)
		
		self.tracks   = []  # active + coasting
		self.retired  = []  # age_missing > max_age, kept for output (adaptive only)
		self._next_id = collections.defaultdict(lambda: 1)   # per-class object_id counter

	# ------------------------------------------------------------------ #
	# helpers
	# ------------------------------------------------------------------ #
	def _bev_thresh_for(self, class_id):
		"""Return the per-class BEV association distance gate, in meters."""
		if 0 <= class_id < len(self.bev_thresh_by_class):
			return self.bev_thresh_by_class[class_id]
		return 2.0

	def _is_person(self, class_id):
		"""Return whether ``class_id`` uses the appearance/ReID path (Person or FourierGR1T2)."""
		return class_id in self.reid_class_ids

	def _cap_for(self, class_id):
		return None

	@staticmethod
	def _spawn_rank(wo):
		"""Rank an unmatched observation for scarce-slot selection."""
		confs = [m.confidence for m in wo.members if m.confidence is not None]
		return (max(confs) if confs else 0.0, len(wo.members))

	def _stalest_ghost(self, class_id, existing, matched_tracks):
		"""Return the longest-missing reclaimable track of ``class_id``, or None."""
		ghosts = [tr for tr in existing
				  if tr.class_id == class_id
				  and tr not in matched_tracks
				  and tr.age_missing >= self.reclaim_ghost_min_age]
		return max(ghosts, key=lambda tr: tr.age_missing) if ghosts else None

	# ------------------------------------------------------------------ #
	# Step 4 -- per-frame cross-camera grouping
	# ------------------------------------------------------------------ #
	def group_observations(self, cam_observations: list) -> list:
		"""Fuse per-camera observations of one frame into world observations."""
		by_class = collections.defaultdict(list)
		for o in cam_observations:
			by_class[o.class_id].append(o)

		world_obs = []
		for class_id, items in by_class.items():
			for cluster in self._cluster(items, class_id):
				world_obs.append(self._make_world_observation(class_id, cluster, self.make_world_observation_mode))
		return world_obs

	def _cluster(self, items, class_id):
		"""Greedy single-linkage with a <=1-per-camera constraint."""
		num_object = len(items)
		if num_object == 0:
			return []
		# Persons use the wider grouping gate (ReID disambiguates within it);
		# other classes rely on the tighter per-class BEV proximity gate alone.
		gate = self._bev_thresh_for(class_id)

		# ---- collect candidate merge edges (i, j) that pass every gate ----
		# Build the upper-triangular set of observation pairs; a pair becomes a
		# candidate edge only if it clears the BEV gate (and, for persons, ReID).
		pairs = []
		for i in range(num_object):
			for j in range(i + 1, num_object):

				# Two observations from the same camera can never be the same
				# physical object in one frame, so they are never merge candidates.
				if items[i].cam_idx == items[j].cam_idx:
					continue

				# BEV gate: ground-plane separation must be under `gate` meters.
				d = float(np.linalg.norm(items[i].world_xy - items[j].world_xy))
				if d >= gate:
					continue

				# Person ReID gate: appearance must also agree, so two distinct
				# people who happen to stand close on the ground are not merged.
				if self._is_person(class_id):
					rd = part_distance(items[i].embedding , items[j].embedding,
									   items[i].visibility, items[j].visibility)
					if (rd >= self.reid_group_thresh):
						continue

				pairs.append((d, i, j))
		# Greedy single-linkage: consider the spatially closest links first so the
		# most confident merge wins whenever two clusters contend for a camera.
		pairs.sort(key=lambda p: p[0])

		# ---- union-find over observations, tracking each cluster's cameras ----
		# `parent` is the disjoint-set forest; `cams[root]` is the set of camera
		# indices already absorbed into that cluster (used for the conflict check).
		parent = list(range(num_object))
		cams   = [{items[i].cam_idx} for i in range(num_object)]

		def find(x):
			"""Return the union-find root of ``x``, compressing the path."""
			while parent[x] != x:
				parent[x] = parent[parent[x]]
				x = parent[x]
			return x

		# Apply edges in ascending-distance order, merging the two clusters unless
		# they are already joined or would share a camera (the <=1-per-camera
		# constraint: a cluster may hold at most one observation per camera).
		for _, i, j in pairs:
			ri, rj = find(i), find(j)
			if ri == rj or (cams[ri] & cams[rj]):
				continue
			parent[rj] = ri
			cams[ri]  |= cams[rj]

		# Collect observations by their final root into the output clusters.
		clusters = collections.defaultdict(list)
		for i in range(num_object):
			clusters[find(i)].append(items[i])
		return list(clusters.values())

	def _make_world_observation(self, class_id, cluster, mode="max_confidence"):
		"""Collapse one cluster into a single :class:`WorldObservation`."""
		# merge the member positions into one world position;
		# for persons, fuse the ReID embeddings into one visibility-weighted descriptor.
		if mode not in ["max_confidence", "mean", "weighted_mean"]:
			logger.warning(f"Unknown mode '{mode}' for _make_world_observation; falling back to 'max_confidence'.")
			mode = "max_confidence"

		if mode == "max_confidence":
			index_of_max_confidence = np.argmax([m.confidence if m.confidence is not None else -np.inf for m in cluster])
			world_xy = cluster[index_of_max_confidence].world_xy
		elif mode == "mean":
			world_xy = np.mean([m.world_xy for m in cluster], axis=0)
		elif mode == "weighted_mean":
			weights = np.array([m.confidence if m.confidence is not None else 0.0 for m in cluster])
			weight_sum = np.sum(weights)
			if weight_sum == 0:
				world_xy = np.mean([m.world_xy for m in cluster], axis=0)
			else:
				weights = weights / weight_sum
				world_xy = np.average([m.world_xy for m in cluster], axis=0, weights=weights)

		emb, vis = (None, None)
		if self._is_person(class_id):
			emb, vis = self._fuse_embeddings(cluster)
		return WorldObservation(class_id=class_id, world_xy=world_xy,
								members=list(cluster), embedding=emb, visibility=vis)

	@staticmethod
	def _fuse_embeddings(cluster):
		"""Fuse member ReID descriptors into one visibility-weighted embedding."""
		embs, viss = [], []
		for m in cluster:
			if m.embedding is not None and m.visibility is not None:
				embs.append(np.asarray(m.embedding, dtype=np.float64))
				viss.append(np.asarray(m.visibility, dtype=np.float64).reshape(-1))
		if not embs:
			return None, None
		E = np.stack(embs)                      # (M, 6, D)
		V = np.stack(viss)                      # (M, 6)
		W = V[:, :, None]
		num = (W * E).sum(axis=0)               # (6, D)
		den = W.sum(axis=0)                      # (6, 1)
		fused = np.where(den > _EPS, num / np.where(den > _EPS, den, 1.0), E.mean(axis=0))
		norms = np.linalg.norm(fused, axis=1, keepdims=True)
		fused = np.where(norms > _EPS, fused / np.where(norms > _EPS, norms, 1.0), fused)
		fused_vis = V.max(axis=0)
		return fused.astype(np.float32), fused_vis.astype(np.float32)

	# ------------------------------------------------------------------ #
	# Step 5 -- frame-to-track association + global ID assignment
	# ------------------------------------------------------------------ #
	def update(self, world_obs: list, frame_id: int) -> None:
		# Live tracks carried in from previous frames. `matched_*` accumulate
		# across both passes so each observation and each track is consumed once.
		existing             = self.tracks
		matched_observations = set()
		matched_tracks       = set()

		# ---- Pass A: carry-forward by single-camera local track id ----
		# Build a reverse lookup from every single-camera member a track owns,
		# keyed by (camera, local track id), back to the owning global track.
		member_index = {}
		for tr in existing:
			for cam, lid in tr.members.items():
				member_index[(cam, lid)] = tr

		# For each observation, count how many of its single-camera members are
		# already owned by each existing global track. A higher count means a
		# stronger carry-forward link. Emit one candidate per (obs, track) pair.
		candidates                = [] # (count, obs_idx, track)
		observations_track_counts = [] # per obs: {track: count}
		for o_index, world_observation in enumerate(world_obs):
			counts = {}
			for member in world_observation.members:
				tr = member_index.get((member.cam_idx, member.local_track_id))
				# Only carry forward when the member's owner is the same class.
				if tr is not None and tr.class_id == world_observation.class_id:
					counts[tr] = counts.get(tr, 0) + 1
			observations_track_counts.append(counts)
			for tr, c in counts.items():
				candidates.append((c, o_index, tr))

		# Greedily bind the strongest links first (most shared members). The
		# `matched_*` guards make this a one-to-one greedy assignment.
		candidates.sort(key=lambda x: -x[0])
		for c, o_index, tr in candidates:
			if o_index in matched_observations or tr in matched_tracks:
				continue

			self._update_track(tr, world_obs[o_index], frame_id)
			matched_observations.add(o_index)
			matched_tracks.add(tr)

		# ---- Pass B: gated BEV(+ReID) assignment for the remainder ----
		# Whatever Pass A could not link (no shared member id) is resolved here.
		r_obs    = [oi for oi in range(len(world_obs)) if oi not in matched_observations]
		r_tracks = [tr for tr in existing if tr not in matched_tracks]
		# Associate per class so the cost matrix never mixes classes; each class's
		# leftovers go through a gated BEV(+ReID) Hungarian assignment.
		for class_id in {world_obs[oi].class_id for oi in r_obs}:
			obs_c = [oi for oi in r_obs if world_obs[oi].class_id == class_id]
			trk_c = [tr for tr in r_tracks if tr.class_id == class_id]
			if not obs_c or not trk_c:
				continue

			matched_pairs = self._associate(world_obs, obs_c, trk_c, class_id)

			for o_index, tr in matched_pairs:
				self._update_track(tr, world_obs[o_index], frame_id)
				matched_observations.add(o_index)
				matched_tracks.add(tr)

		# ---- spawn new tracks for still-unmatched observations (fixed-N cap) ----
		snapshot         = list(existing)
		new_tracks       = []
		live_by_class    = collections.Counter(tr.class_id for tr in snapshot)
		spawned_by_class = collections.Counter()

		unmatched = [oi for oi in range(len(world_obs)) if oi not in matched_observations]

		for o_index in unmatched:
			world_observation = world_obs[o_index]
			class_id          = world_observation.class_id

			# Fixed-N cap: never create track N+1 for a class (cap 0 = not tracked).
			cap = self._cap_for(class_id)
			if cap is not None and live_by_class[class_id] + spawned_by_class[class_id] >= cap:
				# At cap: hand this detection to the stalest coasting ghost slot of
				# the same class (re-using its object_id) rather than dropping it;
				# otherwise the detection is lost (logged for visibility).
				ghost = self._stalest_ghost(class_id, existing, matched_tracks)
				if ghost is not None:
					self._update_track(ghost, world_observation, frame_id)
					matched_tracks.add(ghost)
				else:
					pass
				continue

			# Suppress respawns near an existing same-class track (anti-respawn).
			too_close = any(
				tr.class_id == class_id
				and np.linalg.norm(world_observation.world_xy - tr.world_xy) < self.new_track_min_sep
				for tr in snapshot
			)
			if too_close:
				continue

			new_tracks.append(self._spawn_track(world_observation, frame_id))
			spawned_by_class[class_id] += 1

		# ---- coast unmatched tracks; fixed-count classes never retire ----
		survivors = []
		for tr in existing:
			# Matched tracks were already advanced inside `_update_track`.
			if tr in matched_tracks:
				survivors.append(tr)
				continue
			# Unmatched track: dead-reckon by velocity and age it.
			tr.world_xy = tr.world_xy + tr.velocity
			tr.age_missing += 1
			# Adaptive classes retire past max_age as before; fixed-count classes
			# (cap is not None) keep their slot alive indefinitely (coast).
			if self._cap_for(tr.class_id) is None and tr.age_missing > self.max_age:
				tr.active = False
				self.retired.append(tr)
			else:
				survivors.append(tr)
		# Surviving (matched + still-coasting) tracks plus this frame's spawns
		# become the live set carried into the next frame.
		self.tracks = survivors + new_tracks

	INF = 1e6

	def _associate(self, world_obs, obs_c, trk_c, class_id):
		"""Gated BEV(+ReID) Hungarian assignment for one class's leftovers."""
		base = self._bev_thresh_for(class_id)
		is_person = self._is_person(class_id)
		cost = np.full((len(obs_c), len(trk_c)), self.INF, dtype=np.float64)
		for a, oi in enumerate(obs_c):
			wo = world_obs[oi]
			for b, tr in enumerate(trk_c):
				# `world_xy` was already advanced by `velocity` on each coasted
				# frame, so it reflects the position as of the previous frame; one
				# more velocity step predicts the current frame. (Adding
				# velocity*age_missing here would double-count the coast.)
				pred = tr.world_xy + tr.velocity
				d = float(np.linalg.norm(wo.world_xy - pred))

				gate = base + (self.max_speed_m_s / self.fps) * tr.age_missing

				# Force mode skips the distance/ReID gates so every track is matched
				# to its best available observation (Hungarian stays optimal).
				if not self.force_match_track and d > gate:
					continue

				if is_person:
					rd = (gallery_distance(wo.embedding, wo.visibility, list(tr.gallery), reduce="min")
						  if wo.embedding is not None else math.inf)
					if not self.force_match_track and math.isfinite(rd) and rd > self.reid_track_thresh:
						continue
					rd_eff = rd if math.isfinite(rd) else 1.0
					cost[a, b] = self.w_bev * (d / gate) + (1.0 - self.w_bev) * rd_eff
				else:
					cost[a, b] = d

		rows, cols = linear_sum_assignment(cost)

		return [(obs_c[a], trk_c[b]) for a, b in zip(rows, cols) if cost[a, b] < self.INF]

	def _update_track(self, tr, wo, frame_id):
		"""Fold a matched observation into a track."""
		obs_xy      = np.asarray(wo.world_xy, dtype=np.float64)
		new_pos     = self.alpha_pos * obs_xy + (1.0 - self.alpha_pos) * tr.world_xy
		elapsed     = max(1, frame_id - tr.last_frame) if tr.last_frame >= 0 else 1
		tr.velocity = (new_pos - tr.world_xy) / elapsed
		tr.world_xy = new_pos
		tr.members  = {m.cam_idx: m.local_track_id for m in wo.members}
		if wo.embedding is not None and wo.visibility is not None:
			tr.gallery.append((np.asarray(wo.embedding), np.asarray(wo.visibility)))
		tr.age_missing = 0
		tr.last_frame  = frame_id
		tr.history.append((frame_id, float(new_pos[0]), float(new_pos[1])))
		reduce_height_samples(tr, wo, frame_id)

	def _spawn_track(self, wo, frame_id):
		"""Create a new global track from an unmatched observation."""
		object_id = self._next_id[wo.class_id]
		self._next_id[wo.class_id] += 1
		tr = GlobalTrack(object_id, wo.class_id, wo.world_xy, self.gallery_size)
		tr.members = {m.cam_idx: m.local_track_id for m in wo.members}
		if wo.embedding is not None and wo.visibility is not None:
			tr.gallery.append((np.asarray(wo.embedding), np.asarray(wo.visibility)))
		tr.last_frame  = frame_id
		tr.age_missing = 0
		tr.history.append((frame_id, float(wo.world_xy[0]), float(wo.world_xy[1])))
		reduce_height_samples(tr, wo, frame_id)
		return tr

	# ------------------------------------------------------------------ #
	# Step 6 -- temporal smoothing / gap interpolation + finalize
	# ------------------------------------------------------------------ #
	def finalize(self) -> dict:
		"""Materialize per-frame positions for every track and group by class."""
		result = collections.defaultdict(list)
		fps = self.fps if self.fps > 0 else 45
		for tr in self.tracks + self.retired:
			self._build_frames(tr)
			tr.find_yaw(half_window=max(2, round(fps * 1.5)))
			result[tr.class_id].append(tr)
		return dict(result)

	def _build_frames(self, tr):
		"""Fill ``tr.frames`` from sighting history, interpolating short gaps."""
		hist   = sorted(tr.history)
		frames = {f: (x, y) for f, x, y in hist}
		# for k in range(len(hist) - 1):
		# 	f_0, x_0, y_0 = hist[k]
		# 	f_1, x_1, y_1 = hist[k + 1]
		# 	gap = f_1 - f_0
		# 	if gap <= 1 or (gap - 1) > self.interpolating_missing_frames_max_gap:
		# 		continue
		# 	for f in range(f_0 + 1, f_1):
		# 		a = (f - f_0) / gap
		# 		frames[f] = (x_0 + a * (x_1 - x_0), y_0 + a * (y_1 - y_0))
		tr.frames = frames
