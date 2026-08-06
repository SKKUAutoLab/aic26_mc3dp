import collections
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment

from core.bev_fusion.reid import gallery_distance, part_distance
from core.bev_fusion.lift3d import angle_with_polyfit_degree_2

__all__ = ["CamObservation", "WorldObservation", "GlobalTrack", "BEVFusionTracker"]

_EPS = 1e-12


@dataclass
class CamObservation:
	"""A single-camera detection projected onto the world ground plane.

	One per (camera, local track) sighting in a frame. Person observations
	additionally carry a part-based ReID embedding and per-part visibility;
	all other classes leave those ``None``.

	Attributes:
		cam_idx: Index of the source camera within the scene's camera order.
		class_id: Object class id (Person, Forklift, ...).
		local_track_id: Single-camera tracker id for this object on this camera.
		world_xy: Ground-plane position ``(2,)`` in meters on the Z=0 plane.
		bbox_xywhn: Normalized track bbox ``(4,)`` as ``cx, cy, w, h``, or None.
		embedding: Part-based ReID features ``(6, 512)`` for persons, else None.
		visibility: Per-part visibility weights ``(6,)`` for persons, else None.
		height_sample: Accepted per-detection height estimate in meters for
			persons (see ``height_estimation.PersonHeightSampler``), else None.
		ground_pixel: Ground-contact pixel selected for this observation. Optional
			metadata consumed by the final-stage localization refiner only.
		pose_keypoints: Original COCO-17 pose records, retained so the refiner can
			apply the conservative no/one/edge-ankle fallback online.
		image_size: Source image ``(height, width)`` used for edge checks.
	"""

	cam_idx        : int
	class_id       : int
	local_track_id : int
	world_xy       : np.ndarray  # (2,) meters, Z=0 plane
	confidence     : Optional[float] = None # confidence score of the observation
	bbox_xywhn     : Optional[np.ndarray] = None # (4,) normalized cx, cy, w, h
	embedding      : Optional[np.ndarray] = None # (6, 512) for person, else None
	visibility     : Optional[np.ndarray] = None # (6,) for person, else None
	height_sample  : Optional[float] = None # meters, person height estimate | None
	ground_pixel   : Optional[np.ndarray] = None # (2,) source pixel | None
	pose_keypoints : Optional[list] = None # COCO-17 pose records | None
	image_size     : Optional[tuple] = None # (height, width) | None


@dataclass
class WorldObservation:
	"""A cross-camera group of observations of one object in a single frame.

	Produced by grouping ``CamObservation`` instances that fall onto the same
	world location (and, for persons, are ReID-consistent). Holds at most one
	member per camera. For persons the member embeddings are fused into a single
	visibility-weighted descriptor.

	Attributes:
		class_id: Object class id shared by all members.
		world_xy: Mean ground position ``(2,)`` over member cameras, in meters.
		members: The contributing ``CamObservation`` list (<=1 per camera).
		embedding: Visibility-weighted fused ReID features ``(6, 512)``, or None.
		visibility: Per-part visibility ``(6,)`` of the fused descriptor, or None.
	"""

	class_id   : int
	world_xy   : np.ndarray # (2,) mean over member cameras
	members    : list # list[CamObservation], <=1 per camera
	embedding  : Optional[np.ndarray] = None # (6, 512) visibility-weighted fuse | None
	visibility : Optional[np.ndarray] = None # (6,) | None


class GlobalTrack:
	"""A persistent cross-camera track with a single (scene, class) object_id."""

	def __init__(self, object_id, class_id, world_xy, gallery_size):
		"""Initialize a fresh track at its first sighting.

		Seeds the track's identity and ground position from the first
		observation and sets up the empty state containers that later frames
		fill in: the per-camera member map, the bounded ReID gallery, the raw
		sighting history, and the finalized per-frame position/rotation maps.
		All temporal fields start empty and are populated incrementally as the
		track is matched across frames and then by :meth:`finalize`.

		Args:
			object_id: Per-(scene, class) global id assigned to this track.
			class_id: Object class id this track represents.
			world_xy: Initial ground position ``(2,)`` in meters.
			gallery_size: Max number of recent (embedding, visibility) pairs to
				retain for ReID matching (bounds the gallery deque).

		Attributes:
			object_id: Per-(scene, class) global id assigned to this track.
			class_id: Object class id this track represents.
			world_xy (np.ndarray): Current ground position ``(2,)`` in meters
				(float64); a private copy of the seed ``world_xy``.
			velocity (np.ndarray): Estimated ground velocity ``(2,)`` in
				meters/frame (float64), initialized to zero.
			members (dict): ``{cam_idx: local_track_id}`` for the most recent
				sighting, identifying which per-camera track fed this global track.
			gallery (collections.deque): Bounded (maxlen ``gallery_size``) deque
				of recent ``(embedding, visibility)`` pairs used for ReID matching.
			last_frame (int): Frame index of the last sighting; ``-1`` until the
				track is first updated.
			age_missing (int): Number of consecutive frames the track has gone
				unmatched; used to retire stale tracks.
			active (bool): Whether the track is still live (``True``) or retired.
			history (list): List of ``(frame_id, x, y)`` tuples for real
				sightings only (no interpolation).
			frames (dict): ``{frame_id: (x, y)}`` of materialized per-frame
				positions, populated by :meth:`finalize` (with short gaps
				interpolated).
			height_by_frame (dict): ``{frame_id: median height (m)}`` of that
				frame's accepted member height samples -- current frame only,
				no carry-forward; frames without an entry fall back to the
				class-constant height at write time.
			rotations (dict): ``{frame_id: {'pitch', 'roll', 'yaw'}}`` per-frame
				rotation in radians, populated by :meth:`find_yaw`.
			last_rotation (dict): ``{'pitch', 'roll', 'yaw'}`` carried forward to
				frames lacking a fitted heading; all default to ``0.0``.
		"""
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
		"""Fill ``self.rotations`` with a per-frame rotation from the trajectory.

		Walks the finalized per-frame positions in time order and, for each
		frame, fits a centered window of +/-``half_window`` frames with
		:func:`angle_with_polyfit_degree_2` evaluated at that frame. Frames whose
		window is too short or (near-)stationary inherit ``self.last_rotation['yaw']``
		(the previous frame_id's yaw), or ``0.0`` before any valid heading exists.
		Existing ``pitch``/``roll`` for a frame are kept; absent ones default to
		``0.0`` (ground objects). Must run after :meth:`finalize` has populated
		``self.frames``.

		Args:
			half_window: Frames on each side of a frame included in its fit window
				(window length up to ``2*half_window + 1``).
			min_speed: Forwarded to :func:`angle_with_polyfit_degree_2`; fitted
				speeds below this leave that frame's heading undefined.

		Returns:
			The populated ``self.rotations`` dict
			``{frame_id: {'pitch', 'roll', 'yaw'}}`` with yaw in radians.
		"""
		ordered = sorted(self.frames.items())  # [(frame_id, (x, y)), ...] in time order
		positions = [xy for _, xy in ordered]
		for i, (frame_id, _) in enumerate(ordered):
			frame_start = max(0, i - half_window)
			frame_end   = min(len(positions), i + half_window + 1)
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
	"""Median-reduce this frame's accepted member height samples onto ``tr``.

	Current frame only, by design: the samples carried by ``wo.members`` (each
	camera's accepted person height estimate for THIS frame) are collapsed to
	their median and stored under ``frame_id``; nothing is carried between
	frames. Frames where every member sample was gate-rejected leave no entry,
	so the write path falls back to the class-constant height. Shared by both
	tracker classes so their behavior cannot drift.

	Args:
		tr: The :class:`GlobalTrack` the observation is being committed to.
		wo: The committed :class:`WorldObservation`.
		frame_id: The current frame index.
	"""
	heights = [m.height_sample for m in wo.members if m.height_sample is not None]
	if heights:
		tr.height_by_frame[frame_id] = float(np.median(heights))


class BEVFusionTracker:
	"""Online BEV multi-camera tracker producing stable global object ids.

	Drives the per-frame pipeline: :meth:`group_observations` fuses per-camera
	observations on the ground plane, :meth:`update` associates those groups to
	persistent :class:`GlobalTrack` objects (carry-forward then gated BEV/ReID),
	and :meth:`finalize` interpolates short gaps and emits per-frame positions.

	Tracks live in ``self.tracks`` while active/coasting and move to
	``self.retired`` once unseen for more than ``max_age`` frames; both lists are
	emitted at finalize time. All gates and weights come from ``cfg``.
	"""

	def __init__(self, class_labels, reid_class_ids, scene_id: int, fps: int, cfg: dict):
		"""Configure the tracker from a scene's class labels and MTMC config.

		Resolves the Person class id from ``class_labels`` (falling back to the
		default if lookup fails) and caches every gate/weight from ``cfg`` so the
		hot per-frame path does no dict lookups.

		Args:
			class_labels: Class-label table exposing ``get_id_by_name``; may be
				None, in which case the default Person id is used.
			scene_id: Numeric scene id, recorded for output rows.
			fps: Scene frame rate; clamped to at least 1 and used to convert
				``max_speed_m_s`` into a per-frame gate expansion.
			cfg: MTMC parameter dict (see ``DEFAULT_MTMC``) with keys such as
				``bev_group_gate``, ``reid_track_thresh``, ``max_age``, etc.
		"""
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

		self.tracks   = []  # active + coasting
		self.retired  = []  # age_missing > max_age, kept for output
		self._next_id = collections.defaultdict(lambda: 1)   # per-class object_id counter

	# ------------------------------------------------------------------ #
	# helpers
	# ------------------------------------------------------------------ #
	def _bev_thresh_for(self, class_id):
		"""Return the per-class BEV association distance gate, in meters.

		Args:
			class_id: Object class id to look up.

		Returns:
			The configured threshold for ``class_id``, or a 2.0 m default when
			the id is outside the configured table.
		"""
		if 0 <= class_id < len(self.bev_thresh_by_class):
			return self.bev_thresh_by_class[class_id]
		return 2.0

	def _is_person(self, class_id):
		"""Return whether ``class_id`` uses the appearance/ReID path (Person or FourierGR1T2)."""
		return class_id in self.reid_class_ids

	# ------------------------------------------------------------------ #
	# Step 4 -- per-frame cross-camera grouping
	# ------------------------------------------------------------------ #
	def group_observations(self, cam_observations: list) -> list:
		"""Fuse per-camera observations of one frame into world observations.

		Observations are clustered independently within each class (a Forklift
		never merges with a Person), then each cluster is collapsed to one
		:class:`WorldObservation`.

		Args:
			cam_observations (list[CamObservation]): This frame's ``CamObservation`` list across all cameras.

		Returns:
			A list of :class:`WorldObservation`, one per detected object.
		"""
		by_class = collections.defaultdict(list)
		for o in cam_observations:
			by_class[o.class_id].append(o)

		world_obs = []
		for class_id, items in by_class.items():
			for cluster in self._cluster(items, class_id):
				world_obs.append(self._make_world_observation(class_id, cluster, self.make_world_observation_mode))
		return world_obs

	def _cluster(self, items, class_id):
		"""Greedy single-linkage with a <=1-per-camera constraint.

		Pairs are merged in ascending BEV distance; a merge is skipped when the
		two clusters share a camera, so the spatially closest link wins on
		conflict and every cluster holds at most one observation per camera.

		Persons gate on both BEV distance and part-based ReID distance; other
		classes gate on BEV distance alone.

		The `gate` is a threshold test that decides whether two things are allowed
			to be considered the same object or not. 
			- For non-person classes: the gate is just a BEV distance threshold.
			- For persons, the gate is a combination of 
				a BEV distance threshold and a ReID distance threshold.
				- The BEV distance threshold is wider for persons to allow for more spatial variation, and the
				- ReID distance threshold helps to disambiguate between different people who might be close together in the BEV space.

		Args:
			items (list[CamObservation]): Observations of a single class within one frame.
			class_id: The shared class id (selects the gate and ReID path).

		Returns:
			A list of clusters, each a list of ``CamObservation`` (<=1 per
			camera).
		"""
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
		"""Collapse one cluster into a single :class:`WorldObservation`.

		Find the World_xy (anchor_point) of the cluster by, choose 1: 
			- Averaging the world positions of all members in the cluster.
			- The one have hightest confidence score (if available) as the anchor point.			

		The world position is the mean of member positions; persons additionally
		get a fused embedding/visibility pair.

		Args:
			class_id: Shared class id of the cluster.
			cluster: Member ``CamObservation`` list (<=1 per camera).
			mode: The mode for collapsing the cluster ('max_confidence', 'mean',
				or 'weighted_mean'); unknown values fall back to 'max_confidence'.

		Returns:
			The fused :class:`WorldObservation`.
		"""
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
		"""Fuse member ReID descriptors into one visibility-weighted embedding.

		Each of the 6 body parts is averaged across cameras weighted by that
		part's visibility, so a part seen clearly on one camera dominates the
		fused descriptor; parts with zero total visibility fall back to the plain
		mean. The fused per-part vectors are L2-normalized and the fused
		visibility is the per-part max across members.

		Args:
			cluster: Member ``CamObservation`` list; members without both an
				embedding and visibility are ignored.

		Returns:
			A ``(fused_embedding(6, D), fused_visibility(6,))`` float32 tuple, or
			``(None, None)`` when no member carries ReID features.
		"""
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
		"""Associate one frame's world observations to global tracks.

		Runs two association passes, then handles the leftovers:

		1. **Pass A (carry-forward):** match an observation to a track when they
		   share a single-camera ``(cam_idx, local_track_id)`` member. This is
		   cheap and the main defense against id switches; the strongest match
		   wins when a track is contended.
		2. **Pass B (gated assignment):** resolve the remaining observations and
		   tracks per class with a gated BEV(+ReID) Hungarian assignment.
		3. Spawn new tracks for still-unmatched observations that are not too
		   close to an existing track (anti-respawn).
		4. Coast unmatched tracks forward by their velocity, ageing them and
		   retiring any past ``max_age``.

		Args:
			world_obs: This frame's :class:`WorldObservation` list.
			frame_id: The current frame index (monotonic).
		"""
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
		for observation_index, world_observation in enumerate(world_obs):
			counts = {}
			for member in world_observation.members:
				tr = member_index.get((member.cam_idx, member.local_track_id))
				# Only carry forward when the member's owner is the same class.
				if tr is not None and tr.class_id == world_observation.class_id:
					counts[tr] = counts.get(tr, 0) + 1
			observations_track_counts.append(counts)
			for tr, c in counts.items():
				candidates.append((c, observation_index, tr))

		# Greedily bind the strongest links first (most shared members). The
		# `matched_*` guards make this a one-to-one greedy assignment: once an
		# observation or track is taken, weaker competing candidates are skipped.
		candidates.sort(key=lambda x: -x[0])
		for c, observation_index, tr in candidates:
			if observation_index in matched_observations or tr in matched_tracks:
				continue
				
			# DEBUG:
			# An observation whose members point at >1 track is a contention; we
			# bind it to the strongest (this candidate, sorted first) and log it.
			# if len(observations_track_counts[observation_index]) > 1:
			# 	logger.debug(f"[frame {frame_id}] obs {observation_index} carry-forward spans "
			# 				 f"{len(observations_track_counts[observation_index])} tracks; binding the strongest.")
			
			self._update_track(tr, world_obs[observation_index], frame_id)
			matched_observations.add(observation_index)
			matched_tracks.add(tr)

		# ---- Pass B: gated BEV(+ReID) assignment for the remainder ----
		# Whatever Pass A could not link (no shared member id) is resolved here.
		remain_obs    = [oi for oi in range(len(world_obs)) if oi not in matched_observations]
		remain_tracks = [tr for tr in existing if tr not in matched_tracks]
		# Associate per class so the cost matrix never mixes classes; each class's
		# leftovers go through a gated BEV(+ReID) Hungarian assignment.
		for class_id in {world_obs[oi].class_id for oi in remain_obs}:
			obs_c = [oi for oi in remain_obs if world_obs[oi].class_id == class_id]
			trk_c = [tr for tr in remain_tracks if tr.class_id == class_id]
			if not obs_c or not trk_c:
				continue

			matched_pairs = self._associate(world_obs, obs_c, trk_c, class_id)

			for observation_index, tr in matched_pairs:
				self._update_track(tr, world_obs[observation_index], frame_id)
				matched_observations.add(observation_index)
				matched_tracks.add(tr)

		# ---- spawn new tracks for still-unmatched observations ----
		# min-sep is checked against the existing tracks (anti-respawn), using
		# their current-frame positions; two genuinely distinct unmatched
		# observations are intentionally allowed to both spawn (grouping, not
		# min-sep, is what dedupes within a frame).
		snapshot   = list(existing)
		new_tracks = []
		for observation_index in range(len(world_obs)):
			# Skip anything already bound in Pass A or Pass B.
			if observation_index in matched_observations:
				continue

			# Suppress respawns: if a same-class track already sits within
			# `new_track_min_sep` meters, treat this as that track (which simply
			# went unmatched this frame) rather than minting a duplicate id.
			world_observation = world_obs[observation_index]
			too_close = any(
				tr.class_id == world_observation.class_id
				and np.linalg.norm(world_observation.world_xy - tr.world_xy) < self.new_track_min_sep
				for tr in snapshot
			)
			if too_close:
				continue

			new_tracks.append(self._spawn_track(world_observation, frame_id))

		# ---- coast unmatched tracks; retire those past max_age ----
		survivors = []
		for tr in existing:
			# Matched tracks were already advanced inside `_update_track`.
			if tr in matched_tracks:
				survivors.append(tr)
				continue
			# Unmatched track: extrapolate its position by velocity (dead-reckon)
			# and age it. Keep coasting until it exceeds `max_age`, then retire it.
			tr.world_xy = tr.world_xy + tr.velocity
			tr.age_missing += 1
			if tr.age_missing > self.max_age:
				tr.active = False
				self.retired.append(tr)
			else:
				survivors.append(tr)
		# Surviving (matched + still-coasting) tracks plus this frame's spawns
		# become the live set carried into the next frame.
		self.tracks = survivors + new_tracks

	INF = 1e6

	def _associate(self, world_obs, obs_c, trk_c, class_id):
		"""Gated BEV(+ReID) Hungarian assignment for one class's leftovers.

		Builds a cost matrix over the candidate observations and tracks, gating
		each pair on a velocity-predicted BEV distance (the gate widens with a
		track's missing age to tolerate longer coasts) and, for persons, on a
		gallery ReID distance. The person cost blends normalized BEV distance and
		ReID distance via ``w_bev``; other classes use raw BEV distance. Pairs
		whose cost stays at the ``INF`` sentinel (gated out) are discarded.

		Args:
			world_obs: The full frame observation list (indexed by ``obs_c``).
			obs_c: Indices into ``world_obs`` of unmatched observations.
			trk_c: Unmatched candidate tracks, all of class ``class_id``.
			class_id: The class being associated.

		Returns:
			A list of ``(obs_index, track)`` matched pairs.
		"""
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

				if d > gate:
					continue

				if is_person:
					rd = (gallery_distance(wo.embedding, wo.visibility, list(tr.gallery), reduce="min")
						  if wo.embedding is not None else math.inf)
					if math.isfinite(rd) and rd > self.reid_track_thresh:
						continue
					rd_eff = rd if math.isfinite(rd) else 1.0
					cost[a, b] = self.w_bev * (d / gate) + (1.0 - self.w_bev) * rd_eff
				else:
					cost[a, b] = d

		rows, cols = linear_sum_assignment(cost)
		
		return [(obs_c[a], trk_c[b]) for a, b in zip(rows, cols) if cost[a, b] < self.INF]

	def _update_track(self, tr, wo, frame_id):
		"""Fold a matched observation into a track.

		Smooths the position with an ``alpha_pos`` exponential filter, updates the
		velocity from the per-frame displacement (dividing by frames elapsed so a
		match after a coast gap stays metric), refreshes the track's per-camera
		members and ReID gallery, and resets its missing-age.

		Args:
			tr: The :class:`GlobalTrack` being updated.
			wo: The matched :class:`WorldObservation`.
			frame_id: The current frame index.
		"""
		obs_xy      = np.asarray(wo.world_xy, dtype=np.float64)
		new_pos     = self.alpha_pos * obs_xy + (1.0 - self.alpha_pos) * tr.world_xy
		elapsed     = max(1, frame_id - tr.last_frame) if tr.last_frame >= 0 else 1
		tr.velocity = (new_pos - tr.world_xy) / elapsed
		tr.world_xy = new_pos
		tr.members  = {m.cam_idx: m.local_track_id for m in wo.members}
		if wo.embedding is not None and wo.visibility is not None:
			tr.gallery.append((np.asarray(wo.embedding), np.asarray(wo.visibility)))
		tr.age_missing = 0
		tr.last_frame = frame_id
		tr.history.append((frame_id, float(new_pos[0]), float(new_pos[1])))
		reduce_height_samples(tr, wo, frame_id)

	def _spawn_track(self, wo, frame_id):
		"""Create a new global track from an unmatched observation.

		Assigns the next per-class ``object_id``, seeds the track's position,
		members and ReID gallery from ``wo`` and records its first history point.

		Args:
			wo: The unmatched :class:`WorldObservation` to start a track from.
			frame_id: The current frame index (the track's first sighting).

		Returns:
			The new :class:`GlobalTrack`.
		"""
		object_id = self._next_id[wo.class_id]
		self._next_id[wo.class_id] += 1
		tr = GlobalTrack(object_id, wo.class_id, wo.world_xy, self.gallery_size)
		tr.members = {m.cam_idx: m.local_track_id for m in wo.members}
		if wo.embedding is not None and wo.visibility is not None:
			tr.gallery.append((np.asarray(wo.embedding), np.asarray(wo.visibility)))
		tr.last_frame = frame_id
		tr.age_missing = 0
		tr.history.append((frame_id, float(wo.world_xy[0]), float(wo.world_xy[1])))
		reduce_height_samples(tr, wo, frame_id)
		return tr

	# ------------------------------------------------------------------ #
	# Step 6 -- temporal smoothing / gap interpolation + finalize
	# ------------------------------------------------------------------ #
	def finalize(self) -> dict:
		"""Materialize per-frame positions for every track and group by class.

		Run once after the last frame. Fills each track's ``frames`` map (with
		short gaps interpolated) and its per-frame ``rotations`` (yaw via
		:meth:`GlobalTrack.find_yaw`) over both active and retired tracks.

		Returns:
			A ``{class_id: list[GlobalTrack]}`` dict ready for output.
		"""
		result = collections.defaultdict(list)
		fps = self.fps if self.fps > 0 else 45
		for tr in self.tracks + self.retired:
			self._build_frames(tr)
			tr.find_yaw(half_window=max(2, round(fps * 1.5)))
			result[tr.class_id].append(tr)
		return dict(result)

	def _build_frames(self, tr):
		"""Fill ``tr.frames`` from sighting history, interpolating short gaps.

		Real sightings are kept as-is; for each consecutive pair the in-between
		frames are linearly interpolated, but only when the gap is short enough
		(``gap - 1 <= gap_threshold``) to avoid bridging long occlusions/id reuse.

		Args:
			tr: The :class:`GlobalTrack` whose ``frames`` map is populated.
			gap_threshold: The maximum gap size to interpolate. Gaps longer than this are left as missing to avoid bridging long occlusions or potential id reuse.
				Units are frames; the actual time this corresponds to depends on the scene's fps (e.g., 60 frames at 30 fps is 2 seconds).
		"""
		hist   = sorted(tr.history)
		frames = {f: (x, y) for f, x, y in hist}
		for k in range(len(hist) - 1):
			f_0, x_0, y_0 = hist[k]
			f_1, x_1, y_1 = hist[k + 1]
			gap = f_1 - f_0
			if gap <= 1 or (gap - 1) > self.interpolating_missing_frames_max_gap:
				continue
			for f in range(f_0 + 1, f_1):
				a = (f - f_0) / gap
				frames[f] = (x_0 + a * (x_1 - x_0), y_0 + a * (y_1 - y_0))
		tr.frames = frames
