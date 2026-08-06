

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.bev_fusion.reid import part_distance

from .adapters import replace_row_xy, track1_row_from_value
from .types import (
    FisheyeCandidate,
    FrameRefinementResult,
    LocalizationObservation,
    OnlineRefinerConfig,
    Track1Row,
)


@dataclass(frozen=True)
class _IndexedObservation:
    index: int
    value: LocalizationObservation


@dataclass(frozen=True)
class _PotentialMatch:
    row_index: int
    candidate_index: int
    group_id: str
    score_m: float
    basis: str
    effective_gate_m: float
    nearby_count: int
    target_xy: tuple[float, float]
    target_source: str
    shift_m: float


class OnlineFisheyeLocalizationRefiner:
    """Refine only person x/y using synchronized reference-camera consensus.

    ``refine_frame`` is intentionally stateless. It reads no dataset, video,
    cache, submission file, or future frame, so the caller owns the AIC online
    loop and may invoke this object as its final per-frame layer.
    """

    _EDGE_FALLBACK_REASONS = frozenset({"no ankle", "1 ankle", "ankle edge"})

    def __init__(
        self,
        config: OnlineRefinerConfig | None = None,
        camera_centers_xyz: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.config = config or OnlineRefinerConfig()
        self.camera_centers_xyz = {
            str(camera_id): np.asarray(center, dtype=np.float64).reshape(3)
            for camera_id, center in (camera_centers_xyz or {}).items()
        }

    def refine_frame(
        self,
        frame_id: int,
        observations: Sequence[LocalizationObservation],
        rows: Sequence[Any],
    ) -> FrameRefinementResult:
        """Return refined copies of current-frame rows and diagnostics.

        The API is strict: every row and observation must belong to ``frame_id``.
        Non-person rows pass through untouched. For a matched person, only x/y
        are copied with new values; every id, z/dimension/yaw value and any
        extra mapping key stays unchanged.
        """

        frame_id = int(frame_id)
        output_rows = list(rows)
        normalized_rows = [track1_row_from_value(row) for row in rows]
        wrong_row_frames = sorted(
            {row.frame_id for row in normalized_rows if row.frame_id != frame_id}
        )
        if wrong_row_frames:
            raise ValueError(
                f"refine_frame({frame_id}) received rows from frames "
                f"{wrong_row_frames}"
            )
        indexed_observations = self._current_observations(frame_id, observations)
        candidates = self._build_candidates(frame_id, indexed_observations)

        eligible_rows = [
            index
            for index, row in enumerate(normalized_rows)
            if row.frame_id == frame_id
            and row.class_id == self.config.person_class_id
            and (self.config.scene_id is None or row.scene_id == self.config.scene_id)
        ]
        best_by_row_group: dict[tuple[int, str], _PotentialMatch] = {}
        rejected_pairs: list[dict[str, Any]] = []
        for row_index in eligible_rows:
            row = normalized_rows[row_index]
            effective_gate, nearby_count = self._row_gate(
                row_index, row, eligible_rows, normalized_rows
            )
            row_xy = np.asarray((row.x, row.y), dtype=np.float64)
            for candidate_index, candidate in enumerate(candidates):
                original_distance = float(
                    np.linalg.norm(row_xy - np.asarray(candidate.original_xy))
                )
                target_distance = float(
                    np.linalg.norm(row_xy - np.asarray(candidate.target_xy))
                )
                if original_distance <= target_distance:
                    score, basis = original_distance, "original"
                else:
                    score, basis = target_distance, "target"
                if score > effective_gate:
                    continue
                target_xy, target_source = self._target_for_row(row, candidate)
                shift_m = _distance((row.x, row.y), target_xy)
                if shift_m > self.config.max_shift_m:
                    rejected_pairs.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "group_id": candidate.group_id,
                            "object_id": row.object_id,
                            "reason": "shift_too_large",
                            "shift_m": shift_m,
                        }
                    )
                    continue
                potential = _PotentialMatch(
                    row_index=row_index,
                    candidate_index=candidate_index,
                    group_id=candidate.group_id,
                    score_m=score,
                    basis=basis,
                    effective_gate_m=effective_gate,
                    nearby_count=nearby_count,
                    target_xy=(float(target_xy[0]), float(target_xy[1])),
                    target_source=target_source,
                    shift_m=shift_m,
                )
                key = (row_index, candidate.group_id)
                current = best_by_row_group.get(key)
                if current is None or (
                    potential.score_m,
                    candidate.candidate_id,
                ) < (
                    current.score_m,
                    candidates[current.candidate_index].candidate_id,
                ):
                    best_by_row_group[key] = potential

        candidate_groups = sorted({candidate.group_id for candidate in candidates})
        group_position = {
            group_id: index for index, group_id in enumerate(candidate_groups)
        }
        row_position = {
            row_index: index for index, row_index in enumerate(eligible_rows)
        }
        assignments: list[_PotentialMatch] = []
        if eligible_rows and candidate_groups and best_by_row_group:
            unmatched_cost = 1_000_000.0
            invalid_cost = 2_000_000.0
            cost = np.full(
                (len(eligible_rows), len(candidate_groups) + len(eligible_rows)),
                invalid_cost,
                dtype=np.float64,
            )
            for row_pos in range(len(eligible_rows)):
                cost[row_pos, len(candidate_groups) + row_pos] = unmatched_cost
            for (row_index, group_id), potential in best_by_row_group.items():
                row_pos = row_position[row_index]
                group_pos = group_position[group_id]
                # The tiny stable offsets make equal-cost solutions deterministic.
                cost[row_pos, group_pos] = (
                    potential.score_m + row_pos * 1e-9 + group_pos * 1e-12
                )
            selected_rows, selected_columns = linear_sum_assignment(cost)
            for row_pos, column in zip(selected_rows, selected_columns):
                if column >= len(candidate_groups) or cost[row_pos, column] >= unmatched_cost:
                    continue
                row_index = eligible_rows[row_pos]
                group_id = candidate_groups[column]
                assignments.append(best_by_row_group[(row_index, group_id)])

        applied: list[dict[str, Any]] = []
        changed_count = 0
        zero_shift_count = 0
        for assignment in sorted(assignments, key=lambda item: item.row_index):
            candidate = candidates[assignment.candidate_index]
            row = normalized_rows[assignment.row_index]
            if assignment.shift_m > 1e-12:
                output_rows[assignment.row_index] = replace_row_xy(
                    rows[assignment.row_index],
                    assignment.target_xy[0],
                    assignment.target_xy[1],
                )
                changed_count += 1
            else:
                zero_shift_count += 1
            applied.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "group_id": candidate.group_id,
                    "object_id": row.object_id,
                    "frame_id": row.frame_id,
                    "old_xy": [row.x, row.y],
                    "new_xy": [
                        assignment.target_xy[0],
                        assignment.target_xy[1],
                    ],
                    "shift_m": assignment.shift_m,
                    "match_score_m": assignment.score_m,
                    "match_basis": assignment.basis,
                    "effective_gate_m": assignment.effective_gate_m,
                    "nearby_row_count": assignment.nearby_count,
                    "target_source": assignment.target_source,
                }
            )

        camera_counts: dict[str, int] = {}
        for candidate in candidates:
            camera_counts[candidate.fisheye_camera_id] = (
                camera_counts.get(candidate.fisheye_camera_id, 0) + 1
            )
        report = {
            "mode": "online_current_frame_fisheye_consensus",
            "frame_id": frame_id,
            "num_input_rows": len(rows),
            "num_input_observations": len(observations),
            "num_current_person_observations": len(indexed_observations),
            "num_candidates": len(candidates),
            "num_candidate_groups": len(candidate_groups),
            "num_matches": len(assignments),
            "num_applied": len(applied),
            "num_rows_changed": changed_count,
            "num_zero_shift_fallbacks": zero_shift_count,
            "num_skipped": 0,
            "num_rejected_pairs": len(rejected_pairs),
            "num_unmatched_candidate_groups": len(candidate_groups)
            - len(assignments),
            "candidate_cameras": camera_counts,
            "applied": applied,
            "skipped": [],
            "rejected_pairs": rejected_pairs,
        }
        return FrameRefinementResult(
            rows=output_rows,
            candidates=tuple(candidates),
            report=report,
        )

    def _current_observations(
        self,
        frame_id: int,
        observations: Sequence[LocalizationObservation],
    ) -> list[_IndexedObservation]:
        allowed_cameras = (
            self.config.reference_camera_ids | self.config.fisheye_camera_ids
        )
        current: list[_IndexedObservation] = []
        for index, observation in enumerate(observations):
            if not isinstance(observation, LocalizationObservation):
                raise TypeError(
                    "observations must be LocalizationObservation objects; use "
                    "the adapters for CamObservation or WorldDetection"
                )
            if observation.frame_id != frame_id:
                raise ValueError(
                    f"refine_frame({frame_id}) received an observation from "
                    f"frame {observation.frame_id}"
                )
            if observation.class_id != self.config.person_class_id:
                continue
            if observation.camera_id not in allowed_cameras:
                continue
            if not np.isfinite(np.asarray(observation.world_xy)).all():
                continue
            current.append(_IndexedObservation(index, observation))
        return current

    def _build_candidates(
        self,
        frame_id: int,
        observations: list[_IndexedObservation],
    ) -> list[FisheyeCandidate]:
        candidates: list[FisheyeCandidate] = []
        used_fisheye_indices: set[int] = set()
        clusters = self._group_observations(observations)
        for cluster_index, cluster in enumerate(clusters):
            references = [
                item
                for item in cluster
                if item.value.camera_id in self.config.reference_camera_ids
            ]
            fisheyes = [
                item
                for item in cluster
                if item.value.camera_id in self.config.fisheye_camera_ids
            ]
            if not references or not fisheyes:
                continue
            for fisheye in fisheyes:
                reference, ray_m = self._trusted_reference(references)
                candidate = self._make_candidate(
                    frame_id,
                    fisheye,
                    reference,
                    ray_m,
                    source="cross_camera_cluster",
                    suffix=f"c{cluster_index:03d}",
                    group_id=f"f{frame_id:06d}_cluster{cluster_index:03d}",
                )
                if candidate is None:
                    continue
                candidates.append(candidate)
                used_fisheye_indices.add(fisheye.index)

        pair_index = 0
        by_camera: dict[str, list[_IndexedObservation]] = {}
        for item in observations:
            by_camera.setdefault(item.value.camera_id, []).append(item)
        for fisheye_camera, reference_cameras in self.config.opposing_reference_pairs:
            for fisheye in by_camera.get(fisheye_camera, []):
                if fisheye.index in used_fisheye_indices:
                    continue
                references = [
                    item
                    for camera_id in reference_cameras
                    for item in by_camera.get(camera_id, [])
                ]
                if not references:
                    continue
                reference = min(
                    references,
                    key=lambda item: (
                        _distance(fisheye.value.world_xy, item.value.world_xy),
                        item.value.camera_id,
                        item.index,
                    ),
                )
                error_m = _distance(
                    fisheye.value.world_xy, reference.value.world_xy
                )
                if (
                    error_m <= self.config.group_gate_m
                    or error_m > self.config.opposing_pair_gate_m
                    or error_m < self.config.min_correction_m
                    or not self._reid_pair_allowed(fisheye.value, reference.value)
                ):
                    continue
                ray_m = self._reference_ray(reference.value)
                candidate = self._make_candidate(
                    frame_id,
                    fisheye,
                    reference,
                    ray_m,
                    source="opposing_pair_near_miss",
                    suffix=f"pair{pair_index:03d}",
                    group_id=(
                        f"f{frame_id:06d}_pair_{fisheye.value.camera_id}_"
                        f"{reference.value.camera_id}_{pair_index:03d}"
                    ),
                )
                pair_index += 1
                if candidate is not None:
                    candidates.append(candidate)
                    used_fisheye_indices.add(fisheye.index)

        return sorted(
            candidates,
            key=lambda candidate: (-candidate.error_m, candidate.candidate_id),
        )

    def _group_observations(
        self, observations: list[_IndexedObservation]
    ) -> list[list[_IndexedObservation]]:
        clusters: list[list[_IndexedObservation]] = []
        ordered = sorted(
            observations,
            key=lambda item: (
                -item.value.confidence,
                item.value.camera_id,
                -1
                if item.value.local_track_id is None
                else item.value.local_track_id,
                item.index,
            ),
        )
        for item in ordered:
            best_cluster: list[_IndexedObservation] | None = None
            best_cost = math.inf
            xy = np.asarray(item.value.world_xy, dtype=np.float64)
            for cluster in clusters:
                if any(
                    member.value.camera_id == item.value.camera_id
                    for member in cluster
                ):
                    continue
                center = np.mean(
                    [member.value.world_xy for member in cluster], axis=0
                )
                bev_distance = float(np.linalg.norm(xy - center))
                if bev_distance > self.config.group_gate_m:
                    continue
                appearance = self._appearance_distance(item.value, cluster)
                if (
                    appearance is not None
                    and self.config.reid_gate > 0
                    and appearance > self.config.reid_gate
                ):
                    continue
                normalized_bev = bev_distance / self.config.group_gate_m
                normalized_appearance = (
                    normalized_bev
                    if appearance is None or self.config.reid_gate <= 0
                    else appearance / self.config.reid_gate
                )
                cost = (
                    self.config.reid_bev_weight * normalized_bev
                    + (1.0 - self.config.reid_bev_weight)
                    * normalized_appearance
                )
                if cost < best_cost:
                    best_cluster = cluster
                    best_cost = cost
            if best_cluster is None:
                clusters.append([item])
            else:
                best_cluster.append(item)
        return clusters

    def _appearance_distance(
        self,
        observation: LocalizationObservation,
        cluster: list[_IndexedObservation],
    ) -> float | None:
        if observation.embedding is None:
            return None
        distances: list[float] = []
        for member in cluster:
            other = member.value
            if other.embedding is None:
                continue
            try:
                distance = part_distance(
                    observation.embedding,
                    other.embedding,
                    observation.visibility,
                    other.visibility,
                )
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(distance):
                distances.append(float(distance))
        return min(distances) if distances else None

    def _reid_pair_allowed(
        self,
        first: LocalizationObservation,
        second: LocalizationObservation,
    ) -> bool:
        if self.config.reid_gate <= 0:
            return True
        if first.embedding is None or second.embedding is None:
            return True
        try:
            distance = part_distance(
                first.embedding,
                second.embedding,
                first.visibility,
                second.visibility,
            )
        except (TypeError, ValueError, IndexError):
            return True
        return not math.isfinite(distance) or distance <= self.config.reid_gate

    def _trusted_reference(
        self, references: list[_IndexedObservation]
    ) -> tuple[_IndexedObservation, float | None]:
        with_rays = [
            (self._reference_ray(item.value), item) for item in references
        ]
        finite = [
            (ray, item)
            for ray, item in with_rays
            if ray is not None and math.isfinite(ray)
        ]
        if finite:
            ray, item = min(
                finite,
                key=lambda pair: (
                    pair[0],
                    -pair[1].value.confidence,
                    pair[1].value.camera_id,
                ),
            )
            return item, float(ray)
        item = min(
            references,
            key=lambda member: (
                -member.value.confidence,
                member.value.camera_id,
                member.index,
            ),
        )
        return item, None

    def _reference_ray(
        self, reference: LocalizationObservation
    ) -> float | None:
        center = self.camera_centers_xyz.get(reference.camera_id)
        if center is None:
            return None
        target = np.array(
            [reference.world_xy[0], reference.world_xy[1], 0.0],
            dtype=np.float64,
        )
        return float(np.linalg.norm(target - center))

    def _make_candidate(
        self,
        frame_id: int,
        fisheye: _IndexedObservation,
        reference: _IndexedObservation,
        ray_m: float | None,
        source: str,
        suffix: str,
        group_id: str,
    ) -> FisheyeCandidate | None:
        error_m = _distance(
            fisheye.value.world_xy, reference.value.world_xy
        )
        if error_m < self.config.min_correction_m:
            return None
        candidate_id = (
            f"f{frame_id:06d}_{suffix}_{fisheye.value.camera_id}_"
            f"{fisheye.index:04d}"
        )
        return FisheyeCandidate(
            candidate_id=candidate_id,
            group_id=group_id,
            frame_id=frame_id,
            fisheye_camera_id=fisheye.value.camera_id,
            reference_camera_id=reference.value.camera_id,
            original_xy=fisheye.value.world_xy,
            target_xy=reference.value.world_xy,
            error_m=error_m,
            reference_ray_m=ray_m,
            reference_edge_reason=reference.value.reference_edge_reason,
            source=source,
            observation_index=fisheye.index,
        )

    def _target_for_row(
        self,
        row: Track1Row,
        candidate: FisheyeCandidate,
    ) -> tuple[tuple[float, float], str]:
        if candidate.reference_edge_reason in self._EDGE_FALLBACK_REASONS:
            reason = candidate.reference_edge_reason.replace(" ", "_")
            return (row.x, row.y), f"before_row_reference_{reason}"
        if (
            candidate.reference_ray_m is not None
            and self.config.before_row_reference_ray_m > 0
            and candidate.reference_ray_m
            > self.config.before_row_reference_ray_m
        ):
            return (row.x, row.y), "before_row_reference_ray_gt_threshold"
        return candidate.target_xy, candidate.source

    def _row_gate(
        self,
        row_index: int,
        row: Track1Row,
        eligible_rows: list[int],
        rows: list[Track1Row],
    ) -> tuple[float, int]:
        nearby_count = sum(
            1
            for other_index in eligible_rows
            if other_index != row_index
            and _distance(
                (row.x, row.y),
                (rows[other_index].x, rows[other_index].y),
            )
            <= self.config.crowded_radius_m
        )
        if nearby_count and self.config.crowded_row_gate_m > 0:
            return (
                min(self.config.row_gate_m, self.config.crowded_row_gate_m),
                nearby_count,
            )
        return self.config.row_gate_m, nearby_count


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(
        float(first[0]) - float(second[0]),
        float(first[1]) - float(second[1]),
    )
