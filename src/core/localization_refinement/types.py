

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OnlineRefinerConfig:
    """All gates used by the causal, single-frame refiner."""

    scene_id: int | None = 27
    person_class_id: int = 0
    reference_camera_ids: frozenset[str] = frozenset(
        f"Camera_{index:04d}" for index in range(4)
    )
    fisheye_camera_ids: frozenset[str] = frozenset(
        f"Camera_{index:04d}" for index in range(4, 7)
    )
    group_gate_m: float = 1.25
    opposing_pair_gate_m: float = 1.50
    opposing_reference_pairs: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Camera_0004", ("Camera_0002",)),
    )
    reid_gate: float = 0.35
    reid_bev_weight: float = 0.50
    row_gate_m: float = 1.50
    crowded_radius_m: float = 1.50
    crowded_row_gate_m: float = 0.30
    min_correction_m: float = 0.15
    max_shift_m: float = 2.00
    before_row_reference_ray_m: float = 8.00

    def __post_init__(self) -> None:
        numeric = {
            "group_gate_m": self.group_gate_m,
            "opposing_pair_gate_m": self.opposing_pair_gate_m,
            "reid_gate": self.reid_gate,
            "reid_bev_weight": self.reid_bev_weight,
            "row_gate_m": self.row_gate_m,
            "crowded_radius_m": self.crowded_radius_m,
            "crowded_row_gate_m": self.crowded_row_gate_m,
            "min_correction_m": self.min_correction_m,
            "max_shift_m": self.max_shift_m,
            "before_row_reference_ray_m": self.before_row_reference_ray_m,
        }
        for name, value in numeric.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite (got {value})")
        positive = {
            name: value
            for name, value in numeric.items()
            if name
            in {
                "group_gate_m",
                "opposing_pair_gate_m",
                "row_gate_m",
                "crowded_radius_m",
                "min_correction_m",
                "max_shift_m",
            }
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0 (got {value})")
        if self.crowded_row_gate_m < 0:
            raise ValueError("crowded_row_gate_m must be >= 0")
        if self.reid_gate < 0:
            raise ValueError("reid_gate must be >= 0")
        if self.before_row_reference_ray_m < 0:
            raise ValueError("before_row_reference_ray_m must be >= 0")
        if not 0.0 <= self.reid_bev_weight <= 1.0:
            raise ValueError("reid_bev_weight must be in [0, 1]")
        if self.reference_camera_ids & self.fisheye_camera_ids:
            raise ValueError("reference and fisheye camera ids must be disjoint")
        if self.opposing_pair_gate_m <= self.group_gate_m:
            raise ValueError("opposing_pair_gate_m must exceed group_gate_m")
        for fisheye_camera, reference_cameras in self.opposing_reference_pairs:
            if fisheye_camera not in self.fisheye_camera_ids:
                raise ValueError(
                    f"opposing fisheye camera {fisheye_camera} is not configured"
                )
            unknown = set(reference_cameras) - set(self.reference_camera_ids)
            if unknown:
                raise ValueError(
                    f"opposing reference cameras are not configured: {sorted(unknown)}"
                )


@dataclass(frozen=True)
class LocalizationObservation:
    """One current-frame, single-camera ground-plane observation."""

    camera_id: str
    frame_id: int
    class_id: int
    world_xy: tuple[float, float]
    confidence: float = 1.0
    local_track_id: int | None = None
    embedding: Any = field(default=None, repr=False, compare=False)
    visibility: Any = field(default=None, repr=False, compare=False)
    reference_edge_reason: str | None = None

    def __post_init__(self) -> None:
        if len(self.world_xy) != 2:
            raise ValueError("world_xy must contain exactly two values")
        object.__setattr__(
            self,
            "world_xy",
            (float(self.world_xy[0]), float(self.world_xy[1])),
        )
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(self, "confidence", float(self.confidence))


@dataclass(frozen=True)
class Track1Row:
    """Canonical immutable AI City Track 1 row."""

    scene_id: int
    class_id: int
    object_id: int
    frame_id: int
    x: float
    y: float
    z: float
    width: float
    length: float
    height: float
    yaw: float

    def with_xy(self, x: float, y: float) -> "Track1Row":
        return Track1Row(
            scene_id=self.scene_id,
            class_id=self.class_id,
            object_id=self.object_id,
            frame_id=self.frame_id,
            x=float(x),
            y=float(y),
            z=self.z,
            width=self.width,
            length=self.length,
            height=self.height,
            yaw=self.yaw,
        )


@dataclass(frozen=True)
class FisheyeCandidate:
    """A fisheye observation and its trusted reference-camera target."""

    candidate_id: str
    group_id: str
    frame_id: int
    fisheye_camera_id: str
    reference_camera_id: str
    original_xy: tuple[float, float]
    target_xy: tuple[float, float]
    error_m: float
    reference_ray_m: float | None
    reference_edge_reason: str | None
    source: str
    observation_index: int


@dataclass(frozen=True)
class FrameRefinementResult:
    """Rows plus deterministic diagnostics from one ``refine_frame`` call."""

    rows: list[Any]
    candidates: tuple[FisheyeCandidate, ...]
    report: dict[str, Any]
