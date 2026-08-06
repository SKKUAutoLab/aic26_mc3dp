"""Static configuration / defaults for the AICity DA3 web app."""
from __future__ import annotations

import os

# Project root = .../Depth-Anything-3
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools")

# Defaults (user can override the dataset root in the UI / CLI).
DEFAULT_DATASET_ROOT = "/media/vsw-ws-05/hdd-02/AI_City_Challenge/MTMC_Tracking_2026"
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "aicity2026")

# Candidate split directory names (we only surface those that actually exist).
SPLIT_CANDIDATES = ["train", "val", "valid", "test"]

# Submission files (AICity Track-1 result txt) live here relative to the dataset
# root, e.g. ``<root>/submission/Warehouse_027.txt``. Used by the Submission tab.
DEFAULT_SUBMISSION_SUBDIR = "submission"

# DA3 models offered in the UI. `cached` only marks those we verified locally;
# any other string still works (it just downloads on first use). The nested
# giant-large model gives the strongest, metric depth (best, least-noisy points).
MODELS = [
    {"name": "depth-anything/da3-small", "cached": False},
    {"name": "depth-anything/da3-base", "cached": False},
    {"name": "depth-anything/da3-large", "cached": False},
    {"name": "depth-anything/da3-giant", "cached": True},
    {"name": "depth-anything/da3nested-giant-large", "cached": True},
    {"name": "depth-anything/da3nested-giant-large-1.1", "cached": False},
]
DEFAULT_MODEL = "depth-anything/da3nested-giant-large"

# "Tối đa chất lượng" preset (user-chosen default; all overridable in the UI).
DEFAULT_PROCESS_RES = 1008
DEFAULT_PROCESS_RES_METHOD = "upper_bound_resize"
DEFAULT_REF_VIEW_STRATEGY = "saddle_balanced"
REF_VIEW_STRATEGIES = ["saddle_balanced", "saddle_sim_range", "middle", "first"]
DEFAULT_USE_RAY_POSE = False
DEFAULT_EXPORT_FORMAT = "npz"
DEFAULT_GPU_ID = 1

# Fusion / point-cloud quality defaults.
DEFAULT_CONF_PERCENTILE = 40.0  # adaptive conf lower percentile (official recipe)
DEFAULT_STRIDE = 1              # full resolution points
DEFAULT_OUTLIER_REMOVAL = True
DEFAULT_NB_NEIGHBORS = 20
DEFAULT_STD_RATIO = 2.0
# Depth-range filter (meters). The model folds sky/background into FAR depth, so
# capping max depth removes those noisy points. 0 = no limit. Sky mask (if a
# model exposes one) is auto-dropped on top of this.
DEFAULT_MIN_DEPTH = 0.1
DEFAULT_MAX_DEPTH = 0.0

# GT depth (NVIDIA SmartSpaces "distance_to_image_plane") encoding.
GT_DEPTH_SCALE = 1000.0  # uint16 value / scale -> meters
GT_DEPTH_INVALID = 0  # no return
GT_DEPTH_SKY = 65535  # saturated / sky / out of range

HOST = "0.0.0.0"
PORT = 8010
