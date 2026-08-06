"""Filter noisy person boxes out"""
import argparse
import json
import os

import numpy as np

from . import rows as rows_io, zones as zones_mod

# scene_id (column 0 of the submission) -> the scene whose videos and calibration to use. Any other
# scene works too: pass --scene / --zone-dir and point them at its data.
SCENE_BY_ID = {23: "Warehouse_023", 24: "Warehouse_024"}
SCENE_FRAMES = {23: (0, 8999), 24: (0, 8999)}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_OUTPUT = os.path.join("output", "filter_object_noise")
# The calibration that carries the zone map's scaleFactor / translationToGlobalCoordinates.
CALIBRATION = "calibration_modified.json"


def plan_run(*, submission, dataset, pose_model, gpu=0, output="", split="test", scene="",
             start=None, end=None, zone_dir="", verbose=True) -> dict:
    """Resolve a run from the submission path, the dataset root and the weights.

    The scene comes from the submission's first row; the zone polygons and the calibration follow
    from the scene. Shared by the CLI and :class:`~.filterer.NoiseFilter`, so the two cannot drift.
    """
    inp = os.path.abspath(submission)
    if not os.path.isfile(inp):
        raise SystemExit(f"submission not found: {inp}")
    scene_id = rows_io.scene_id_of(inp)
    image_scene = scene or SCENE_BY_ID.get(scene_id)
    if not image_scene:
        raise SystemExit(f"unknown scene_id {scene_id} (known: {sorted(SCENE_BY_ID)})")
    if scene_id not in SCENE_FRAMES and (start is None or end is None):
        raise SystemExit(f"scene_id {scene_id} has no known frame range; pass start= and end=")

    if not os.path.isdir(pose_model):
        raise SystemExit(
            f"ViTPose weights not found: {pose_model}\n"
            f"  --pose-model is a DIRECTORY in HuggingFace format, not a single file:\n"
            f"      vitpose-plus-huge/config.json\n"
            f"      vitpose-plus-huge/model.safetensors\n"
            f"      vitpose-plus-huge/preprocessor_config.json\n"
            f"  The weights are not in this repository.")
    missing = [f for f in ("config.json", "model.safetensors")
               if not os.path.isfile(os.path.join(pose_model, f))]
    if missing:
        raise SystemExit(f"{pose_model} is not a ViTPose weights directory: missing {missing}")

    scene_dir = os.path.join(dataset, split, image_scene)
    calib_path = os.path.join(scene_dir, CALIBRATION)
    if not os.path.isfile(calib_path):
        raise SystemExit(f"{CALIBRATION} not found for {image_scene}: {calib_path}")
    calibration = json.load(open(calib_path, encoding="utf-8"))
    sensors = [s for s in calibration["sensors"] if s.get("type") == "camera"]
    cameras = sorted(s["id"] for s in sensors)
    K = {s["id"]: np.asarray(s["intrinsicMatrix"], dtype=np.float64) for s in sensors}
    E = {s["id"]: np.asarray(s["extrinsicMatrix"], dtype=np.float64) for s in sensors}

    zdir = zone_dir or os.path.join(PROJECT_ROOT, "configs", image_scene, "zone_bev")
    if not os.path.isdir(zdir):
        raise SystemExit(f"zone polygons not found: {zdir}")
    zones = zones_mod.load_zones(zdir, calibration, cameras)
    if not zones:
        raise SystemExit(f"no camera zones in {zdir} (expected Camera_XXXX.json)")

    rows_by_frame, _, n_rows = rows_io.read_by_frame(inp)
    lo, hi = SCENE_FRAMES.get(scene_id, (min(rows_by_frame), max(rows_by_frame)))
    start = lo if start is None else start
    end = hi if end is None else end

    out_dir = os.path.abspath(os.path.join(output or DEFAULT_OUTPUT, f"Warehouse_{scene_id:03d}"))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(inp))[0]
    # A partial range is named after the range, so a short test run cannot quietly overwrite the
    # full scene's output and be mistaken for it later.
    partial = "" if (start, end) == (lo, hi) else f"_{start}_{end}"
    out_path = os.path.join(out_dir, f"{stem}_filtered{partial}.txt")

    n_persons = sum(1 for fr in rows_by_frame.values() for r in fr if r.is_person)
    if verbose:
        print(f"scene      : {image_scene} (submission scene_id {scene_id})")
        print(f"input      : {inp}")
        print(f"             {n_rows} rows, {n_persons} of them person (class 0)")
        print(f"frames     : {start}..{end}  (in order; cameras decoded in lockstep)")
        print(f"cameras    : {len(cameras)} | zones: {len(zones)} "
              f"({sum(len(z) for z in zones.values())} polygons)")
        print(f"pose       : {pose_model}, gpu {gpu}")
        print(f"output     : {out_path}\n")

    return {"scene": image_scene, "scene_id": scene_id, "scene_dir": scene_dir,
            "start": start, "end": end, "out_path": out_path, "out_dir": out_dir,
            "rows_by_frame": rows_by_frame, "calibration": calibration,
            "cameras": cameras, "K": K, "E": E, "zones": zones,
            "n_rows": n_rows, "n_persons": n_persons}


def parse_args(prog=None):
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True, help="submission .txt to filter (11 columns)")
    ap.add_argument("--dataset", "--root", dest="dataset", required=True,
                    help="dataset root: <dataset>/<split>/<scene>/{videos,calibration_modified.json}")
    ap.add_argument("--pose-model", required=True,
                    help="ViTPose weights DIRECTORY (HuggingFace format: config.json + "
                         "model.safetensors + preprocessor_config.json). Not a single file, and not "
                         "in this repository -- e.g. .../vitpose-plus-huge")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"output directory (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gpu", type=int, default=0, help="GPU id. One GPU, one process.")
    ap.add_argument("--scene", default="",
                    help="override which scene's videos and calibration to use")
    ap.add_argument("--zone-dir", default="",
                    help="override the zone polygons; default configs/<scene>/zone_bev")
    ap.add_argument("--start", type=int, default=None,
                    help="narrow the range, for a quick check; default is the whole scene (0)")
    ap.add_argument("--end", type=int, default=None,
                    help="narrow the range, for a quick check; default is the whole scene (8999)")
    return ap.parse_args()
