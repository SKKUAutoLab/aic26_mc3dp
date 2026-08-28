"""Refine a Track 1 submission against the DA3 point cloud, one frame at a time.

    python -m matching.core.box_refinement.refine_scene25 --input <submission>.txt --dataset <root>

For every frame in the submission:

    DA3 inference -> results.npz            (~67 MB)
      [scene 27]  -> zone mask
                  -> multi-view fuse  -> this frame's point cloud
                  -> drop results.npz
                  -> refine this frame's boxes against the full-density cloud
                  -> write this frame's rows, flush, then keep or drop the cloud

The causal state (previous refined pose, submission history, velocity windows) carries across
frames, so the loop is strictly sequential -- no chunks, no warmup. One GPU per scene.

The submission decides everything about WHAT to refine: its first row names the scene, and the
scene fixes the frame range and selects the tuned parameters and the zone polygons. The tuned
parameters are deliberately not flags -- these are the numbers the results were validated with.
"""
import argparse
import json
import os
import time

from .engine import orchestrator

# scene_id (column 0 of the submission) -> scene folder name. Only these two ship a profile.
SCENE_BY_ID = {25: "Warehouse_025", 27: "Warehouse_027"}

# Frame range is a property of the SCENE, not of the submission file. Reading it from the file was
# wrong: refinement runs on a file the tracker is still filling in, or has only partly written, and
# the range would then silently shrink to whatever happened to be in it. These match the videos
# (checked: Warehouse_025 has 9000 frames, Warehouse_027 has 1800).
SCENE_FRAMES = {25: (0, 8999), 27: (0, 1799)}
PKG = os.path.dirname(os.path.abspath(__file__))
# The locked per-scene profiles ship WITH the package: same numbers that produced the validated runs.
PROFILE_DIR = os.path.join(PKG, "profiles")
# So do the zone polygons. Scene 27's are a hand-refined set, DIFFERENT from both the dataset's own
# zone folders and the project's regions.zone_* configs -- keeping them here stops the three from
# being confused for one another.
ZONE_DIR = os.path.join(PKG, "zones")

# Voxel size for a KEPT .ply. It shrinks the file that is written; the refine always runs on the
# full-density cloud, so results never depend on it. Not a CLI flag -- keeping the clouds at all is
# a debugging move, and this is the size that was useful when debugging.
PLY_VOXEL = 0.02

# Everything a run produces goes under <output>/<scene>/ -- both the deliverable and the scratch.
# Writing the scratch next to the input would mean dirtying the previous stage's output directory.
DEFAULT_OUTPUT = os.path.join("output", "box_refinement")
# Directories that hold a shared DA3 cache elsewhere on this machine. --da3-output must never point
# at one of them, because the per-frame cleanup would delete it.
CACHE_ROOTS = [os.environ.get("DA3_CACHE_ROOT", "")] if os.environ.get("DA3_CACHE_ROOT") else []


def scene_of(path):
    """The scene id, from the first row of the submission. The file is not scanned any further."""
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            parts = line.split()
            if len(parts) >= 4:
                return int(float(parts[0]))
    raise SystemExit(
        f"{path} has no rows, so there is no scene id to read.\n"
        f"  Refinement runs after the tracking stage: give it the submission that stage wrote.")


def plan_run(*, submission, dataset, gpu=None, output="", split="test", keep_ply=True,
             start=None, end=None, profile="", zone_dir="", expect_scene_id=None,
             verbose=True) -> dict:
    """Resolve everything a run needs from just the submission path and the dataset root.

    The scene comes from the submission's first column; the tuned parameters and the zone polygons
    come from what is packaged for that scene. Returns the kwargs for
    :func:`orchestrator.iter_final`, so the CLI and :class:`~..refiner.BoxRefiner` build a run the
    same way and cannot drift apart.
    """
    inp = os.path.abspath(submission)
    if not os.path.isfile(inp):
        raise SystemExit(f"submission not found: {inp}")
    scene_id = scene_of(inp)                        # first row only; the range comes from the scene
    scene = SCENE_BY_ID.get(scene_id)
    if not scene:
        raise SystemExit(f"unknown scene_id {scene_id} (known: {sorted(SCENE_BY_ID)})")
    f_lo, f_hi = SCENE_FRAMES[scene_id]
    if expect_scene_id is not None and scene_id != expect_scene_id:
        raise SystemExit(
            f"this entry point refines scene {expect_scene_id} "
            f"({SCENE_BY_ID[expect_scene_id]}), but {os.path.basename(inp)} is scene {scene_id}.\n"
            f"  The two scenes use different algorithms -- use the other entry point.")

    prof_path = profile or os.path.join(PROFILE_DIR, f"{scene}.json")
    if not os.path.isfile(prof_path):
        raise SystemExit(f"profile not found: {prof_path}")
    cfg = json.load(open(prof_path, encoding="utf-8"))["config"]

    out_dir = os.path.abspath(os.path.join(output or DEFAULT_OUTPUT, scene))
    workdir = os.path.join(out_dir, "da3")     # npz + JPEGs + cloud, cleaned frame by frame
    for c in CACHE_ROOTS:                      # the guard that protects a shared DA3 cache
        if os.path.abspath(workdir) == os.path.abspath(c):
            raise SystemExit(f"output must not resolve onto a shared DA3 cache ({c}); "
                             "the per-frame cleanup would delete it.")
    os.makedirs(workdir, exist_ok=True)

    if cfg.get("apply_zone") and not zone_dir:
        packaged = os.path.join(ZONE_DIR, scene, cfg.get("zone_name") or "zone")
        if not os.path.isdir(packaged):
            raise SystemExit(
                f"{scene} needs zone '{cfg.get('zone_name')}' but it is not packaged at {packaged}."
                f"\n  Pass zone_dir=<dir of Camera_*.json>.")
        zone_dir = packaged

    start = f_lo if start is None else start
    end = f_hi if end is None else end
    gpu = cfg.get("gpu_id", 0) if gpu is None else gpu
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(inp))[0] + "_refined.txt")

    if verbose:
        print(f"scene      : {scene} (id {scene_id})")
        print(f"input      : {inp}")
        print(f"frames     : {start}..{end}  (sequential, no chunks, no warmup)")
        print(f"cloud      : DA3 {cfg['model']} @ {cfg['process_res']}, gpu {gpu}, "
              f"zone={cfg.get('zone_name') or 'off'}")
        print(f"per-frame  : npz + JPEGs deleted · cloud {'KEPT' if keep_ply else 'DELETED'}")
        if keep_ply:
            # Per-frame cloud size swings by two orders of magnitude between scenes (measured: ~5 MB
            # on scene 27, ~120 MB on scene 25), so one number here would be wrong for one of them.
            print(f"             {end - start + 1} clouds will be kept "
                  f"(~5-120 MB each, scene-dependent) -- --no-keep-ply keeps the disk flat")
        print(f"output     : {out_path}\n")

    q = cfg.get
    kwargs = dict(
        root=dataset, split=split, scene=scene, output_root=workdir,
        frame_start=start, frame_end=end, frame_step=1, warmup=0,
        sub_path_override=inp, out_path_override=out_path,
        model_name=cfg["model"], process_res=cfg["process_res"],
        ref_view=q("ref_view_strategy", "saddle_balanced"), use_ray_pose=q("use_ray_pose", False),
        gpu_id=gpu, apply_zone=q("apply_zone", False), zone_name=q("zone_name", ""),
        zone_dir=zone_dir,
        fuse_kwargs=q("fuse"), refine_method=q("refine_method", "meanshift"),
        person_params=q("person"), static_sizes=q("static_sizes"), static_poses=q("static_poses"),
        ht_params=q("height"), use_height_track=q("use_height_track", False),
        use_tracking=q("use_tracking", False), track_params=q("tracking"),
        use_prev_reseed=q("use_prev_reseed", False), prev_iou_thresh=q("prev_iou_thresh", 0.5),
        fix_person_height=q("fix_person_height", False), person_height=q("person_height", 1.75),
        person_w=q("person_w", 0.55), person_l=q("person_l", 0.40),
        use_yaw_smooth=q("use_yaw_smooth", True), yaw_alpha=q("yaw_alpha", 0.4),
        yaw_rate=q("yaw_rate", 0.08), clamp_vmax=q("clamp_vmax", 0.18),
        lowpass_floor=q("lowpass_floor", 0.25), motion_keep_fit=q("motion_keep_fit", 200),
        lost_near=q("lost_near", 0.45), lost_reanchor=q("lost_reanchor", 10),
        oriented_params=q("oriented_params"), s25_params=q("s25_params"),
        timestamp=time.strftime("%Y%m%d_%H%M%S"),
        cleanup_root=workdir, keep_ply=keep_ply, delete_npz=True,
        ply_voxel=PLY_VOXEL, ply_dir=os.path.join(out_dir, "cloud") if keep_ply else "",
        progress=(lambda m, fr: print(f"[{fr * 100:5.1f}%] {m}", flush=True)) if verbose
                 else (lambda m, fr: None))
    return {"scene": scene, "scene_id": scene_id, "start": start, "end": end,
            "out_path": out_path, "out_dir": out_dir, "workdir": workdir, "kwargs": kwargs}


def main(expect_scene_id=None, prog=None):
    """Shared CLI. The per-scene entry points pin `expect_scene_id`, so a file for the wrong scene
    is rejected instead of being silently refined with the other scene's algorithm."""
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True,
                    help="submission .txt from the previous stage (11 columns)")
    ap.add_argument("--dataset", "--root", dest="dataset", required=True,
                    help="dataset root: <dataset>/<split>/<scene>/{videos,calibration.json}")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"output directory (default: {DEFAULT_OUTPUT}). Everything this run "
                         "produces lands under <output>/<scene>/: the refined .txt, and a da3/ "
                         "subdirectory for the per-frame npz, JPEGs and cloud. Nothing is written "
                         "next to the input, so the previous stage's directory stays clean.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gpu", type=int, default=None, help="GPU id (default: the scene profile's)")
    ap.add_argument("--profile", default="", help="override the locked per-scene profile JSON")
    ap.add_argument("--zone-dir", default="",
                    help="directory of zone polygons (Camera_*.json). Default: the set packaged at "
                         "box_refinement/zones/<scene>/<zone_name>. Only scene 27 uses a zone.")
    ap.add_argument("--start", type=int, default=None,
                    help="narrow the range; default is the scene's first frame (0)")
    ap.add_argument("--end", type=int, default=None,
                    help="narrow the range; default is the scene's last frame "
                         "(Warehouse_025: 8999, Warehouse_027: 1799)")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--keep-ply", dest="keep_ply", action="store_true",
                   help="keep each frame's fused point cloud (default)")
    g.add_argument("--no-keep-ply", dest="keep_ply", action="store_false",
                   help="delete it once the frame is refined; disk then stays flat")
    ap.set_defaults(keep_ply=True)
    a = ap.parse_args()

    plan = plan_run(submission=a.input, dataset=a.dataset, gpu=a.gpu, output=a.output,
                    split=a.split, keep_ply=a.keep_ply, start=a.start, end=a.end,
                    profile=a.profile, zone_dir=a.zone_dir,
                    expect_scene_id=expect_scene_id)
    out_dir, out_path = plan["out_dir"], plan["out_path"]
    res = orchestrator.run_final(**plan["kwargs"])

    print(f"\nREFINED -> {out_path}")
    print(f"  {res['n_lines']} lines · {res['n_frames']} frames · {res['n_tracks']} tracks")
    print(f"  cleanup: {res['n_npz_deleted']} npz, {res['n_dyn_ply_deleted']} cloud ply deleted")
    cloud_dir = os.path.join(out_dir, "cloud")
    if os.path.isdir(cloud_dir):
        n = sum(os.path.getsize(os.path.join(cloud_dir, f)) for f in os.listdir(cloud_dir))
        print(f"  clouds : {len(os.listdir(cloud_dir))} files, {n / 1e9:.2f} GB -> {cloud_dir}")
    print(f"  time: {res['seconds_total']}s total · {res['seconds_per_frame']}s/frame "
          f"(DA3 {res['da3_per_frame']}s, refine {res['refine_per_frame']}s)")


