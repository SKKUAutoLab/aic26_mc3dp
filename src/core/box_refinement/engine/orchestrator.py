"""Final batch pipeline (tab "Final").

For a frame range and ONE config: per frame, REUSE the existing DA3+fuse
``estimated_fused.ply`` if present (DA3 npz + fuse are deterministic per scene/frame, so
trying different refine/height configs never re-runs them) else run DA3 -> fuse; refine
each frame with ONE method. Scene 25 refines every class while NovaCarter/Transporter
keep fixed size; scenes 26/27 still use their manual/static-object plans. The result is
a standard submission ``scene<id>_<start>_<end>_<timestamp>.txt``.
"""
from __future__ import annotations

import math
import os
import shutil
import time

import numpy as np
import open3d as o3d

from . import (dataset, fusion, heighttrack, oriented_fit, pipeline,
               submission, tracking)
from ..scenes import realworld as boxrefine        # scene 27: annealed mean-shift
from ..scenes import synthetic                     # scene 25: submission-anchored fits
from ..scenes.synthetic import (SCENE25_FIXED_CLASS_SIZES, SCENE25_FORKLIFT_SIZE,
                                SCENE25_ORIENTED_PARAMS, SCENE25_PARAMS, SCENE25_WALK_LENGTH)

STATIC_CLASSES = (2, 3)
# scene 25 — FIXED w,l,h per REFINED class (Forklift=1 is manual/stationary, sized below).

def _frame_export(output_root, split, scene, frame_id, model_san, res, zone, zone_name="zone"):
    rd = dataset.frame_run_dir(output_root, split, scene, frame_id)
    return os.path.join(rd, f"{model_san}_res_{res}") + pipeline.zone_suffix(zone, zone_name)


def _fmt_eta(sec):
    sec = int(round(sec))
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _under(path, root):
    """True only if `path` lives inside `root` — the guard that makes per-frame cleanup safe.

    Cleanup NEVER deletes outside the caller's own workdir, so a run pointed at the shared
    DA3 cache (output/aicity2026) can never wipe it.
    """
    if not path or not root:
        return False
    p, r = os.path.abspath(path), os.path.abspath(root)
    return p == r or p.startswith(r + os.sep)


def _rm(path, root):
    if _under(path, root) and os.path.isfile(path):
        try:
            os.remove(path)
            return True
        except OSError:
            pass
    return False


def _rmtree(path, root):
    if _under(path, root) and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _prune(leaf, root):
    """Drop `leaf`'s empty subtree, then every ancestor that just went empty. Stops at `root`."""
    if not _under(leaf, root):
        return
    for cur, _d, _f in os.walk(leaf, topdown=False):   # bottom-up: children are tried before parents
        try:
            os.rmdir(cur)                              # succeeds only while empty
        except OSError:
            pass
    d = os.path.abspath(leaf)
    while _under(d, root) and os.path.abspath(d) != os.path.abspath(root):
        try:
            os.rmdir(d)
        except OSError:
            return
        d = os.path.dirname(d)


def _shrink_ply(path, voxel):
    """Voxel-downsample a KEPT .ply in place. Runs AFTER the frame is refined, so it only ever
    shrinks the on-disk artifact — the refine itself always saw the FULL-density cloud."""
    if voxel <= 0 or not os.path.isfile(path):
        return
    try:
        pc = o3d.io.read_point_cloud(path).voxel_down_sample(float(voxel))
        o3d.io.write_point_cloud(path, pc)
    except Exception:  # noqa: BLE001
        pass


def _sub_line(scene_id, class_id, object_id, frame_id, x, y, z, w, l, h, yaw):
    return (f"{int(scene_id)} {int(class_id)} {int(object_id)} {int(frame_id)} "
            f"{x:.4f} {y:.4f} {z:.4f} {w:.4f} {l:.4f} {h:.4f} {yaw:.5f}")


def _scene26(scene):
    return scene.endswith("026") or "_026" in (scene or "")


def _scene25(scene):
    return scene.endswith("025") or "_025" in (scene or "")


def _track_key(box):
    return f"{int(box.get('class_id', -1))}:{int(box.get('object_id', -1))}"


def _final_cfg(scene, static_sizes=None):
    """Per-scene plan for the hard-fix (manual) vs auto-refined objects.

    scene 27: NovaCarter(2)+Transporter(3) = manual (by class); persons refined.
    scene 26: Forklift id 11 = manual (by object); NovaCarter(2) DROPPED entirely; persons +
              Forklift id 10 refined (yaw + position, FIXED forklift size).
    scene 25 (synthetic, ALL objects MOVE): NO manual placement — every class goes through the ONE
              common refine; NovaCarter(2)+Transporter(3) keep a FIXED size (from static_sizes),
              everyone else estimated. Position + yaw refined for all."""
    if _scene25(scene):
        # SAME algorithm family as scene 27 (multi_candidate continuity + online smoothing), but a
        # TIGHTER refine window (objects sit close to their submission points) and FIXED size per class.
        # Forklift (class 1 — the only stationary object) is HAND-PLACED like Nova/Trans in scene 27;
        # every other class (0,2,3,4,5) is refined at its fixed size. SELF-CONTAINED branch: it shares
        # NONE of scene 27's config and the scene-27 branch below is untouched.
        cs = {k: list(v) for k, v in SCENE25_FIXED_CLASS_SIZES.items()}   # LOCKED (no UI override)
        return {"exclude_classes": [1], "exclude_objs": [],
                "person_class_sizes": cs, "person_yaw": False,
                "manual": [{"cls": 1}], "manual_sizes": {1: list(SCENE25_FORKLIFT_SIZE)},
                "tight_footprint": False, "multi_candidate": True,
                # NovaCarter(2)+Transporter(3) use the SEPARATE oriented-box fit (raise search → drop
                # floor → rotate fixed footprint to best coverage, biased to prev-yaw → place at z=0).
                # Person(0)+Fourier(4)+Agility(5) keep the default person-like refine. Per-class params:
                "oriented_classes": [2, 3],
                "oriented_params": dict(SCENE25_ORIENTED_PARAMS),
                # Submission-anchored 2-stage refine for ALL scene-25 classes (scenes/synthetic_fit.py).
                "sub_anchor": True, "s25_classes": [0, 2, 3, 4, 5], "s25_params": dict(SCENE25_PARAMS),
                "algo": {"margin": 0.5, "bandwidth": 0.4, "min_fit": 500,
                         "clamp_vmax": 0.18, "lowpass_floor": 0.25, "motion_keep_fit": 500,  # #1 less lag
                         "jump_reset_dist": 0.2, "jump_reset_frames": 10,  # FOURIER(4)+Person(0): box > 0.2m from sub > 10 frames → SNAP back to submission (un-strands a frozen re-localize). Agility(5) is EXCLUDED — its hold-coast state machine owns its own snap/timeout.
                         "lost_near": 0.45, "lost_reanchor": 10,
                         "subcrop_margin": 1.0,   # crop cloud to 1.5m around objects (≫ person/vehicle ROI → same result, far fewer pts)
                         "use_yaw_smooth": True, "yaw_smooth_all": True,   # mod-π yaw EWMA on EVERY class (not just person)
                         "yaw_alpha": 0.4, "yaw_rate": 0.15,   # v9 stable base (still humans)
                         # HUMAN-shaped (0,4,5) while MOVING: a walking heading is already smooth, so chase it
                         # with an EVEN SMALLER step → the box yaw barely jitters frame-to-frame (extra stable).
                         "yaw_alpha_move": 0.25, "yaw_rate_move": 0.07,
                         # VEHICLES (2,3): yaw = point-fit (cover-fit search), rate-limited so the box tracks
                         # the driving-arc turn but never spins on noise (≈2.9°/f cap, ~GT-rate + headroom).
                         "yaw_alpha_veh": 0.5, "yaw_rate_veh": 0.05}}
    if _scene26(scene):
        return {"exclude_classes": [2], "exclude_objs": [11],
                "person_class_sizes": {1: [1.214, 2.313, 2.155]}, "person_yaw": True,
                "manual": [{"cls": 1, "obj": 11}], "tight_footprint": False, "multi_candidate": False}
    # scene 27 (+ other real scenes): persons keep the OLD size estimation (no radius cap) but use
    # the multi-candidate wide→narrow search (closest valid mode, dedup re-selects alternatives).
    # ``algo`` = the LOCKED per-scene online-smoothing profile (clamp + adaptive low-pass + lost-counter
    # + anti-spin yaw, all gated on multi_candidate so they touch ONLY this scene). It OVERRIDES the
    # run_final defaults, so scene 27 reproduces EXACTLY regardless of UI/global defaults — and editing
    # another scene's branch can never change it. Different scenes may define a different ``algo``.
    return {"exclude_classes": [2, 3], "exclude_objs": [],
            "person_class_sizes": {}, "person_yaw": False,
            "manual": [{"cls": 2}, {"cls": 3}], "tight_footprint": False, "multi_candidate": True,
            "algo": {"clamp_vmax": 0.18, "lowpass_floor": 0.25, "motion_keep_fit": 200,
                     "lost_near": 0.45, "lost_reanchor": 10,
                     "use_yaw_smooth": True, "yaw_alpha": 0.4, "yaw_rate": 0.08}}


def _static_boxes(sub_path, frame_id, sizes, poses, floor_z, manual):
    """Hard-fix objects for this frame: keep the submission object_id but override
    position+yaw with the manual pose and size with the fixed size (keyed by object_id when the
    spec is object-specific, else by class_id). Submission boxes not matching any spec are skipped
    (handled by the person refine instead)."""
    out = []
    for b in submission.get_boxes(sub_path, frame_id)["boxes"]:
        cid, oid = b["class_id"], b["object_id"]
        spec = None
        for m in manual:
            if (m.get("obj") is not None and oid == m["obj"]) or (m.get("obj") is None and cid == m["cls"]):
                spec = m
                break
        if spec is None:
            continue
        key = spec.get("obj") if spec.get("obj") is not None else spec["cls"]
        sz = (sizes or {}).get(key) or (sizes or {}).get(str(key)) or b["size"]
        w, l, h = float(sz[0]), float(sz[1]), float(sz[2])
        pose = (poses or {}).get(key) or (poses or {}).get(str(key))
        if pose and (pose[0] or pose[1]):
            x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])   # yaw in radians
        else:
            x, y, yaw = b["center"][0], b["center"][1], b["yaw"]
        out.append({"class_id": cid, "object_id": oid, "x": x, "y": y,
                    "z": floor_z + h / 2.0, "w": w, "l": l, "h": h, "yaw": yaw})
    return out


def _prior_refine_boxes(sub_path, frame_id, cfg, floor_z):
    """Fallback boxes when point-cloud refine fails: keep submission pose, apply fixed sizes."""
    out = []
    exc_cls = set(cfg.get("exclude_classes") or [])
    exc_obj = set(int(x) for x in (cfg.get("exclude_objs") or []))
    class_sizes = cfg.get("person_class_sizes") or {}
    for b in submission.get_boxes(sub_path, frame_id)["boxes"]:
        cid, oid = int(b["class_id"]), int(b["object_id"])
        if cid in exc_cls or oid in exc_obj:
            continue
        w, l, h = class_sizes.get(cid, b["size"])
        w, l, h = float(w), float(l), float(h)
        z = floor_z + h / 2.0 if cid in class_sizes else float(b["center"][2])
        out.append({
            "class_id": cid, "object_id": oid,
            "world_xyz": [float(b["center"][0]), float(b["center"][1]), z],
            "size": [w, l, h], "yaw": float(b["yaw"]),
            "z_top": None, "foot": None, "n_points": 0, "source": "prior_fallback",
        })
    return out


def _oriented_boxes(P, sub_path, frame_id, classes, params, sizes, prev_refined, floor_z,
                    exclude_objs):
    """Scene-25 NovaCarter/Transporter ONLY (separate algorithm, see oriented_fit): fit the FIXED-size
    box by RAISING the search above the floated DA3 points, dropping the floor layer, and ROTATING the
    footprint to maximum point coverage biased toward the prev-refined yaw; the box drops back to z=0.
    Returns boxes in the same dict shape boxrefine.refine emits, so they flow through the existing
    position-smoothing / prev-tracking / write path. The person yaw-smoother skips them (class != 0).
    """
    classes = set(int(c) for c in classes)
    exo = set(int(x) for x in (exclude_objs or []))
    out = []
    for b in submission.get_boxes(sub_path, frame_id)["boxes"]:
        cid = int(b["class_id"])
        if cid not in classes:
            continue
        oid = int(b.get("object_id", -1))
        if oid in exo:
            continue
        sz = (sizes or {}).get(cid) or (sizes or {}).get(str(cid)) or b["size"]
        w, l, h = float(sz[0]), float(sz[1]), float(sz[2])
        sx, sy, scz = b["center"]
        pv = prev_refined.get(f"{cid}:{oid}")
        if pv is not None:
            seed, ref_yaw = (pv[0], pv[1]), (float(pv[4]) if len(pv) > 4 else float(b["yaw"]))
        else:
            seed, ref_yaw = (float(sx), float(sy)), float(b["yaw"])
        kw = dict((params or {}).get(cid) or (params or {}).get(str(cid)) or {})
        r = oriented_fit.fit_box(P, seed, ref_yaw, w, l, h, floor_z=floor_z, **kw)
        if r is not None:
            x, y, yaw, npts = r["x"], r["y"], r["yaw"], r["fit"]
        elif pv is not None:                      # lost this frame → hold prev (continuity)
            x, y, yaw, npts = pv[0], pv[1], ref_yaw, 0
        else:                                     # first frame & lost → submission
            x, y, yaw, npts = float(sx), float(sy), float(b["yaw"]), 0
        wxyz = [float(x), float(y), float(floor_z)]
        out.append({
            "object_id": oid, "class_id": cid,
            "orig_xyz": [round(float(sx), 3), round(float(sy), 3), round(float(scz), 3)],
            "world_xyz": wxyz, "size": [w, l, h],
            "yaw": round(float(yaw), 4), "yaw_deg": round(float(np.degrees(yaw)), 1),
            "yaw_found": True, "shift_m": 0.0, "n_points": int(npts),
            "source": "oriented", "z_top": None, "foot": None,
        })
    return out


def _frame_lines(f, frame_persons, frame_statics, scene_id, floor_z, pos_tf, h_tf, yaw_tf,
                 fix_person_height, person_height, person_w, person_l):
    """The submission rows for ONE frame, exactly as the end-of-run writer produces them.

    Split out so the rows can also be flushed to disk as each frame finishes: a 9000-frame scene is
    a ~60-hour run, and buffering every row in memory until the end means a crash at frame 8000
    throws the whole thing away.
    """
    out, n_h_fixed = [], 0
    for b in frame_persons[f]:
        tid = _track_key(b)
        x, y, _ = b["world_xyz"]
        x, y = pos_tf.get((tid, f), (x, y))   # OC-SORT-stabilised x,y (if on)
        w, l, h0 = b["size"]
        h = h_tf.get((tid, f), h0)
        if fix_person_height and int(b["class_id"]) == 0:   # force every person to a FIXED size
            h, w, l = float(person_height), float(person_w), float(person_l)
        if abs(h - h0) > 1e-3:
            n_h_fixed += 1
        yaw = yaw_tf.get((tid, f), b["yaw"])   # rate-limited stabilised yaw (anti-spin) if on
        out.append(_sub_line(scene_id, b["class_id"], b["object_id"], f,
                             x, y, floor_z + h / 2.0, w, l, h, yaw))
    for b in frame_statics[f]:
        out.append(_sub_line(scene_id, b["class_id"], b["object_id"], f,
                             b["x"], b["y"], b["z"], b["w"], b["l"], b["h"], b["yaw"]))
    return out, n_h_fixed


def run_final(**kwargs) -> dict:
    """Refine every frame in one call and return the run summary. Drains :func:`iter_final`."""
    gen = iter_final(**kwargs)
    while True:
        try:
            next(gen)
        except StopIteration as done:
            return done.value


def iter_final(*, root, split, scene, output_root, frame_start, frame_end, frame_step, warmup=0,
              model_name, process_res, ref_view, use_ray_pose, gpu_id, apply_zone,
              fuse_kwargs, refine_method, person_params, static_sizes, static_poses,
              ht_params, timestamp,
              zone_name="zone", zone_dir="", use_height_track=True, use_tracking=False,
              track_params=None,
              use_prev_reseed=False, prev_iou_thresh=0.5,
              fix_person_height=False, person_height=1.75, person_w=0.55, person_l=0.40,
              use_yaw_smooth=True, yaw_alpha=0.4, yaw_rate=0.08,
              clamp_vmax=0.18, lowpass_floor=0.25, motion_keep_fit=200,
              lost_near=0.45, lost_reanchor=10, oriented_params=None, s25_params=None,
              sub_path_override="", cleanup_root="", keep_ply=True, delete_npz=False, ply_voxel=0.0,
              ply_dir="", out_path_override="",
              progress=lambda m, f: None) -> dict:
    info = dataset.scan_scene(root, split, scene)
    nfr = info.get("num_frames") or 0
    end = min(frame_end, nfr - 1) if nfr else frame_end
    # WARMUP: refine `warmup` frames BEFORE frame_start to build the online per-track state (sub_hist, prev_refined,
    # agi_state, yaw_hist, velocity/straight/stationary windows) so a PARALLEL-CHUNK start behaves like a continuous
    # run — kills the chunk-boundary seam. These warmup frames are processed but NOT written (write loop skips
    # f < frame_start). Assumes frame_step == 1 (the full build uses step 1).
    proc_start = max(0, frame_start - int(warmup))
    frames = list(range(proc_start, end + 1, max(1, frame_step)))
    if not frames:
        raise ValueError("empty frame range")
    sub_path = (sub_path_override or os.environ.get("SUB_PATH_OVERRIDE")
                or submission.find_path_for_scene(root, scene))
    if not sub_path:
        raise FileNotFoundError(f"no submission file for scene {scene}")
    scene_id = submission._scene_num_from_name(scene) or 0
    model_san = pipeline.sanitize_model_name(model_name)
    floor_z = float((person_params or {}).get("floor_z", 0.0))
    _cfg = _final_cfg(scene, static_sizes)   # per-scene manual vs auto-refined plan
    # LOCKED per-scene online-smoothing profile (in code) overrides the run defaults → exact reproduce,
    # isolated per scene. A scene without an "algo" block just uses the passed/global defaults.
    _algo = _cfg.get("algo", {})
    clamp_vmax = float(_algo.get("clamp_vmax", clamp_vmax))
    lowpass_floor = float(_algo.get("lowpass_floor", lowpass_floor))
    motion_keep_fit = int(_algo.get("motion_keep_fit", motion_keep_fit))
    lost_near = float(_algo.get("lost_near", lost_near))
    lost_reanchor = int(_algo.get("lost_reanchor", lost_reanchor))
    use_yaw_smooth = bool(_algo.get("use_yaw_smooth", use_yaw_smooth))
    yaw_alpha = float(_algo.get("yaw_alpha", yaw_alpha))
    yaw_rate = float(_algo.get("yaw_rate", yaw_rate))
    yaw_alpha_move = float(_algo.get("yaw_alpha_move", yaw_alpha))   # MOVING humans: smaller step = extra stable
    yaw_rate_move = float(_algo.get("yaw_rate_move", yaw_rate))      # (fallback = v9 base)
    yaw_alpha_veh = float(_algo.get("yaw_alpha_veh", yaw_alpha))     # VEHICLES: rate-limited to GT ~1°/f
    yaw_rate_veh = float(_algo.get("yaw_rate_veh", yaw_rate))
    # STANDING human shoulder-yaw smoother (scene-25): confidence gate + consistency-adaptive rate-limit.
    sy_conf_min = float(_algo.get("still_yaw_conf_min", 1.4))   # min PCA anisotropy to trust a measurement
    sy_win = int(_algo.get("still_yaw_hist", 7))                # history window for the circular mean / trend
    sy_rate_lo = float(_algo.get("still_yaw_rate_lo", 2.0))     # deg/f slew when measurements JITTER (oscillate)
    sy_rate_hi = float(_algo.get("still_yaw_rate_hi", 10.0))    # deg/f slew when a CONSISTENT real turn is seen
    sy_anchor_w = float(_algo.get("still_yaw_anchor_w", 0.20))  # soft pull of the target toward submission yaw
    sy_max_dev = math.radians(float(_algo.get("still_yaw_max_dev_deg", 60.0)))  # HARD cap on |yaw − submission|
    # Cloud subcrop: 0 = off (scene 27 / others = untouched). Only a scene whose ``algo`` sets it
    # (scene 25 = 1.5) crops the cloud to its objects before refine — same result, far fewer points.
    subcrop_margin = float(_algo.get("subcrop_margin", 0.0))
    # Oriented classes: empty for scene 27 / others (untouched). Scene 25 = [2,3] (Nova/Transporter)
    # → handled by the separate oriented_fit, EXCLUDED from the person-like refine below.
    _oriented = list(_cfg.get("oriented_classes", []))
    _oriented_params = {int(k): dict(v) for k, v in _cfg.get("oriented_params", {}).items()}
    for _k, _ov in (oriented_params or {}).items():     # UI/profile per-class per-key override
        _ck = int(_k)
        _oriented_params.setdefault(_ck, {}).update({_pk: _pv for _pk, _pv in (_ov or {}).items()})
    if _cfg.get("s25_params") is not None and s25_params:   # UI/profile common 2-stage override
        _cfg["s25_params"] = {**_cfg["s25_params"], **{k: v for k, v in s25_params.items()}}
    if _scene25(scene):                        # scene-25-ONLY tweaks — scene 27 never enters this block
        person_params = dict(person_params or {})
        person_params["margin"] = float(_algo.get("margin", person_params.get("margin", 0.5)))
        person_params["bandwidth"] = float(_algo.get("bandwidth", person_params.get("bandwidth", 0.4)))
        person_params["min_fit"] = int(_algo.get("min_fit", person_params.get("min_fit", 500)))
        static_sizes = dict(static_sizes or {})            # give the HAND-PLACED forklift its fixed size
        for _k, _v in _cfg.get("manual_sizes", {}).items():
            static_sizes.setdefault(str(_k), list(_v))

    obs_by_track: dict = {}
    pos_by_track: dict = {}
    prev_refined: dict = {}                # last refined [x,y,w,l,yaw] per class:object track
    prev2_refined: dict = {}               # refined position 2 frames ago → online velocity/motion model
    jcnt: dict = {}                        # consecutive frames a track stayed > jump_reset_dist from its submission
    jump_reset_dist = float(_algo.get("jump_reset_dist", 0.2))   # box > this from submission counts toward the reset
    jump_reset_frames = int(_algo.get("jump_reset_frames", 10))  # stuck far > this many frames → snap back to submission
    agi_state: dict = {}                   # per-Agility(5)-track hold-coast-return-jump state (submission-only owner)
    vel_vec: dict = {}                     # EWMA velocity VECTOR per track → adaptive low-pass gain
    sub_hist: dict = {}                    # last ~60 SUBMISSION xy per track → DRIFT-FREE vehicle heading
    ref_hist: dict = {}                    # last ~32 non-jump submission xy per track → heading trail (jump-excluded)
    last_seen: dict = {}                    # last frame each track was present → jump continuity guard (adjacent only)
    lost_count: dict = {}                  # consecutive frames a track's own cluster is missing near prev
    yaw_smooth: dict = {}                  # rate-limited circular-EWMA yaw per track (anti-spin)
    yaw_hist: dict = {}                    # STANDING human: recent CONFIDENT shoulder-PCA yaws (median/trend)
    yaw_tf: dict = {}                      # stabilised yaw per (track, frame) → used at write time
    # Filled AFTER the frame loop, and only when their feature is on (OC-SORT / height-track). They
    # are declared here because the streaming writer reads them inside the loop; streaming is only
    # enabled when both features are off, so what it reads is these empty dicts.
    pos_tf: dict = {}
    h_tf: dict = {}
    frame_persons: dict = {}
    frame_statics: dict = {}
    fuse_kwargs = dict(fuse_kwargs or {})
    # Rows are flushed as each frame finishes, so a crash late in a 60-hour run does not throw away
    # everything before it. Only possible when nothing needs the WHOLE run to decide a row:
    # OC-SORT (pos_tf) and height-track (h_tf) are both fitted after the frame loop, so with either
    # of them on we fall back to writing once at the end. yaw_tf is filled inside the loop already.
    stream_rows = not use_tracking and not use_height_track
    stream_fp = None
    lines, n_h_fixed = [], 0

    if out_path_override:                      # write straight to the caller's file: no staging
        out_path = out_path_override           # copy, so a crash still leaves the rows in place
        name = os.path.basename(out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    else:
        sub_dir = os.path.join(output_root, "final", scene)
        os.makedirs(sub_dir, exist_ok=True)
        name = f"scene{scene_id}_{frame_start}_{frame_end}_{timestamp}.txt"
        out_path = os.path.join(sub_dir, name)
    if stream_rows:
        stream_fp = open(out_path, "w", encoding="utf-8")

    reused, refine_failed, n_full_ply_deleted, t_total, t_da3, t_refine = 0, 0, 0, time.perf_counter(), 0.0, 0.0
    n_npz_deleted = n_dyn_ply_deleted = 0                      # per-frame cleanup
    for k, f in enumerate(frames):
        if k > 0:                                        # running avg s/frame + ETA
            avg = (time.perf_counter() - t_total) / k
            msg = (f"frame {f} ({k + 1}/{len(frames)}) · {avg:.2f}s/frame · "
                   f"còn ~{_fmt_eta((len(frames) - k) * avg)}")
        else:
            msg = f"frame {f} (1/{len(frames)})"
        progress(msg, 0.03 + 0.84 * k / len(frames))
        export = _frame_export(output_root, split, scene, f, model_san, process_res,
                               apply_zone, zone_name)
        npz = os.path.join(export, "exports", "npz", "results.npz")
        ply = os.path.join(export, "pointcloud", "estimated_fused.ply")
        # Every fit runs on the frame's fused cloud, so the frame is ready once that ply exists.
        need = not os.path.isfile(ply)
        if not need:
            reused += 1
        else:
            t0 = time.perf_counter()
            # The DA3 npz is zone-INDEPENDENT (zone mask is applied AFTER inference). If the
            # full export's npz already exists, switching zone only needs a re-mask + re-fuse —
            # NOT a full DA3 re-run. Huge save when re-running a scene under a new zone version.
            full_export = _frame_export(output_root, split, scene, f, model_san, process_res, False)
            full_npz = os.path.join(full_export, "exports", "npz", "results.npz")
            if apply_zone and not os.path.isfile(npz) and os.path.isfile(full_npz):
                from . import zonemask
                zonemask.apply(full_export, os.path.join(root, split, scene), zone_name,
                               zone_dir=zone_dir)
            elif not os.path.isfile(npz):
                res = pipeline.run_inference(
                    root=root, split=split, scene=scene, frame_id=f, cameras="all",
                    model_name=model_name, process_res=process_res, export_format="npz",
                    gpu_id=gpu_id, output_root=output_root, ref_view_strategy=ref_view,
                    use_ray_pose=use_ray_pose, apply_zone=apply_zone, zone_name=zone_name,
                    zone_dir=zone_dir)
                zinfo = res.get("zone")
                export = (zinfo["zone_export_dir"] if apply_zone and zinfo and not zinfo.get("error")
                          else res["export_dir"])
                npz = os.path.join(export, "exports", "npz", "results.npz")
                ply = os.path.join(export, "pointcloud", "estimated_fused.ply")
            if not os.path.isfile(ply):
                fusion.fuse(export_dir=export, **fuse_kwargs)
            t_da3 += time.perf_counter() - t0

        # The frame's cloud IS the fused cloud. Naming it here keeps ONE cleanup path below and
        # gives the scene-25 fit its shared point array (P_shared), which is loaded from this path.
        cloud_path = ply

        tr = time.perf_counter()
        # ONLINE motion model: predict each track's position THIS frame = prev + (prev − prev2).
        # Only uses our own PAST refined frames (no future, no submission temporal) → keeps the
        # tracker online (+10% bonus). The refine uses it to stay on the track's own cluster.
        motion_pred = {}
        if _cfg.get("multi_candidate"):
            for tk, pv in prev_refined.items():
                p2 = prev2_refined.get(tk)
                if p2 is not None:
                    motion_pred[tk] = [2.0 * pv[0] - p2[0], 2.0 * pv[1] - p2[1]]
        # scene-25 oriented (Nova/Transporter): load the cloud ONCE here and share it with the refine
        # AND the oriented fit, so the frame's .ply is parsed a single time.
        P_shared = None
        if (_oriented or _cfg.get("sub_anchor")) and cloud_path and os.path.isfile(cloud_path):
            try:
                P_shared = np.asarray(o3d.io.read_point_cloud(cloud_path).points, dtype=np.float64)
            except Exception:  # noqa: BLE001
                P_shared = None
        _sp = _cfg.get("s25_params", {})
        # HUMAN jump-handling: FOURIER(4) is stateless per-frame in _scene25_boxes (foot-BEV re-locate,
        # scene25_refine.fit_human_feet); AGILITY(5) is the STATEFUL hold-coast-return-jump owner (_agi_step,
        # applied to the returned boxes below); Person(0) always FOLLOWs.
        # NOVACARTER (class 2 ONLY): 30-frame MEDIAN per-frame submission speed → its velocity state machine
        # (MOVING >= nova_speed_thr → submission verbatim; braked/STOPPED < thr → BEV fit). Transporter uses the
        # net-based `veh_straight` gate (below), not a speed, so no separate veh_speed dict is needed.
        nova_win = int(_sp.get("nova_speed_win", 30))
        nova_speed = {}
        for tk in set(sub_hist):
            if tk.split(":")[0] != "2":
                continue
            h = sub_hist.get(tk)
            if h and len(h) >= 3:
                ww = min(nova_win, len(h) - 1)
                steps = [math.hypot(h[-k][0] - h[-k - 1][0], h[-k][1] - h[-k - 1][1]) for k in range(1, ww + 1)]
                nova_speed[tk] = float(np.median(steps))
            else:
                nova_speed[tk] = 0.0
        # VEHICLE going STRAIGHT (classes 2,3): net / total displacement over veh_str_win frames. ≈1 = a straight
        # line; < veh_str_smin = a curve (turn) or jitter/back-and-forth; tiny net = stopped. ONLY a STRAIGHT
        # MOVING vehicle trusts the (clean) submission yaw — turning & stopped both solve yaw from the cloud (PCA).
        # Robust: jitter inflates `total` but not `net` → its ratio drops → never false-positives "straight".
        vstr_win = int(_sp.get("veh_str_win", 8))
        vstr_smin_raw = _sp.get("veh_str_smin", 0.9)        # PER-CLASS dict {2:..,3:..} or a scalar
        vstr_dmin_raw = _sp.get("veh_str_dmin", 0.08)       # PER-CLASS dict {2:..,3:..} or a scalar

        def _per_cls(raw, cls, default):
            if isinstance(raw, dict):
                return float(raw.get(cls, raw.get(str(cls), default)))
            return float(raw)
        veh_straight = {}
        for tk in set(sub_hist):
            c0 = tk.split(":")[0]
            if c0 not in ("2", "3"):
                continue
            h = sub_hist.get(tk)
            if h and len(h) >= vstr_win + 1:
                pts = h[-(vstr_win + 1):]
                net = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
                total = sum(math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1])
                            for k in range(1, len(pts)))
                veh_straight[tk] = bool(net > _per_cls(vstr_dmin_raw, int(c0), 0.08)
                                        and total > 1e-6
                                        and net / total > _per_cls(vstr_smin_raw, int(c0), 0.9))
            else:
                veh_straight[tk] = False
        # NEAR-STATIONARY from the NON-JUMP trail (ref_hist = submission xy on FOLLOW frames, FROZEN during a
        # jump). Using the clean trail (not raw sub_hist) so a GLITCH no longer flips a standing person to
        # "moving" (which sent it into the cloud-search wander). On non-jump frames refined==submission, so it
        # is still un-lagged. A brand-new track (no history) defaults to still so the start frame is corrected.
        st_win = int(_sp.get("still_win", 8))
        st_net = float(_sp.get("still_net", 0.08))
        # PER-CLASS still gate. Person(0) + default: NET displacement over still_win vs `still_net`.
        # Fourier(4)/Agility(5): they walk slow & curved, so NET (≈0 on a curve) only caught 80% of their
        # moving frames → switch to a robust MEDIAN per-frame step over still_win vs `still_mstep_slow`
        # (0.005) → recall 97-98%, ~0% false-positive on a true stand-still.
        st_mstep_cls = _sp.get("still_mstep_slow", {4: 0.005, 5: 0.005})
        stationary = {}
        for tk, h in ref_hist.items():
            n = min(st_win, len(h))
            if n < 2:
                stationary[tk] = True
                continue
            cls = int(tk.split(":")[0])
            mstep = st_mstep_cls.get(cls, st_mstep_cls.get(str(cls)))
            if mstep is not None:                            # robust MEDIAN-STEP (Fourier/Agility)
                steps = [math.hypot(h[-k][0] - h[-k - 1][0], h[-k][1] - h[-k - 1][1]) for k in range(1, n)]
                stationary[tk] = float(np.median(steps)) < float(mstep)
            else:                                            # NET displacement (Person + default)
                stationary[tk] = math.hypot(h[-1][0] - h[-n][0], h[-1][1] - h[-n][1]) < st_net
        # MOVING-jump search cone: heading = direction over the last head_win NON-JUMP submission positions
        # (ref_hist skips jump frames, so a glitch/excursion never corrupts it — it FREEZES at the last clean
        # walking direction through the jump). Only set when net motion over the window ≥ head_min (clearly
        # walking); a ~stationary run gives heading None → no cone, pure strong-near hold-search in fit_one.
        hd_win = int(_sp.get("head_win", 10))
        hd_min = float(_sp.get("head_min", 0.10))
        heading = {}
        for tk, rh in ref_hist.items():
            if len(rh) >= 2:
                n = min(hd_win, len(rh))
                dx, dy = rh[-1][0] - rh[-n][0], rh[-1][1] - rh[-n][1]
                if math.hypot(dx, dy) >= hd_min:
                    heading[tk] = math.atan2(dy, dx)
        # (Agility(5) glitch handling no longer uses a coast-velocity vector here — it is the stateful
        # hold-coast-return-jump owner _agi_step applied AFTER _scene25_boxes, driven purely by the submission.)
        try:
            if _cfg.get("sub_anchor") and P_shared is not None:
                # scene-25 NEW: submission-anchored 2-stage refine for ALL classes (scene 27 never here)
                persons = synthetic._scene25_boxes(P_shared, sub_path, f, _cfg["s25_classes"],
                                         _cfg["person_class_sizes"], _cfg["s25_params"],
                                         _oriented_params, prev_refined, floor_z, _cfg["exclude_objs"],
                                         last_seen=last_seen, stationary=stationary,
                                         veh_straight=veh_straight, heading=heading,
                                         nova_speed=nova_speed)
            else:
                persons = boxrefine.refine(
                    export_dir=export, root=root, scene=scene, frame_id=f, method=refine_method,
                    exclude_class_ids=list(_cfg["exclude_classes"]) + _oriented,   # refine SKIPS oriented
                    exclude_object_ids=_cfg["exclude_objs"],
                    class_sizes=_cfg["person_class_sizes"], estimate_yaw=_cfg["person_yaw"],
                    cloud_path=cloud_path,
                    prev_boxes=(prev_refined if (use_prev_reseed or _cfg.get("multi_candidate")) else None),
                    prev_iou_thresh=prev_iou_thresh, use_prev_reseed=use_prev_reseed,
                    tight_footprint=_cfg.get("tight_footprint", False),
                    multi_candidate=_cfg.get("multi_candidate", False),
                    motion_pred=motion_pred, motion_keep_fit=motion_keep_fit,
                    subcrop_margin=subcrop_margin,    # 0 for scene 27/others → no crop; scene 25 → 1.5
                    cloud_pts=P_shared,               # None for scene 27/others → refine loads it itself
                    **(person_params or {}))["boxes"]
                if _oriented and P_shared is not None:        # Nova/Transporter via the separate fit
                    persons += _oriented_boxes(P_shared, sub_path, f, _oriented, _oriented_params,
                                               _cfg["person_class_sizes"], prev_refined, floor_z,
                                               _cfg["exclude_objs"])
        except Exception as exc:  # noqa: BLE001
            refine_failed += 1
            progress(f"refine lỗi frame {f}: {exc}; dùng prior submission", 0.03 + 0.84 * (k + 0.5) / len(frames))
            persons = _prior_refine_boxes(sub_path, f, _cfg, floor_z)
        t_refine += time.perf_counter() - tr
        # AGILITY(5) hold-coast-return-jump OWNER (submission-ONLY, stateful, NO cloud). _scene25_boxes/fit_one
        # returned the FOLLOW submission x,y for every Agility box (orig_xyz = the raw submission, immune to the
        # anti-jump revert inside _scene25_boxes); here the per-track state machine OVERRIDES that x,y with its
        # HOLD/COAST/GLIDE output. Runs BEFORE the human `continue` below so the jump_reset SNAP no longer fires
        # on Agility (it now owns its own snap/timeout); Fourier(4)+Person(0) keep the reset.
        if _cfg.get("sub_anchor"):
            for b in persons:
                if int(b["class_id"]) != 5:
                    continue
                tk = _track_key(b)
                sub_xy = (float(b["orig_xyz"][0]), float(b["orig_xyz"][1]))   # the raw submission this frame
                rx, ry = synthetic._agi_step(agi_state.setdefault(tk, {}), f, sub_xy, _sp)
                b["world_xyz"] = [rx, ry, b["world_xyz"][2]]
        # ONLINE smoothness guarantee: a real person at 30fps can't teleport. Clamp each track's
        # per-frame displacement to a plausible step (walk/stand) from its previous refined position,
        # so any residual messy-cloud / occlusion mis-pick becomes a SMOOTH move instead of a jump.
        # Uses only past refined frames (no future, no submission temporal) → stays online.
        if _cfg.get("multi_candidate"):
            for b in persons:
                if _cfg.get("sub_anchor") and int(b["class_id"]) in (0, 4, 5):   # scene-25 HUMAN-shaped
                    # FOURIER(4)+Person(0) hard deadline: a box that stays > jump_reset_dist (0.2 m) from its
                    # submission for > jump_reset_frames (10) CONSECUTIVE frames is SNAPPED straight back to the
                    # submission in ONE frame (the safety net for a frozen Fourier re-localize; Person never
                    # triggers it, drift ≈ 0). AGILITY(5) is EXCLUDED — the hold-coast-return-jump owner above
                    # already OWNS its position (its own return-jump / gradual / timeout / reloc-snap resume).
                    # All three humans still skip the clamp/low-pass below (`continue`).
                    if int(b["class_id"]) != 5:
                        tk = _track_key(b)
                        sxg, syg = float(b["orig_xyz"][0]), float(b["orig_xyz"][1])
                        if math.hypot(b["world_xyz"][0] - sxg, b["world_xyz"][1] - syg) > jump_reset_dist:
                            jcnt[tk] = jcnt.get(tk, 0) + 1
                            if jcnt[tk] > jump_reset_frames:    # jump lasted > N frames → SNAP back to submission NOW
                                b["world_xyz"] = [sxg, syg, b["world_xyz"][2]]
                                jcnt[tk] = 0
                        else:
                            jcnt[tk] = 0
                    continue
                tk = _track_key(b)
                pv = prev_refined.get(tk)
                if pv is None:
                    continue
                # LOST-COUNTER (continuity): boxrefine returns the cluster CLOSEST to prev, else the
                # submission when no cluster sits near prev. A big jump from prev ⟹ the track lost its
                # own cluster this frame (occlusion / crossing). HOLD at prev through a SHORT loss; only
                # re-anchor to the submission after a LONG loss (real reacquisition) → no jerk toward sub.
                rx, ry = b["world_xyz"][0], b["world_xyz"][1]
                if math.hypot(rx - pv[0], ry - pv[1]) <= lost_near:
                    lost_count[tk] = 0                     # own cluster near prev → continuity
                else:
                    lost_count[tk] = lost_count.get(tk, 0) + 1
                    if lost_count[tk] < lost_reanchor:     # short loss → HOLD prev (don't jerk)
                        b["world_xyz"] = [pv[0], pv[1], b["world_xyz"][2]]
                    # else: long loss → keep submission position (re-anchor)
                p2 = prev2_refined.get(tk)
                vmag = math.hypot(pv[0] - p2[0], pv[1] - p2[1]) if p2 is not None else 0.0
                g = min(clamp_vmax, max(0.15, 1.3 * vmag + 0.08))   # cap → no cascade (walk/stand)
                x, y = b["world_xyz"][0], b["world_xyz"][1]
                dx, dy = x - pv[0], y - pv[1]
                step = math.hypot(dx, dy)
                if step > g:                               # 1) HARD CLAMP: implausible jump → cap step
                    s = g / step
                    x, y = pv[0] + dx * s, pv[1] + dy * s
                # 2) ADAPTIVE LOW-PASS: kill residual jitter when ~stationary (|velocity vector| ≈ 0,
                # e.g. a person standing in a cluttered corner whose cluster bounces frame-to-frame),
                # but follow closely when genuinely walking. Online (uses only past refined frames).
                vv = vel_vec.get(tk, (0.0, 0.0))
                a = max(lowpass_floor, min(1.0, math.hypot(vv[0], vv[1]) / 0.05))
                x, y = pv[0] + a * (x - pv[0]), pv[1] + a * (y - pv[1])
                vel_vec[tk] = (0.4 * (x - pv[0]) + 0.6 * vv[0], 0.4 * (y - pv[1]) + 0.6 * vv[1])
                b["world_xyz"] = [x, y, b["world_xyz"][2]]
        # YAW stabilisation (anti-spin). The submission yaw is very noisy — 78–167 frames/track rotate
        # >0.5 rad/frame (>860°/s, impossible for a person). Track it with a RATE-LIMITED circular EWMA
        # in mod-π space (the box is symmetric under +π), so the box rotates at most ~human turn speed
        # (yaw_rate rad/frame). Online (only past). Stationary garbage averages out to a steady heading.
        if _cfg.get("multi_candidate") and use_yaw_smooth:
            for b in persons:
                if not _algo.get("yaw_smooth_all") and int(b["class_id"]) != 0:
                    continue            # scene 27 = person only; scene 25 (yaw_smooth_all) = every class
                tk = _track_key(b)
                tgt = float(b.get("yaw", 0.0))
                # HUMAN-shaped while MOVING (NOT stationary) → yaw FOLLOWS the submission RAW (no rate-limit):
                # the submission yaw is good while walking. STILL humans + scene-27 → rate-limited base.
                cls_b = int(b["class_id"])
                if _cfg.get("sub_anchor") and cls_b in (2, 3):       # VEHICLES → NO yaw gate (30° block removed
                    # per user request). Use the fit yaw directly: submission-locked while going STRAIGHT,
                    # cloud-PCA while turning/stopped (the straight-detector + fit_vehicle decide). No per-frame
                    # turn cap, no hold-previous — orientation follows the fit verbatim.
                    yaw_smooth[tk] = tgt
                    yaw_tf[(tk, f)] = tgt
                    continue
                elif (_cfg.get("sub_anchor") and cls_b in (0, 4, 5)  # WALKING humans (submission moving, the
                      and not stationary.get(tk, True)):             # responsive 8-frame flag) → yaw FOLLOWS
                    continue                                         # the submission RAW (fit_one set sub_yaw−90,
                    # no rate-limit): everything tracks the submission while walking; jump-handler covers a jump.
                elif _cfg.get("sub_anchor") and cls_b in (0, 4, 5):  # scene-25 STANDING human → adaptive smoother
                    # The raw shoulder-PCA yaw (tgt) is noisy (offline circ-std ~15-30°), so: (1) CONFIDENCE
                    # GATE — only feed measurements whose PCA anisotropy (yaw_conf) is high enough; (2) target =
                    # circular MEAN of the recent confident yaws (robust to a single 90° flip); (3) CONSISTENCY-
                    # ADAPTIVE rate-limit — slew slowly when the window oscillates (jitter) but FAST when it shows
                    # a consistent net rotation (a real turn; a person turns faster than a vehicle). All mod-π.
                    sub_y = float(b.get("sub_yaw", tgt))             # submission orientation = the yaw anchor
                    ys = yaw_smooth.get(tk)
                    if ys is None:
                        ys = sub_y                                   # start from the submission (safe), correct toward torso
                    hh = yaw_hist.setdefault(tk, [])
                    if float(b.get("yaw_conf", 0.0)) >= sy_conf_min:
                        hh.append(tgt)
                        if len(hh) > sy_win:
                            hh.pop(0)
                    if len(hh) >= 2:
                        a2 = np.unwrap(np.asarray(hh) * 2.0)              # 2θ space → handle the π wrap
                        net = abs(a2[-1] - a2[0]) / 2.0                   # net rotation across the window
                        var = float(np.abs(np.diff(a2)).sum()) / 2.0     # total path (net + jitter)
                        trend = net / (var + 1e-6)                       # ≈1 consistent turn, ≈0 oscillating jitter
                        rate = math.radians(sy_rate_lo + (sy_rate_hi - sy_rate_lo) * min(1.0, trend))
                        target = 0.5 * math.atan2(float(np.sin(a2).sum()), float(np.cos(a2).sum()))  # circular mean
                        db = (sub_y - target) % math.pi                  # SOFT bias: pull the target toward submission
                        if db > math.pi / 2.0:
                            db -= math.pi
                        target += sy_anchor_w * db
                        d = (target - ys) % math.pi
                        if d > math.pi / 2.0:
                            d -= math.pi
                        ys = ys + max(-rate, min(rate, d))
                    dd = (ys - sub_y) % math.pi                          # HARD cap: never drift far from submission
                    if dd > math.pi / 2.0:
                        dd -= math.pi
                    if abs(dd) > sy_max_dev:
                        ys = sub_y + math.copysign(sy_max_dev, dd)
                    yaw_smooth[tk] = ys
                    yaw_tf[(tk, f)] = ys
                    continue
                else:                                                # scene-27 → base rate (smooth)
                    ya, yr = yaw_alpha, yaw_rate
                ys = yaw_smooth.get(tk)
                if ys is None:
                    ys = tgt                                   # init from submission yaw
                else:
                    d = (tgt - ys) % math.pi                   # nearest mod-π difference
                    if d > math.pi / 2.0:
                        d -= math.pi
                    ys = ys + max(-yr, min(yr, ya * d))
                yaw_smooth[tk] = ys
                yaw_tf[(tk, f)] = ys
        if _cfg.get("sub_anchor"):     # scene-25 hard guarantee: no two track_ids overlap > iou_thresh
            synthetic._decollide(persons, float(_cfg.get("s25_params", {}).get("iou_thresh", 0.55)))
        frame_persons[f] = persons
        fixed_size_classes = set(_cfg["person_class_sizes"].keys())
        for b in persons:
            tid = _track_key(b)
            if b.get("class_id") not in fixed_size_classes:
                obs_by_track.setdefault(tid, []).append(
                    {"frame_id": f, "z_top": b.get("z_top"), "foot": b.get("foot"),
                     "npts": b.get("n_points")})
            pos_by_track.setdefault(tid, []).append(
                {"frame_id": f, "x": b["world_xyz"][0], "y": b["world_xyz"][1],
                 "w": b["size"][0], "l": b["size"][1], "yaw": b.get("yaw", 0.0)})
        if use_prev_reseed or _cfg.get("multi_candidate"):   # remember last refined pose per track
            for b in persons:
                tk = _track_key(b)
                if tk in prev_refined:                       # shift prev → prev2 (for velocity)
                    prev2_refined[tk] = prev_refined[tk]
                prev_refined[tk] = [b["world_xyz"][0], b["world_xyz"][1],
                                    b["size"][0], b["size"][1], b.get("yaw", 0.0)]
                if not b.get("jumped", False):           # HEADING trail = SUBMISSION xy on NON-JUMP frames only
                    rh = ref_hist.setdefault(tk, [])      # (jump frames excluded so a glitch/excursion never
                    rh.append((b["orig_xyz"][0], b["orig_xyz"][1]))   # corrupts the direction — heading FREEZES
                    # Cap 32 frames of clean trail; heading (hd_win 10) / stationary (still_win 8) take
                    # min(win, len) so the larger cap doesn't change them.
                    if len(rh) > 32:                      # (at its last clean value through the whole jump)
                        rh.pop(0)
                last_seen[tk] = f                        # for the swap detector's "adjacent frame" gate
                sh = sub_hist.setdefault(tk, [])         # SUBMISSION track → drift-free vehicle heading
                sh.append((b["orig_xyz"][0], b["orig_xyz"][1]))
                if len(sh) > 60:
                    sh.pop(0)
        frame_statics[f] = _static_boxes(sub_path, f, static_sizes, static_poses, floor_z, _cfg["manual"])

        # PER-FRAME ARTIFACT CLEANUP — build the cloud, use it, drop it.
        # Runs only AFTER this frame is fully refined, and only on paths under `cleanup_root`, so a run
        # aimed at the shared cache can never delete it. Off by default (cleanup_root="").
        if cleanup_root:
            frame_dir = dataset.frame_run_dir(output_root, split, scene, f)
            if delete_npz:
                n_npz_deleted += _rm(npz, cleanup_root)
                # With a zone mask, DA3 writes the UNMASKED npz to the plain export dir and zonemask
                # then writes the masked one next to it. `npz` above is only the masked copy, so the
                # 67 MB original would survive (249 MB per 5 frames, measured on scene 27).
                if apply_zone:
                    plain = _frame_export(output_root, split, scene, f, model_san, process_res, False)
                    n_npz_deleted += _rm(os.path.join(plain, "exports", "npz", "results.npz"),
                                         cleanup_root)
                    _rmtree(os.path.join(plain, "metadata"), cleanup_root)
                # The per-camera JPEGs decoded from the videos (~4 MB/frame) and the run metadata:
                # both regenerate in milliseconds. Left behind they reach ~38 GB over a 9000-frame scene.
                _rmtree(os.path.join(frame_dir, "frames"), cleanup_root)
                _rmtree(os.path.join(export, "metadata"), cleanup_root)
            if keep_ply:
                _shrink_ply(cloud_path, ply_voxel)   # keep the frame's cloud, just lighter on disk
                if ply_dir and os.path.isfile(cloud_path):
                    # Lift it out of the deep DA3 export tree into one flat directory, named by
                    # frame -- otherwise the kept clouds are buried six levels down under a
                    # model-and-resolution directory nobody wants to navigate.
                    os.makedirs(ply_dir, exist_ok=True)
                    shutil.move(cloud_path, os.path.join(ply_dir, f"frame_{f:06d}.ply"))
            else:
                n_dyn_ply_deleted += _rm(cloud_path, cleanup_root)
                _rm(ply, cleanup_root)
            _prune(frame_dir, cleanup_root)          # drop whatever directories just went empty

        if stream_fp is not None and f >= frame_start:   # this frame is final -> put it on disk now
            fl, nh = _frame_lines(f, frame_persons, frame_statics, scene_id, floor_z, pos_tf, h_tf,
                                  yaw_tf, fix_person_height, person_height, person_w, person_l)
            stream_fp.write("".join(ln + "\n" for ln in fl))
            stream_fp.flush()
            lines.extend(fl)
            n_h_fixed += nh
            # Hand this frame's rows to whoever is driving the loop. run_final() below just drains
            # the generator; BoxRefiner steps it one frame at a time so a pipeline can own the loop
            # without owning the 21 state dicts that live across frames.
            yield f, fl

    # OC-SORT-inspired per-track IoU stabilisation of x,y (optional, keeps object_id)
    pos_tf: dict = {}
    if use_tracking:
        progress("tracking (OC-SORT)...", 0.88)
        tk = track_params or {}
        for tid, ob in pos_by_track.items():
            for fr, (sx, sy) in tracking.track_sequence(
                    ob, iou_thresh=float(tk.get("iou_thresh", 0.2)),
                    smooth=float(tk.get("smooth", 0.5)),
                    max_gap=int(tk.get("max_gap", 30))).items():
                pos_tf[(tid, fr)] = (sx, sy)

    # stabilise each person track's height across frames (optional)
    h_tf: dict = {}
    if use_height_track:
        progress("height-track...", 0.90)
        htp = heighttrack.make_params(ht_params)
        for tid, obs in obs_by_track.items():
            for r in heighttrack.track_sequence(obs, params=htp, forward_backward=True):
                h_tf[(tid, r["frame_id"])] = r["height"]
    # else: h_tf stays empty → each box keeps its raw per-frame refine height (b["size"][2])

    # write submission .txt
    progress("ghi submission...", 0.96)
    if stream_fp is not None:
        stream_fp.close()
    else:
        lines, n_h_fixed = [], 0
        for f in frames:
            if f < frame_start:                      # WARMUP frame — seeds state only, NOT emitted
                continue
            fl, nh = _frame_lines(f, frame_persons, frame_statics, scene_id, floor_z, pos_tf, h_tf,
                                  yaw_tf, fix_person_height, person_height, person_w, person_l)
            lines.extend(fl)
            n_h_fixed += nh
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines) + ("\n" if lines else ""))

    elapsed = time.perf_counter() - t_total
    ran = len(frames) - reused
    return {"submission": out_path, "submission_name": name,
            "n_frames": len(frames), "reused": reused, "ran": ran,
            "n_tracks": len(pos_by_track), "n_lines": len(lines), "n_height_fixed": n_h_fixed,
            "n_refine_failed": refine_failed, "n_full_ply_deleted": n_full_ply_deleted,
            "n_npz_deleted": n_npz_deleted, "n_dyn_ply_deleted": n_dyn_ply_deleted,
            "frame_start": frame_start, "frame_end": frames[-1],
            "seconds_total": round(elapsed, 1), "seconds_per_frame": round(elapsed / len(frames), 2),
            "da3_seconds": round(t_da3, 1), "da3_per_frame": round(t_da3 / ran, 2) if ran else 0.0,
            "refine_seconds": round(t_refine, 1), "refine_per_frame": round(t_refine / len(frames), 2)}
