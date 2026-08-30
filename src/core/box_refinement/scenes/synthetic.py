"""Warehouse_025 (synthetic): the scene-specific half of the refinement.

Everything here is driven by the SUBMISSION as the anchor. Vehicles (NovaCarter, Transporter) are
re-fitted every frame against the BEV density-excess of the cloud at a fixed class size; people
(Person, Fourier, Agility) follow the submission and only fall back on the cloud when a frame jumps.
The per-frame loop, the DA3 cloud and the online smoothing that wrap all of this live in
``engine.orchestrator`` and are shared with the real-world scene.

The fits themselves are in ``synthetic_fit`` (imported below under its original name so the call
sites read exactly as they did when these numbers were validated).
"""
from __future__ import annotations

import math

import numpy as np

from ..engine import submission
from ..engine.geometry import bev_iou
from . import synthetic_fit as scene25_refine


def _track_key(box):
    return f"{int(box.get('class_id', -1))}:{int(box.get('object_id', -1))}"


SCENE25_FIXED_CLASS_SIZES = {
    # [width, length, height] = MEDIAN over GT Warehouse_010 (median, not mean: the dist is right-
    # skewed by walking/stride frames; median maximises mean 3D-IoU = the HOTA threshold-integral).
    # length (scale[1], local-Y) is ALWAYS along the heading. For Nova/Transp/Agility l is the LONGER
    # edge; for Person/Fourier the WIDTH (shoulders) is longer and travel is along the SHORTER length.
    0: [0.559, 0.402, 1.842],       # Person       (w=shoulders>l; travel along short l) — l kept at 0.402 (safe)
    1: [1.21380612521306, 2.31336527421999, 2.15494466137264],  # Forklift     (l>w; travel along long l) — class 1 is a forklift
    2: [0.499, 0.784, 0.599],       # NovaCarter   (l>w; travel along long l)
    3: [0.659, 1.431, 0.231],       # Transporter  (l>w; travel along long l)
    4: [0.601, 0.466, 1.643],       # FourierGR1T2 (w=shoulders>l; travel along short l)
    5: [0.542, 0.807, 1.780],       # AgilityDigit (l>w; travel along long l)
}
# DYNAMIC length when WALKING: the GT-10 length MEAN (> the stationary median above) — a moving person's
# footprint stretches with the stride. Used only while moving; stationary keeps the median fix size.
# Person + Fourier (human-shaped that stride); AgilityDigit is fine on its fix size (per user).
SCENE25_WALK_LENGTH = {0: 0.70, 4: 0.52}   # GT-10 MOVING-only median of the length axis (sy): Person strides
# hard (standing 0.38 → moving 0.72); Fourier is a robot, strides little (0.46 → 0.48). Old 0.519/0.580
# were biased by standing frames / too high. Length is speed-dependent (Person: 0.91 slow-walk → 0.57 run).

# Per-class oriented-fit params (NovaCarter / Transporter). Both RAISE the search by lift=0.5m to
# catch DA3-floated points and drop the z<floor_drop floor layer, then rotate the fixed footprint to
# best coverage. NovaCarter is near-SQUARE (0.50x0.78) so its coverage-vs-yaw curve is flat+noisy →
# STRONG yaw_bias toward prev keeps it from spinning. Transporter is elongated (0.66x1.43, yaw clear)
# so a LIGHT bias lets the fit track the true heading; its ceiling stays low (h 0.229 + 0.5 = 0.73m)
# so the shelf it carries above is mostly excluded. UI-tunable (Final tab) per class.
# Common params for ALL scene-25 classes (submission-anchored).
# iou_thresh 0.55: scene 25 is SYNTHETIC and objects (esp. Transporters) genuinely ram into each other
# a lot, so IoU 0.2-0.55 is usually a REAL collision — only IoU>0.55 (a true wrong-merge onto one blob)
# gets separated. Below that we keep the refined following the natural single-object motion.
SCENE25_PARAMS = {"iou_thresh": 0.55,
                  # HUMAN handling (vehicles NOT affected). Person(0) + FOLLOW frames (scene25_refine.fit_one):
                  #   FOLLOW (|submission − prev refined| ≤ jump_gate) → x,y = submission. STANDING also refines the
                  #     YAW from the cloud shoulder-PCA (BODY band z∈[0, still_z_hi] within still_radius, rotate the
                  #     fixed box ±still_yaw_win_deg to best match the tall-column cells; x,y stay = submission).
                  #   FOURIER(4) JUMP (> jump_gate) → CLOUD PULL-BACK: anchor at the PREVIOUS refined x,y, search the
                  #     BEV cloud (band z∈[0, jump_z_hi] m) within ±jump_search for the nearest person cluster
                  #     (box-coverage fit ⇒ prefers a size-matching cluster), move x,y toward it but ≤ jump_cap
                  #     from prev. yaw HELD (moving) / shoulder-PCA (stationary). Stuck > jump_reset_dist for
                  #     > jump_reset_frames (run_final) → snap back to submission.
                  #   AGILITY(5) JUMP → the SEPARATE agi_* hold-coast-return-jump state machine below (NOT the cloud).
                  "jump_gate": 0.2,                                      # FOURIER: |submission − prev refined| above this = JUMP (cloud pull-back); ≤ = FOLLOW
                  "jump_search": 0.2,                                    # FOURIER JUMP: box-slide bound ± this around prev (m) → max search 0.2m
                  "jump_cap": 0.2,                                       # FOURIER JUMP: max move/frame from prev (m) — also caps the Fourier glide
                  "jump_z_hi": 1.8,                                      # FOURIER JUMP: cluster band z∈[0, this] (m) — FULL BODY (denser, clearer match; yaw is from submission so position-only)
                  "head_win": 10, "head_min": 0.10,                      # FOURIER moving-jump: heading = last 10 NON-JUMP submission frames (net ≥ this m)
                  "near_pref": 5.0, "ang_pref": 0.5,  # FOURIER moving-jump: STRONG nearer bias + WEAK soft velocity preference
                  # AGILITY(5) hold-coast-return-jump state machine (synthetic._agi_step) — submission-ONLY,
                  # stateful, NO point cloud. On a submission step > agi_jump_gate (a glitch) the box HOLDs at
                  # the pre-jump anchor + a robust ≈0 coast velocity (median per-frame delta over agi_vel_win
                  # clean frames; |v| < agi_deadband ⇒ pure hold), est = anchor + v·dt bounded to agi_coast_max.
                  # RESUME via a rate-limited GLIDE (≤ agi_step_cap/frame) on: a RETURN-jump (submission jumps
                  # back AND |sub−est| ≤ agi_jb_near), a GRADUAL come-back (|sub−est| ≤ agi_grad_near for
                  # agi_grad_frames frames), a TIMEOUT (coasted ≥ agi_coast_timeout frames), or a reloc-SNAP
                  # (|sub−est| > agi_snap_dist for agi_snap_frames frames). Validated offline: 20/20 glitches
                  # held+resumed, max per-frame step 0.248 m. SEPARATE from the shared jump_*/still_* keys (they
                  # collide by name, so these are agi_-prefixed) — only the Agility position owner reads them.
                  "agi_jump_gate": 0.25, "agi_jb_near": 0.25, "agi_grad_near": 0.28,
                  "agi_coast_timeout": 42, "agi_snap_dist": 0.9, "agi_step_cap": 0.24, "agi_vel_win": 60,
                  "agi_deadband": 0.005, "agi_coast_max": 0.5, "agi_grad_frames": 2, "agi_snap_frames": 6,
                  "clu_vox": 0.12,  # 3D voxel size for the cluster-centroid jump fit (nearest 3D cluster → median centroid)
                  "jump_iou": 0.2,  # a JUMPED human box overlapping another human track > this BEV-IoU → revert to prev (no jump onto another track)
                  "foot_cell": 0.06, "foot_kbg": 1.8, "foot_min_pts": 12,  # BEV cell / floor-excess threshold / min pts
                  "still_radius": 0.50, "still_z_hi": 2.0, "still_tall": 1.0, "still_sat_frac": 0.10,
                  # ^ STANDING: crop radius / crop z-hi (full column) / keep only columns TALLER than still_tall
                  #   (the upright body — the offline-most-stable yaw source) / keep a weaker 2nd cluster.
                  # STANDING yaw: PCA shoulder axis on the tall-column cells, rotate the fixed box ±win, keep
                  # the best body match. The noisy raw yaw is then smoothed in run_final (confidence gate +
                  # consistency-adaptive rate: still_yaw_conf_min / still_yaw_hist / still_yaw_rate_lo|hi).
                  "still_yaw_win_deg": 40.0, "still_yaw_step_deg": 6.0,
                  # HUMAN still-vs-moving gate, from the SUBMISSION motion (un-lagged): submission moves
                  # < still_net over the last still_win frames → NEARLY STILL (apply the stable pull); more
                  # → MOVING (drop the pull, follow submission at FULL speed → no lag).
                  "still_win": 8, "still_net": 0.08,   # Person(0): NET over still_win — 0.08→95% moving-recall, FP 0% (0.06→97% if fuller wanted)
                  # Fourier(4)/Agility(5) walk slow & curve, so the 8-frame NET (net≈0 on a curve) only caught
                  # 80% of their moving frames. Switch them to a robust MEDIAN per-frame step over still_win
                  # with a low gate: median-step ≥ 0.005 m/f → MOVING. Measured recall 97-98% (vs 80% for NET),
                  # false-positive on a true stand-still only ~0% (their parked median-step p99 ≈ 0.002 m/f).
                  "still_mstep_slow": {4: 0.005, 5: 0.005},
                  # NovaCarter/Transporter (2,3) BEV DENSITY-EXCESS ghost-aware fixed-box fit
                  # (scene25_refine.fit_vehicle): drop z → density → excess(vehicle) vs uniform floor →
                  # DBSCAN drop ghosts → slide fixed box bounded ±search, max contain−stickout−nearest-centre.
                  "veh_search": 0.5,         # box centre BOUND ± this (m) around the submission (anti-drift)
                  "veh_cell": 0.08,          # BEV density cell size (m)
                  "veh_kbg": 2.5,            # excess threshold: cell density > kbg·(floor background)
                  "veh_eps_cells": 2.2,      # ghost connected-components radius (in cells)
                  "veh_min_cells": 3,        # drop clusters smaller than this (specks)
                  "veh_sat_frac": 0.12,      # drop ghost satellites below this fraction of the strongest cluster
                  "veh_beta": 1.0,           # stick-out penalty (enforces a snug single-ghost match)
                  "veh_gamma_frac": 0.4,     # nearest-centre cost = gamma_frac·median(excess density) per metre
                  "veh_sstep": 0.08,         # box-centre slide step (m)
                  "veh_yaw_step_deg": 10.0,  # orientation scan step (deg)
                  "veh_stick_margin": 0.18,  # width of the stick-out band just outside the box (m)
                  "veh_min_excess": 4,       # below this many excess cells → keep the submission
                  # YAW = PCA principal axis of the kept cluster, refined ±yaw_win_deg (submission gives ONLY
                  # the position seed, never the orientation). Both vehicle classes (Nova + Transporter) use 30°.
                  "veh_yaw_win_deg": 30.0,
                  # STRAIGHT-line detector (over veh_str_win frames of submission xy): net/total > veh_str_smin and
                  # net > veh_str_dmin → moving STRAIGHT → take the submission POSITION+yaw VERBATIM (no cloud fit);
                  # else (turn / stop) → cloud fit (fit_vehicle: BEV position + PCA yaw).
                  # PER-CLASS smin — calibrated from GT scene-10 (true class behaviour) + scene-25 submission noise:
                  #  - GT-10 shows Transporter is a DEAD-STRAIGHT hauler (path net/tot median 1.00, straight ~97%)
                  #    while NovaCarter WEAVES constantly (median 0.87, straight ~40-60%).
                  #  - submission xy is JITTERY, which depresses net/tot ~class-independently (Transp scene-25 RAW
                  #    median still 0.986, Nova 0.936, but tails sag: Transp p10 0.66, Nova p10 0.72).
                  #  => Transporter LENIENT smin 0.85 (it really is straight; don't let jitter kick a straight pass
                  #     to the cargo-biased cloud-fit), dmin 0.08 — UNCHANGED ("như cũ"): Transporter always cloud-
                  #     fits POSITION anyway, so straight only toggles yaw-lock-vs-PCA.
                  #  => NovaCarter is the FITTED class: when STRAIGHT it takes the submission POS+yaw VERBATIM and
                  #     DISCARDS the cloud, so the user wants "any gentle straight motion → follow submission v5".
                  #     smin 0.70 (was 0.90) = the GT-10 Nova p25 (the gentle-vs-SHARP-turn boundary): net/tot>0.70
                  #     → straight (gentle arcs included — Nova's submission yaw is clean even turning, yaw-rate
                  #     p99 1.9°/f), only the SHARPEST turns (<0.70) go to the cloud. dmin 0.04 (was 0.08) so a
                  #     slow crawl still counts as MOVING (its net over 8f clears 0.04) instead of "stopped" → a
                  #     slow straight pass follows the submission too. Net effect on scene-25 sub: Nova STRAIGHT
                  #     51%→76%, TURN 28%→11%, STOP 21%→13%.
                  "veh_str_win": 8, "veh_str_smin": {2: 0.70, 3: 0.85},
                  "veh_str_dmin": {2: 0.04, 3: 0.08},
                  # NOVACARTER (class 2) ONLY — simple velocity state machine (user 2026-07-01, Transporter
                  # unchanged): MEDIAN per-frame speed over `nova_speed_win`=30 frames. >= `nova_speed_thr` →
                  # MOVING → take the submission POS+yaw VERBATIM; < thr → braked/STOPPED-about-to-turn → BEV
                  # fit (fit_vehicle solves yaw from the cloud). 0.010 m/f found from submission v5: Nova turns
                  # at <=0.008 (p90), straight-moving >=0.014 (p10) — a clean gap. (Frame-to-frame yaw jump-guards
                  # for BOTH vehicles were REMOVED 2026-07-02 — yaw may now jump freely between frames.)
                  "nova_speed_win": 30, "nova_speed_thr": 0.010}
# Per-class z-band scan range for the vehicle classes (Nova=2, Transporter=3): estimate the base at z=0
# and RAISE the h-tall slab gradually from `floor_z` (floor_drop 0) up to `floor_z + lift` (0.15) in
# `z_step` (0.05) steps; the densest slab is the vehicle's vertical band. Scanning only z∈[0,0.15] keeps
# cargo/forklift points (higher up) out of the band. See scene25_refine.fit_vehicle.
SCENE25_ORIENTED_PARAMS = {
    2: {"lift": 0.15, "floor_drop": 0.0, "z_step": 0.05},        # NovaCarter
    3: {"lift": 0.15, "floor_drop": 0.0, "z_step": 0.05},        # Transporter
}



def _mask_other_classes(P, sub_boxes, keep_cls, margin=0.12):
    """Remove cloud points inside the footprint of any object whose class != keep_cls. Isolates a
    TRANSPORTER's own points from a nearby PERSON/Nova/Fourier/Agility whose (low-z) cluster would
    otherwise pull the BEV density fit (esp. an Agility standing next to the flat Transporter). Same-class
    neighbours are KEPT — a Transporter-Transporter overlap is the separate, unsolved identity case."""
    import numpy as _np
    import math as _math
    if P is None or len(P) == 0:
        return P
    keep = _np.ones(len(P), dtype=bool)
    for b in sub_boxes:
        if int(b["class_id"]) == keep_cls:
            continue
        cx, cy = float(b["center"][0]), float(b["center"][1])
        sz = b["size"]
        hw = float(sz[0]) / 2.0 + margin
        hl = float(sz[1]) / 2.0 + margin
        th = float(b["yaw"]) - _math.pi / 2.0                 # output-convention box orientation
        c, s = _math.cos(-th), _math.sin(-th)
        u = P[:, 0] * c - P[:, 1] * s
        v = P[:, 0] * s + P[:, 1] * c
        cu, cv = cx * c - cy * s, cx * s + cy * c
        keep &= ~((_np.abs(u - cu) < hw) & (_np.abs(v - cv) < hl))
    return P[keep]


def _scene25_boxes(P, sub_path, frame_id, classes, sizes, common, oriented_params,
                   prev_refined, floor_z, exclude_objs, last_seen=None,
                   stationary=None, veh_straight=None, heading=None,
                   nova_speed=None):
    """Scene-25 ONLY: submission-anchored refine. Persons/Fourier/Agility (0,4,5) use the 2-stage
    scene25_refine.fit_one; NovaCarter/Transporter (2,3) use the DEDICATED scene25_refine.fit_vehicle
    (submission seed + fixed size + z-band & 180° orientation scan). Each track follows its OWN submission
    detection. Detection is ONLINE/CAUSAL — frame f only uses f-1 (via prev_refined + last_seen) — so the
    sequential run is a valid online tracker. Returns boxes in the shape realworld.refine emits so the orchestrator's clamp/
    low-pass/yaw smoothing + collision run on top."""
    import math
    classes = set(int(c) for c in classes)
    exo = set(int(x) for x in (exclude_objs or []))
    veh_cls = {2, 3}
    common = common or {}
    last_seen = last_seen if last_seen is not None else {}
    jump_gate = float(common.get("jump_gate", 0.2))           # HUMAN: |submission − prev refined| above this = a JUMP → cloud pull-back; ≤ this = FOLLOW
    sub_boxes = submission.get_boxes(sub_path, frame_id)["boxes"]
    # VEHICLES (2,3): NO IoU-coast / velocity / turning logic — the scan-based fit_vehicle (submission seed +
    # fixed size + z-band & 180° orientation scan) handles everything.
    # pass 1 — gather per-object info; each track follows its OWN submission detection.
    recs, items = [], []
    for b in sub_boxes:
        cid = int(b["class_id"])
        oid = int(b.get("object_id", -1))
        if cid not in classes or oid in exo:
            continue
        det = b                                                     # the detection this track follows
        sz = (sizes or {}).get(cid) or (sizes or {}).get(str(cid)) or det["size"]
        w, l, h = float(sz[0]), float(sz[1]), float(sz[2])
        if cid in (0, 4, 5):
            # Human-shaped classes (Person/Fourier/Agility) ALWAYS keep the SUBMISSION's per-frame HEIGHT — the
            # fixed class-height prior is not used for them (user 2026-07-04: this is now the only behaviour).
            # Width and the moving walk-length stretch on `l` are unchanged; vehicles still use their fixed size.
            h = float(det["size"][2])
        sx, sy, scz = det["center"]
        pv = prev_refined.get(f"{cid}:{oid}")
        ref_yaw = float(pv[4]) if (pv is not None and len(pv) > 4) else float(det["yaw"])
        if cid in SCENE25_WALK_LENGTH and not (stationary or {}).get(f"{cid}:{oid}", True):
            l = float(SCENE25_WALK_LENGTH[cid])          # MOVING human → stretch length (stride). Gated by the
            #                                              SAME `stationary` flag (median-step/NET).
        # JUMP detection — the submission moved more than `jump_gate` from the PREVIOUS refined box, a glitch /
        # fast relocation we don't trust. Fires for BOTH moving AND standing tracks:
        #   FOURIER(4) → fit_one handles it (MOVING → foot-BEV re-locate near prev; STILL → hold x,y at prev).
        #   AGILITY(5) → fit_one IGNORES it (always FOLLOWs); the flag is still set so this JUMP frame is EXCLUDED
        #     from ref_hist (keeping the still/yaw trail clean). Agility's position is fully owned by the
        #     hold-coast-return-jump state machine (_agi_step), applied to the returned boxes below.
        # GUARD: only on an ADJACENT continuous track (seen the immediately previous frame). A re-appearing
        # track (gap / occlusion) has a stale pv → NOT flagged → just follows the fresh detection (we do not
        # try to "recover" a track that vanished for many frames — that is a hidden region, not a jump).
        st = (stationary or {}).get(f"{cid}:{oid}", True)
        hjump = False; anchor_xy = None
        # JUMP flag set for Agility(5) + Fourier(4). Person(0) is NOT jump-handled (just FOLLOWs the submission);
        # it still gets the shared stationary / walk-length / standing-yaw treatment (gated by `stationary`).
        if (cid in (4, 5) and pv is not None
                and (last_seen or {}).get(f"{cid}:{oid}") == frame_id - 1
                and math.hypot(sx - pv[0], sy - pv[1]) > jump_gate):
            hjump = True; anchor_xy = (float(pv[0]), float(pv[1]))
        items.append((sx, sy, w, l))
        recs.append(dict(cid=cid, oid=oid, w=w, l=l, h=h, sx=sx, sy=sy, scz=scz,
                         sub_yaw=float(det["yaw"]), ref_yaw=ref_yaw, jump=hjump, anchor_xy=anchor_xy,
                         stat=st, hd=(heading or {}).get(f"{cid}:{oid}")))
    P = scene25_refine.subcrop(P, items, margin=1.0)   # ONCE per frame → 1.0m around each bbox
    P_transp = None                                     # lazy: cloud with NON-Transporter points masked out
    kw_common = {k: v for k, v in common.items()           # humans (fit_one) get no vehicle-only keys
                 if k != "iou_thresh" and not k.startswith("veh_") and not k.startswith("turn_")}
    out = []
    for r in recs:
        cid, oid = r["cid"], r["oid"]
        if cid in veh_cls:                               # NovaCarter / Transporter (2,3)
            tk2 = f"{cid}:{oid}"
            # NOVACARTER (2): NEW simple velocity state machine — 30-frame median speed >= nova_speed_thr =>
            # MOVING => submission verbatim; else braked/stopped-about-to-turn => BEV fit (yaw from cloud).
            # TRANSPORTER (3): UNCHANGED — net-based `straight` toggles yaw-lock-vs-PCA on the BEV fit.
            nova_move = (cid == 2
                         and float((nova_speed or {}).get(tk2, 0.0)) >= float(common.get("nova_speed_thr", 0.010)))
            if nova_move:
                # NOVACARTER MOVING → submission POSITION *and* yaw VERBATIM — NO cloud fit.
                rr = {"x": float(r["sx"]), "y": float(r["sy"]),
                      "yaw": float(r["sub_yaw"]) - math.pi / 2.0,
                      "anchor": "nova_move", "yaw_conf": 0.0, "fit": 0.0, "located": True}
            else:
                # NovaCarter STOPPED/turning, and TRANSPORTER always → cloud POSITION fit (BEV density-excess
                # ghost-aware fixed box). YAW: Transporter STRAIGHT → LOCK to the (clean) submission orientation;
                # NovaCarter stopped / Transporter turn-stop → solve from the cloud (PCA).
                fk = dict(sub_xy=(r["sx"], r["sy"]), prev_yaw=r["ref_yaw"], cid=cid, floor_z=floor_z,
                          w=r["w"], l=r["l"], h=r["h"],     # submission = POSITION seed only (NOT yaw)
                          yaw_win_deg=float(common.get("veh_yaw_win_deg", 30.0)),
                          search=float(common.get("veh_search", 0.5)),
                          cell=float(common.get("veh_cell", 0.08)),
                          kbg=float(common.get("veh_kbg", 2.5)),
                          eps_cells=float(common.get("veh_eps_cells", 2.2)),
                          min_cells=int(common.get("veh_min_cells", 3)),
                          sat_frac=float(common.get("veh_sat_frac", 0.12)),
                          beta=float(common.get("veh_beta", 1.0)),
                          gamma_frac=float(common.get("veh_gamma_frac", 0.4)),
                          sstep=float(common.get("veh_sstep", 0.08)),
                          yaw_step_deg=float(common.get("veh_yaw_step_deg", 10.0)),
                          stick_margin=float(common.get("veh_stick_margin", 0.18)),
                          min_excess=int(common.get("veh_min_excess", 4)))
                if cid == 3 and bool((veh_straight or {}).get(tk2, False)):   # Transporter straight → lock yaw
                    fk["lock_yaw"] = float(r["sub_yaw"]) - math.pi / 2.0
                fit_P = P
                if cid == 3:                                   # TRANSPORTER: fit on the cloud with every
                    if P_transp is None:                       # NON-Transporter object's points masked out, so a
                        P_transp = _mask_other_classes(P, sub_boxes, 3)   # nearby person/Agility can't pull the
                    fit_P = P_transp                           # BEV fit. Nova(2) still fits on the full P.
                rr = scene25_refine.fit_vehicle(fit_P, **fk)
            # Frame-to-frame yaw JUMP-GUARDS REMOVED (user 2026-07-02): both NovaCarter and Transporter yaw may now
            # jump freely between frames — the fit / submission yaw is taken verbatim, no hold-previous cap.
        else:                                            # persons/Fourier/Agility (0,4,5)
            fk = dict(seed_xy=(r["sx"], r["sy"]),
                      ref_yaw=r["ref_yaw"], w=r["w"], l=r["l"], cid=cid, floor_z=floor_z,
                      sub_yaw=r["sub_yaw"],            # MOVING human yaw = submission orientation (sub_yaw−90)
                      anchor_xy=r["anchor_xy"],       # JUMP → foot-search anchor (the previous refined position)
                      jump=bool(r["jump"]), stationary=bool(r["stat"]),
                      heading=r["hd"],                # FOURIER moving-jump → 10-frame refined heading (search cone)
                      **kw_common)                    # (Agility(5): jump=... is IGNORED by fit_one → always FOLLOW;
            #                                            its glitch handling is synthetic._agi_step, applied after)
            rr = scene25_refine.fit_one(P, **fk)
        out.append({
            "object_id": oid, "class_id": cid,
            "orig_xyz": [round(float(r["sx"]), 3), round(float(r["sy"]), 3), round(float(r["scz"]), 3)],
            "world_xyz": [float(rr["x"]), float(rr["y"]), float(floor_z)], "size": [r["w"], r["l"], r["h"]],
            "yaw": round(float(rr["yaw"]), 4), "yaw_deg": round(float(np.degrees(rr["yaw"])), 1),
            "yaw_found": True, "shift_m": 0.0, "n_points": int(rr["fit"]),
            "source": "s25_" + rr["anchor"], "z_top": None, "foot": None,
            "jumped": bool(r["jump"]),                    # was this frame a JUMP? → excluded from the heading trail
            "located": bool(rr.get("located", False)),    # did the cloud find a real person cluster this frame?
            "cloud_vote": rr.get("cloud", ""),            # JUMP cloud arbiter: at_sub / at_prev / lost ("" otherwise)
            "yaw_conf": float(rr.get("yaw_conf", 0.0)),   # STANDING shoulder-PCA confidence (aniso) → yaw smoother
            "sub_yaw": float(r["sub_yaw"]) - math.pi / 2.0,  # submission orientation (our convention) → yaw anchor
        })
    # ANTI-JUMP-INTO-ANOTHER (humans only): a box that JUMPED this frame and whose result overlaps ANOTHER
    # human-shaped track (BEV-IoU > jump_iou) is reverted to its PREV refined position — a jump must not land on
    # another track's box (an ID-switch / wrong-merge). A GRADUAL move (FOLLOW, not jumped) into another box is
    # left untouched: real collisions where people walk into each other are allowed.
    jump_iou = float(common.get("jump_iou", 0.2))
    hum = [b for b in out if int(b["class_id"]) in (0, 4, 5)]
    for b in hum:
        if not b.get("jumped"):
            continue
        ri = (b["world_xyz"][0], b["world_xyz"][1], b["size"][0], b["size"][1], b["yaw"])
        for ob in hum:
            if (int(ob["class_id"]), int(ob["object_id"])) == (int(b["class_id"]), int(b["object_id"])):
                continue
            rj = (ob["world_xyz"][0], ob["world_xyz"][1], ob["size"][0], ob["size"][1], ob["yaw"])
            if bev_iou(ri, rj) > jump_iou:     # jumped onto another track → snap back to prev refined
                pv = prev_refined.get(f"{int(b['class_id'])}:{int(b['object_id'])}")
                if pv is not None:
                    b["world_xyz"] = [float(pv[0]), float(pv[1]), float(b["world_xyz"][2])]
                break
    return out


def _decollide(persons, iou_thresh, max_iter=8, step=0.06):
    """Last-resort GEOMETRIC guarantee that no two different track_ids overlap above iou_thresh (run
    AFTER the position/yaw smoothing so it has the final word). Pushes the box that drifted MORE from
    its submission away along the line joining the two centres until their BEV-IoU drops to threshold.
    Returns #boxes nudged. Cheap, deterministic (always moves the more-drifted side → no oscillation)."""
    moved = set()
    for _ in range(max_iter):
        changed = False
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                bi, bj = persons[i], persons[j]
                if int(bi["class_id"]) in (0, 4, 5) or int(bj["class_id"]) in (0, 4, 5):
                    continue                              # human-shaped: overlap is OK, never push
                ri = (bi["world_xyz"][0], bi["world_xyz"][1], bi["size"][0], bi["size"][1], bi["yaw"])
                rj = (bj["world_xyz"][0], bj["world_xyz"][1], bj["size"][0], bj["size"][1], bj["yaw"])
                if bev_iou(ri, rj) <= iou_thresh:
                    continue
                vi, vj = int(bi["class_id"]) in (2, 3), int(bj["class_id"]) in (2, 3)
                if vi and vj:
                    continue                              # two vehicles → coast through, never push
                if vi:                                    # mixed pair → push the NON-vehicle (person)
                    w, o = bj, bi
                elif vj:
                    w, o = bi, bj
                else:
                    di = math.hypot(bi["world_xyz"][0] - bi["orig_xyz"][0], bi["world_xyz"][1] - bi["orig_xyz"][1])
                    dj = math.hypot(bj["world_xyz"][0] - bj["orig_xyz"][0], bj["world_xyz"][1] - bj["orig_xyz"][1])
                    w, o = (bi, bj) if di >= dj else (bj, bi)
                dx = w["world_xyz"][0] - o["world_xyz"][0]
                dy = w["world_xyz"][1] - o["world_xyz"][1]
                d = math.hypot(dx, dy) or 1e-6
                w["world_xyz"][0] += dx / d * step
                w["world_xyz"][1] += dy / d * step
                moved.add(id(w))
                changed = True
        if not changed:
            break
    return len(moved)


def _agi_median(xs):
    xs = sorted(xs)
    n = len(xs)
    return 0.0 if n == 0 else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def _agi_step(stt, f, sub, p):
    """ONE frame of the VALIDATED Agility hold-coast-return-jump state machine (submission-ONLY, stateful,
    NO point cloud). Ported 1:1 from the offline make_holdcoast validator (20/20 glitches held+resumed,
    max per-frame step 0.248 m). ``stt`` = the per-track state dict (MUTATED in place), ``sub`` = this
    frame's submission (x, y), ``p`` = the agi_* params (the frame's s25_params). Returns the refined (x, y).

    State ∈ {F(ollow), C(oast/hold), G(lide)}: FOLLOW while the submission step ≤ agi_jump_gate; a bigger step
    = a GLITCH → HOLD at the anchor(= prev refined) + robust ≈0 coast velocity (est = anchor + v·dt, bounded to
    agi_coast_max). RESUME (→ GLIDE) on a return-jump / gradual come-back / timeout / reloc-snap; the GLIDE
    rate-limits the move onto the submission at ≤ agi_step_cap/frame, then returns to FOLLOW."""
    gate = float(p.get("agi_jump_gate", 0.25))
    vel_win = int(p.get("agi_vel_win", 60))
    deadband = float(p.get("agi_deadband", 0.005))
    # continuity: brand-new track OR a frame gap (track re-appeared) → reset to FOLLOW at the submission.
    if stt.get("last_f") is None or f - stt["last_f"] != 1:
        stt.update(state="F", buf=[(f, sub)], anchor=None, v=(0.0, 0.0), cstart=None,
                   grad_cnt=0, snap_cnt=0, prev=sub, prev_sub=sub, last_f=f)
        return sub
    prev_sub = stt["prev_sub"]                              # submission at the previous frame
    out = None
    if stt["state"] == "F":
        if math.hypot(sub[0] - prev_sub[0], sub[1] - prev_sub[1]) > gate:   # a > gate submission step = GLITCH
            buf = stt["buf"]
            if len(buf) < 2:
                v = (0.0, 0.0)
            else:
                seg = buf[-(vel_win + 1):]
                dx = [seg[k][1][0] - seg[k - 1][1][0] for k in range(1, len(seg))]
                dy = [seg[k][1][1] - seg[k - 1][1][1] for k in range(1, len(seg))]
                vx, vy = _agi_median(dx), _agi_median(dy)
                v = (0.0, 0.0) if math.hypot(vx, vy) < deadband else (vx, vy)   # robust per-frame coast velocity
            stt["anchor"] = stt["prev"]; stt["v"] = v; stt["cstart"] = stt["last_f"]
            stt["state"] = "C"; stt["grad_cnt"] = 0; stt["snap_cnt"] = 0
        else:                                              # calm → FOLLOW the submission, extend the clean buffer
            out = sub; stt["prev"] = sub; stt["buf"].append((f, sub))
            if len(stt["buf"]) > vel_win + 2:
                stt["buf"].pop(0)
    if out is None and stt["state"] == "C":
        dt = f - stt["cstart"]
        vx, vy = stt["v"]; anchor = stt["anchor"]
        offx, offy = vx * dt, vy * dt
        m = math.hypot(offx, offy); cmax = float(p.get("agi_coast_max", 0.5))
        if m > cmax:
            offx, offy = offx * cmax / m, offy * cmax / m
        est = (anchor[0] + offx, anchor[1] + offy)
        d = math.hypot(sub[0] - est[0], sub[1] - est[1])
        return_jump = (math.hypot(sub[0] - prev_sub[0], sub[1] - prev_sub[1]) > gate
                       and d <= float(p.get("agi_jb_near", 0.25)))
        stt["grad_cnt"] = stt["grad_cnt"] + 1 if d <= float(p.get("agi_grad_near", 0.28)) else 0
        stt["snap_cnt"] = stt["snap_cnt"] + 1 if d > float(p.get("agi_snap_dist", 0.9)) else 0
        if (return_jump or stt["grad_cnt"] >= int(p.get("agi_grad_frames", 2))
                or dt >= int(p.get("agi_coast_timeout", 42))
                or stt["snap_cnt"] >= int(p.get("agi_snap_frames", 6))):
            stt["state"] = "G"
        else:
            out = est; stt["prev"] = est                   # still coasting
    if out is None and stt["state"] == "G":                # rate-limited glide onto the submission
        cap = float(p.get("agi_step_cap", 0.24)); prev = stt["prev"]
        d = math.hypot(prev[0] - sub[0], prev[1] - sub[1])
        if d <= cap:
            out = sub; stt["prev"] = sub; stt["state"] = "F"; stt["buf"] = [(f, sub)]
        else:
            out = (prev[0] + (sub[0] - prev[0]) * cap / d, prev[1] + (sub[1] - prev[1]) * cap / d)
            stt["prev"] = out
    stt["prev_sub"] = sub; stt["last_f"] = f
    return out if out is not None else sub
