
from __future__ import annotations

import os

import numpy as np
import open3d as o3d
from sklearn.cluster import HDBSCAN
from sklearn.mixture import GaussianMixture

from ..engine import gt, submission
from ..engine.geometry import bev_iou as _bev_iou

METHODS = ("meanshift", "hdbscan", "gmm")


def _load_cloud(export_dir: str, cloud_path: str = "") -> np.ndarray:
    ply = cloud_path if cloud_path else os.path.join(export_dir, "pointcloud", "estimated_fused.ply")
    if not os.path.isfile(ply):
        raise FileNotFoundError(f"cloud not found: {ply} (fuse first)")
    pcd = o3d.io.read_point_cloud(ply)
    return np.asarray(pcd.points, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Methods — operate on a (subsampled) BEV xy array, return a (cx, cy) mode/None.
# Gating + height are done afterwards on the FULL ROI in refine().
# --------------------------------------------------------------------------- #
def _meanshift_bev(xy, seed, bandwidth, iters=40):
    c = np.asarray(seed, dtype=np.float64)
    for _ in range(iters):
        d = np.linalg.norm(xy - c, axis=1)
        w = np.exp(-((d / bandwidth) ** 2))      # gaussian kernel → densest mode
        if w.sum() < 1e-9:
            break
        c_new = (w[:, None] * xy).sum(0) / w.sum()
        if np.linalg.norm(c_new - c) < 1e-4:
            c = c_new
            break
        c = c_new
    return c


def _meanshift_annealed(xy, seed, bandwidth, floor_ref):
    """Coarse-to-fine mode seeking. Start with the (wide) bandwidth so a submission box sitting
    FAR from its real object can still reach it; then re-seed at progressively SMALLER bandwidths
    to lock precisely onto the LOCAL density peak — a person-scale cluster. Wide → narrow → decide.
    floor = min(bandwidth, floor_ref) keeps the final kernel at ~person scale (no fragmenting)."""
    c = np.asarray(seed, dtype=np.float64)
    bw_floor = min(bandwidth, max(0.2, floor_ref))
    if bandwidth <= bw_floor + 1e-6:
        return _meanshift_bev(xy, c, bandwidth)
    mid = (bandwidth * bw_floor) ** 0.5                 # geometric mid step
    for bw in (bandwidth, mid, bw_floor):
        c = _meanshift_bev(xy, c, bw)
    return c


def _hdbscan_bev(xy, seed, min_cluster_size, radius):
    labels = HDBSCAN(min_cluster_size=int(min_cluster_size)).fit_predict(xy)
    best, best_d = None, 1e9
    for lab in set(labels):
        if lab == -1:
            continue
        ctr = np.median(xy[labels == lab], axis=0)
        d = float(np.linalg.norm(ctr - seed))
        if d < best_d:
            best_d, best = d, ctr
    if best is None or best_d > radius * 3:
        return None
    return best


def _gmm_bev(xy, seed, n_components):
    n = int(min(n_components, max(1, len(xy) // 30)))
    gm = GaussianMixture(n_components=n, covariance_type="full",
                         reg_covar=1e-4, random_state=0).fit(xy)
    k = int(np.linalg.norm(gm.means_ - seed, axis=1).argmin())
    return gm.means_[k]


def _find_center(method, xy, seed, p):
    if method == "meanshift":
        return _meanshift_annealed(xy, seed, p["bandwidth"], p["radius"])
    if method == "hdbscan":
        return _hdbscan_bev(xy, seed, p["min_cluster_size"], p["radius"])
    if method == "gmm":
        return _gmm_bev(xy, seed, p["n_components"])
    raise ValueError(f"unknown method: {method}")


def _foot_extent(xy):
    """Long-axis horizontal extent of the BEV cluster (PCA), metres. Standing ≈ 0.4–0.5;
    bending/reaching elongates it. The height tracker uses this to tell legs-only
    (compact) apart from a real bend (elongated) when the top (head) is not visible."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 5:
        return 0.0
    cov = np.cov(xy.T)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return 0.0
    ev = np.linalg.eigvalsh(cov)
    return float(2.0 * np.sqrt(max(ev[-1], 0.0)))


def _pca_yaw(xy, center, w, l, min_pts=8, min_ratio=1.12):
    """Yaw (rad) that aligns the box's LONGER axis with the points' principal axis.

    PCA on the BEV points around the center: the largest-eigenvalue eigenvector is the
    object's main direction. We align the box's longer side (length if l≥w, else width)
    with it — so a fixed w×l rectangle matches the real point spread. Returns None when
    the blob is too round (no reliable orientation) → caller keeps the submission yaw.
    """
    pts = np.asarray(xy, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    if len(pts) < min_pts:
        return None
    cov = np.cov(pts.T)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None
    evals, evecs = np.linalg.eigh(cov)                 # ascending eigenvalues
    ratio = np.inf if evals[0] <= 1e-9 else float(np.sqrt(evals[-1] / evals[0]))
    if ratio < min_ratio:                              # ~round → orientation unreliable
        return None
    v = evecs[:, -1]                                   # principal (longest-spread) axis
    if l >= w:                                          # length (local y) ← principal axis
        return float(np.arctan2(-v[0], v[1]))
    return float(np.arctan2(v[1], v[0]))               # width (local x) ← principal axis


def _core_center_size(xy, cell=0.05, dens_frac=0.2):
    """Connectivity-dense CORE of a BEV cluster, for a TIGHT, well-CENTRED footprint.

    The DA3 cloud is a one-sided surface + edge ghosts, so the raw centroid/extent is biased and
    inflated. We rasterise the gated points to a fine grid (`cell`), keep only the connected
    component of DENSE cells (≥ `dens_frac`×peak) that contains the density PEAK — dropping loose
    edge/ghost points — and return ``(center, in_core_mask)`` where center = mean of the core points
    (a cleaner centre than the noisy centroid). Returns None when the core is too small to trust."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 8:
        return None
    mn = xy.min(0)
    ij = np.floor((xy - mn) / cell).astype(int)
    W, H = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
    if W * H > 400000:                                  # pathologically spread → skip (keep caller's centre)
        return None
    grid = np.zeros((W, H), dtype=np.int32)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1)
    thr = max(1.0, grid.max() * dens_frac)
    dense = grid >= thr
    core = np.zeros_like(dense)                          # flood-fill (8-conn) from the peak cell
    pk = np.unravel_index(int(grid.argmax()), grid.shape)
    core[pk] = True
    stack = [pk]
    while stack:
        a, b = stack.pop()
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                na, nb = a + da, b + db
                if 0 <= na < W and 0 <= nb < H and dense[na, nb] and not core[na, nb]:
                    core[na, nb] = True
                    stack.append((na, nb))
    in_core = core[ij[:, 0], ij[:, 1]]
    if int(in_core.sum()) < 8:
        return None
    return xy[in_core].mean(0), in_core


def _track_key(box):
    return f"{int(box.get('class_id', -1))}:{int(box.get('object_id', -1))}"


_DBG = set(k for k in os.environ.get("RF_DEBUG", "").split(",") if k)   # e.g. RF_DEBUG="0:3,0:1"


def _dlog(frame_id, key, *msg):
    import sys
    print(f"[RF f{frame_id} {key}]", *(str(m) for m in msg), file=sys.stderr, flush=True)


def _gen_candidates(xy, seed, bandwidth, radius, merge=0.15, n_offset=4, offset_r=0.4):
    """Enumerate candidate cluster MODES (instead of one chained wide→narrow descent).

    Run mean-shift SEPARATELY at each bandwidth (wide … narrow) from the submission seed AND from a
    small ring of offset seeds — wide finds the dominant/far blob, narrow finds the LOCAL blob by the
    seed, offsets discover neighbouring blobs. Merge near-duplicate modes, score each by local
    density (`fit` = points within `radius`). Returns ``[{c, fit}]`` for the caller to pick from."""
    seed = np.asarray(seed, dtype=np.float64)
    bw_floor = min(bandwidth, max(0.2, radius))
    bws = sorted({round(float(bandwidth), 3), round(float((bandwidth * bw_floor) ** 0.5), 3),
                  round(float(bw_floor), 3)}, reverse=True)
    seeds = [seed]
    for k in range(n_offset):
        a = 2.0 * np.pi * k / max(1, n_offset)
        seeds.append(seed + offset_r * np.array([np.cos(a), np.sin(a)]))
    uniq = []
    for s in seeds:
        for bw in bws:
            try:
                c = _meanshift_bev(xy, s, bw)
            except Exception:  # noqa: BLE001
                continue
            if any(np.hypot(c[0] - u["c"][0], c[1] - u["c"][1]) < merge for u in uniq):
                continue
            fit = int(np.count_nonzero(np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) < radius))
            uniq.append({"c": c, "fit": fit})
    return uniq


def _select_candidate(cands, seed, min_pts, frac=0.3, exclude_boxes=None,
                      w=0.6, l=0.6, yaw=0.0, dedup_iou=0.2, prefer="close"):
    """Pick a VALID candidate (fit ≥ max(min_pts, frac·best_fit)) that is clear of `exclude_boxes`
    (box IoU ≤ dedup_iou). `prefer="close"` → the one nearest `seed`; `prefer="dense"` → the
    DENSEST one (= the real person blob, used when a box must move off a collision). None if none."""
    if not cands:
        return None
    best = max(c["fit"] for c in cands)
    thr = max(min_pts, frac * best)
    seed = np.asarray(seed, dtype=np.float64)
    valid = [c for c in cands if c["fit"] >= thr]
    if prefer == "dense":
        valid.sort(key=lambda c: -c["fit"])                 # densest blob first (a real person)
    else:
        valid.sort(key=lambda c: float(np.hypot(c["c"][0] - seed[0], c["c"][1] - seed[1])))
    for c in valid:
        if exclude_boxes:
            cb = (c["c"][0], c["c"][1], w, l, yaw)
            if any(_bev_iou(cb, ob) > dedup_iou for ob in exclude_boxes):
                continue
        return np.asarray(c["c"], dtype=np.float64)
    return None


# --------------------------------------------------------------------------- #
def refine(*, export_dir: str, root: str, scene: str, frame_id: int,
           method: str = "meanshift", margin: float = 0.4, radius: float = 0.35,
           bandwidth: float = 0.3, floor_z: float = 0.0, min_h: float = 1.0,
           max_h: float = 2.0, min_pts: int = 15, keep_size: bool = True,
           min_cluster_size: int = 20, n_components: int = 2,
           class_ids=None, exclude_class_ids=None, exclude_object_ids=None,
           class_sizes=None, estimate_yaw: bool = False,
           remove_static: str = "", static_voxel: float = 0.06,
           cloud_path: str = "", dedup_iou: float = 0.0, min_fit: int = 200,
           skip_refine_object_ids=None, max_roi: int = 4000,
           prev_boxes=None, prev_iou_thresh: float = 0.5, prev_local_margin: float = 0.6,
           use_prev_reseed: bool = False,
           tight_footprint: bool = False, multi_candidate: bool = False,
           motion_pred=None, motion_keep_fit: int = 200, subcrop_margin: float = 0.0,
           cloud_pts=None) -> dict:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    P = cloud_pts if cloud_pts is not None else _load_cloud(export_dir, cloud_path)
    if remove_static and os.path.isfile(remove_static):     # drop static-bg points before cluster
        from .static_point import _voxel_keys
        sp = np.asarray(o3d.io.read_point_cloud(remove_static).points)
        if len(sp) and len(P):
            sk = np.unique(_voxel_keys(sp, static_voxel))
            kk = _voxel_keys(P, static_voxel)
            loc = np.clip(np.searchsorted(sk, kk), 0, len(sk) - 1)
            P = P[sk[loc] != kk]
    sub_path = submission.find_path_for_scene(root, scene)
    if not sub_path:
        raise FileNotFoundError(f"no submission file matching scene {scene}")
    boxes = submission.get_boxes(sub_path, frame_id)["boxes"]
    if class_ids:                      # keep only these classes (e.g. [2,3])
        inc = set(class_ids)
        boxes = [b for b in boxes if b.get("class_id") in inc]
    if exclude_class_ids:              # drop these classes (e.g. person card skips 2,3)
        exc = set(exclude_class_ids)
        boxes = [b for b in boxes if b.get("class_id") not in exc]
    if exclude_object_ids:             # drop specific objects (handled by the other panel)
        exo = set(int(x) for x in exclude_object_ids)
        boxes = [b for b in boxes if int(b.get("object_id", -1)) not in exo]

    p = {"radius": radius, "bandwidth": bandwidth, "min_pts": min_pts,
         "min_cluster_size": min_cluster_size, "n_components": n_components}
    cs = {}
    for k, v in (class_sizes or {}).items():
        vals = [float(x) for x in v]
        if len(vals) != 3 or any((not np.isfinite(x)) or x <= 0 for x in vals):
            raise ValueError(f"invalid fixed size for class {k}: expected positive [w,l,h]")
        cs[int(k)] = vals
    _skip_ids = set(int(x) for x in (skip_refine_object_ids or []))  # keep these at submission pos

    # previous-frame refined position per track (class:object), threaded from the Final batch — used
    # as a TEMPORAL ANCHOR so a track stays on its OWN cluster instead of being stolen by a denser
    # neighbour when depth-bg decimates its points (see Pass 1 multi_candidate anchoring).
    _dpos = {str(k): v for k, v in (prev_boxes or {}).items()}

    def _prevxy(box):
        v = _dpos.get(_track_key(box)) or _dpos.get(str(int(box.get("object_id", -1))))
        return None if v is None else np.array([float(v[0]), float(v[1])], dtype=np.float64)

    # motion-predicted position per track (prev + velocity), threaded from the Final batch. Online
    # (only past refined frames). Drives the Pass-1 motion decision: a track stays where physics says
    # it is (prev + velocity) as long as a real cluster still sits there.
    _dpred = {str(k): v for k, v in (motion_pred or {}).items()}

    def _predxy(box):
        v = _dpred.get(_track_key(box)) or _dpred.get(str(int(box.get("object_id", -1))))
        return None if v is None else np.array([float(v[0]), float(v[1])], dtype=np.float64)

    # SUBCROP (opt-in via subcrop_margin>0; callers gate it per scene). Each box's ROI below is only
    # the SQUARE `|x-cx|,|y-cy| < max(w,l)/2 + margin` around its SUBMISSION center (line ~346, with
    # w,l = the FORCED class size when set), and the continuity window is a subset of that ROI. So
    # points outside every box's ROI are dead weight that only slow the per-box ROI scans. Drop them
    # up front by keeping the union of those squares GROWN to half-extent + subcrop_margin (∪ prev-
    # refined ± subcrop_margin for robustness). Two things make this EXACTLY identical, not approximate:
    #   • Chebyshev metric (p=inf) so the kept region is the SAME square shape as the ROI (a Euclidean
    #     circle would clip the ROI's corners — that was a real 0.07m bug);
    #   • the SAME effective size as the ROI (forced class size when set), grown by subcrop_margin
    #     (>= the ROI margin) so each ROI square is strictly contained.
    # A point in any ROI is within that box's rxy (<= max half-extent + margin < subcrop_margin) of its
    # centre, so its nearest crop centre's radius (>= subcrop_margin) always keeps it.
    if subcrop_margin > 0 and len(P) and boxes:
        from scipy.spatial import cKDTree
        cc, rr = [], []
        for b in boxes:
            bx, by, _bz = b["center"]
            bw, bl, _bh = b["size"]
            if b.get("class_id") in cs:          # forced fixed size → match the ROI's effective size
                bw, bl = cs[b["class_id"]][0], cs[b["class_id"]][1]
            cc.append((bx, by))
            rr.append(max(bw, bl) / 2.0 + subcrop_margin)
        for v in _dpos.values():                 # prev-refined continuity windows (robustness)
            cc.append((float(v[0]), float(v[1])))
            rr.append(subcrop_margin)
        rr = np.asarray(rr, dtype=np.float64)
        d, idx = cKDTree(np.asarray(cc, dtype=np.float64)).query(P[:, :2], p=np.inf)
        P = P[d < rr[idx]]

    rng = np.random.default_rng(0)

    # ---- Pass 1: per box, crop ROI + tentative cluster center (independent) ----
    prelim = []
    for b in boxes:
        cx, cy, cz = b["center"]
        cid = b.get("class_id")
        w, l, h = b["size"]
        forced = cid in cs                    # this class has a hard-fixed w,l,h
        if forced:
            w, l, h = cs[cid]
        yaw = float(b["yaw"])
        rxy = max(w, l) / 2.0 + margin
        z_hi = floor_z + (h if forced else max_h) + 0.3
        m = ((np.abs(P[:, 0] - cx) < rxy) & (np.abs(P[:, 1] - cy) < rxy)
             & (P[:, 2] > floor_z - 0.15) & (P[:, 2] < z_hi))
        roi = P[m]
        xy = roi[:, :2]
        if len(xy) > max_roi:                       # subsample → fast clustering
            xy = xy[rng.choice(len(xy), max_roi, replace=False)]
        seed = np.array([cx, cy], dtype=np.float64)
        center, cands = None, None
        # object IDs in skip_refine_object_ids keep their SUBMISSION position (no clustering)
        if int(b.get("object_id", -1)) not in _skip_ids and len(roi) >= min_pts:
            if multi_candidate and method == "meanshift":
                # CONTINUITY, REFINED-ONLY (online). The box's own cluster is LOCAL to the PREVIOUS
                # REFINED position and its points are ALWAYS there (verified: track4's own spot keeps
                # ~1000-3000 pts even while a NEIGHBOUR blob grows to 4600). The drift happened because
                # the iterative mean-shift, run over the WHOLE ROI, slid down a connected density ridge
                # to that denser neighbour. Fix: find the cluster from ONLY the points within near_r of
                # prev — the far denser neighbour is OUTSIDE this window, so the mode CANNOT slide to it.
                # The window moves with the person (it follows a real walk) but can't jump to a far blob.
                # Submission is NOT used as the anchor (it can be off); only the first frame / a long
                # loss (the orchestrator's lost-counter) falls back to it.
                cands = _gen_candidates(xy, seed, bandwidth, radius)   # for first-frame fallback + debug
                pp = _prevxy(b)                              # previous REFINED position
                gate = pred_used = None
                if pp is not None and len(xy) >= min_pts:
                    near_r = max(0.30, radius)
                    loc = xy[np.hypot(xy[:, 0] - pp[0], xy[:, 1] - pp[1]) < near_r]   # ONLY local points
                    if len(loc) >= motion_keep_fit:
                        try:
                            center = np.asarray(_meanshift_bev(loc, pp, max(0.15, radius * 0.55)),
                                                dtype=np.float64)
                        except Exception:  # noqa: BLE001
                            center = loc.mean(axis=0)
                        pred_used = True
                    else:
                        center = None      # own cluster gone (occluded) → the orchestrator's lost-counter handles
                        pred_used = False
                    motion_active = True
                else:
                    motion_active = False
                if center is None and not motion_active:        # first frame of a track → submission
                    center = _select_candidate(cands, seed, min_pts)
                if _track_key(b) in _DBG:
                    _dlog(frame_id, _track_key(b), "PASS1 seed", [round(float(x), 2) for x in seed],
                          "prev", None if pp is None else [round(float(x), 2) for x in pp],
                          "located", pred_used,
                          "-> sel", None if center is None else [round(float(x), 2) for x in center])
            else:
                try:
                    center = _find_center(method, xy, seed, p)
                except Exception:  # noqa: BLE001
                    center = None
        prelim.append({"b": b, "cx": cx, "cy": cy, "cz": cz, "w": w, "l": l, "h": h,
                       "yaw": yaw, "forced": forced, "cid": cid, "roi": roi, "xy": xy,
                       "seed": seed, "center": center, "cands": cands})

    # ---- Pass 2: de-duplicate by RE-SEPARATING merged clusters with a FINER kernel ----
    # When two boxes overlap (BEV-IoU > dedup_iou) their wide-bandwidth mean-shift has MERGED
    # onto one dense blob: a large bandwidth (needed to reach far objects) also smears two nearby
    # people into a single mode. Two close people DO have two distinguishable point blobs — we
    # just need a sharper kernel to resolve them. So on collision we re-run mean-shift for BOTH
    # from their OWN submission seeds with a progressively SMALLER bandwidth, so each climbs to
    # its LOCAL density peak instead of the global one. We accept the finest bandwidth at which
    # the two centres SEPARATE (IoU <= dedup_iou), BOTH sit on a real cluster (fit >= min_fit),
    # and neither hits another claim. If they never separate (truly one person), the higher-fit
    # box keeps the cluster and the other reverts to its submission position. dedup_iou == 0
    # skips this entirely (byte-identical to no de-dup).
    if dedup_iou and dedup_iou > 0:
        def _fit(pr, c):                                # density: ROI points around the centre
            if c is None:
                return 0
            roi = pr["roi"]
            return int(np.count_nonzero(np.hypot(roi[:, 0] - c[0], roi[:, 1] - c[1]) < radius))

        def _box(pr, c):
            return (c[0], c[1], pr["w"], pr["l"], pr["yaw"])

        bw_steps = [bandwidth * f for f in (0.5, 0.35, 0.25, 0.18)]   # sharper kernels to split

        order = sorted(range(len(prelim)), key=lambda j: -_fit(prelim[j], prelim[j]["center"]))
        claimed = []                                    # prelim indices already placed
        if multi_candidate:
            # VĐ2 + PREVIOUS-FRAME anchored separation: when two DIFFERENT track_ids collide
            # (IoU>dedup_iou) we re-seed the search at each box's PREVIOUS refined position (not the
            # submission). The box whose contested cluster is CLOSER to its OWN prev KEEPS it; the
            # other re-searches a DENSE cluster near ITS prev (clear of the kept box) — that is where
            # this person actually moved to. prev_boxes is threaded per-frame from the Final batch
            # (keyed class:object). First frame / no prev → fall back to the submission candidates.
            _dp = {str(k): v for k, v in (prev_boxes or {}).items()}

            def _prevpos(pr):
                v = _dp.get(_track_key(pr["b"])) or _dp.get(str(int(pr["b"].get("object_id", -1))))
                return None if v is None else np.array([float(v[0]), float(v[1])], dtype=np.float64)

            def _okey(j):
                # The OWNER of a contested cluster is the box that DRIFTED LEAST from its own
                # SUBMISSION (its submission already sat on a real blob). Order by that drift so the
                # rightful owner CLAIMS first; a box pulled far off its submission (its person is
                # sparse / occluded → it drifted onto a neighbour) is processed last and must move.
                # Submission is the reliable upstream track (distinct per person); prev can be drifted.
                pr = prelim[j]
                if pr["center"] is None:
                    return 1e9
                return float(np.hypot(pr["center"][0] - pr["seed"][0], pr["center"][1] - pr["seed"][1]))

            order = sorted(range(len(prelim)),
                           key=lambda j: (_okey(j), -_fit(prelim[j], prelim[j]["center"])))
            for i in order:
                pr = prelim[i]
                if pr["center"] is None:
                    continue
                cboxes = [_box(prelim[j], prelim[j]["center"]) for j in claimed]
                _coll = [(_track_key(prelim[j]["b"]), round(_bev_iou(_box(pr, pr["center"]), cb), 3))
                         for j, cb in zip(claimed, cboxes) if _bev_iou(_box(pr, pr["center"]), cb) > dedup_iou]
                if _track_key(pr["b"]) in _DBG:
                    _dlog(frame_id, _track_key(pr["b"]), "DEDUP pass1", [round(float(x), 2) for x in pr["center"]],
                          "prev", None if _prevpos(pr) is None else [round(float(x), 2) for x in _prevpos(pr)],
                          "collides_with", _coll or "NONE(keep)")
                if not _coll:
                    claimed.append(i)                   # closest-to-prev keeps the contested cluster
                    continue
                # this box must MOVE off the collision: pool its candidates from BOTH the submission
                # seed AND a local re-search around its previous position, then pick the DENSEST
                # candidate that is CLEAR of every claimed box = the real person it belongs to (not a
                # sparse in-between, not the neighbour's blob).
                pp = _prevpos(pr)
                pvc = None
                if pp is not None:
                    rloc = max(pr["w"], pr["l"]) / 2.0 + margin
                    mloc = ((np.abs(P[:, 0] - pp[0]) < rloc) & (np.abs(P[:, 1] - pp[1]) < rloc)
                            & (P[:, 2] > floor_z - 0.15) & (P[:, 2] < floor_z + max_h + 0.3))
                    xyl = P[mloc][:, :2]
                    if len(xyl) > max_roi:
                        xyl = xyl[rng.choice(len(xyl), max_roi, replace=False)]
                    if len(xyl) >= min_pts:
                        pvc = _gen_candidates(xyl, pp, bandwidth, radius)
                pool = list(pr.get("cands") or [])
                if pvc:
                    pool += pvc
                alt = _select_candidate(pool, pr["seed"], min_pts, frac=0.25, exclude_boxes=cboxes,
                                        w=pr["w"], l=pr["l"], yaw=pr["yaw"], dedup_iou=dedup_iou,
                                        prefer="dense")
                if _track_key(pr["b"]) in _DBG:
                    _dlog(frame_id, _track_key(pr["b"]), "RESEARCH ->",
                          None if alt is None else [round(float(x), 2) for x in alt],
                          "| prev_cands(x,y,fit):",
                          "n/a" if not pvc else [(round(c["c"][0], 2), round(c["c"][1], 2), c["fit"]) for c in pvc])
                pr["center"] = alt
                if alt is not None:
                    claimed.append(i)
        else:
            for i in order:
                pr = prelim[i]
                if pr["center"] is None:
                    continue
                box_i = _box(pr, pr["center"])
                coll = [j for j in claimed
                        if _bev_iou(box_i, _box(prelim[j], prelim[j]["center"])) > dedup_iou]
                if not coll:
                    claimed.append(i)
                    continue
                a = coll[0]                                 # the (first) colliding claimed box
                pa, pb = prelim[a], pr
                others = [_box(prelim[j], prelim[j]["center"]) for j in claimed if j != a]
                # try to SEPARATE the pair: each box re-climbs from its OWN seed with a finer kernel
                sep = None
                for bw in bw_steps:
                    try:
                        ca = _meanshift_bev(pa["xy"], pa["seed"], bw)
                        cb = _meanshift_bev(pb["xy"], pb["seed"], bw)
                    except Exception:  # noqa: BLE001
                        continue
                    if _fit(pa, ca) < min_fit or _fit(pb, cb) < min_fit:
                        continue                            # one landed on too few points → finer
                    ba, bb = _box(pa, ca), _box(pb, cb)
                    if _bev_iou(ba, bb) > dedup_iou:
                        continue                            # still merged → finer
                    if any(_bev_iou(ba, ob) > dedup_iou or _bev_iou(bb, ob) > dedup_iou for ob in others):
                        continue                            # collides with another claim → finer
                    sep = (ca, cb)
                    break
                if sep is not None:                         # both now on their own local peak
                    pa["center"], pb["center"] = sep
                    claimed.append(i)
                elif _fit(pb, pb["center"]) > _fit(pa, pa["center"]):
                    pa["center"] = None                     # truly one cluster: denser box keeps it
                    claimed = [j for j in claimed if j != a]
                    claimed.append(i)
                else:
                    pb["center"] = None                     # the other reverts to its submission pos

    # ---- Pass 2.5: anti background-jump via the PREVIOUS frame's refined box ----
    # Two IoU rules, kept CONSISTENT with the Pass-2 dedup:
    #   • SAME track_id, across frames → CONTINUITY (prev_iou_thresh, e.g. 0.5). A person near a wall
    #     with few points can let the mean-shift drift onto the (un-removed) wall blob → the box JUMPS
    #     out to background. If the tentative centre is FAR from where this track sat last frame
    #     (BEV-IoU < prev_iou_thresh), re-search LOCALLY around the previous position; keep the new
    #     centre only if it overlaps the previous box (IoU > prev_iou_thresh) — a real cluster IS there.
    #   • DIFFERENT track_ids → SEPARATION (dedup_iou, e.g. 0.2). The continuity re-seed must NEVER
    #     land on ANOTHER track's cluster: if the previous box was itself wrong (sitting on another
    #     person), re-seeding there would collapse two different track_ids onto one blob. So a re-seed
    #     whose box overlaps any OTHER track's current box by > dedup_iou is REJECTED, and the box
    #     keeps its OWN Pass-1/2 cluster (which lets it escape the wrong previous position).
    # prev_boxes supports {"class_id:object_id":[x,y,w,l,yaw]} and the legacy object_id key.
    # Pass 2.5 runs ONLY when explicitly enabled — prev_boxes is also passed (without it) just to
    # feed the Pass-2 dedup, so gate on use_prev_reseed (not merely on prev_boxes being present).
    _prev = {str(k): v for k, v in (prev_boxes or {}).items()}
    if _prev and use_prev_reseed:
        _col_thr = dedup_iou if (dedup_iou or 0) > 0 else 0.2   # DIFFERENT-track overlap ceiling
        for pr in prelim:
            oid = int(pr["b"].get("object_id", -1))
            pv = _prev.get(_track_key(pr["b"])) or _prev.get(str(oid))
            if pv is None:
                continue
            pvb = (float(pv[0]), float(pv[1]), pr["w"], pr["l"], pr["yaw"])
            cur = None if pr["center"] is None else (
                pr["center"][0], pr["center"][1], pr["w"], pr["l"], pr["yaw"])
            if cur is not None and _bev_iou(cur, pvb) >= prev_iou_thresh:
                continue                                  # same track already continuous with last frame
            px, py = float(pv[0]), float(pv[1])           # re-search locally around the prev box
            rloc = max(pr["w"], pr["l"]) / 2.0 + prev_local_margin
            z_hi = floor_z + (pr["h"] if pr["forced"] else max_h) + 0.3
            mloc = ((np.abs(P[:, 0] - px) < rloc) & (np.abs(P[:, 1] - py) < rloc)
                    & (P[:, 2] > floor_z - 0.15) & (P[:, 2] < z_hi))
            roi_loc = P[mloc]
            new_center = None
            if len(roi_loc) >= min_pts:
                xyl = roi_loc[:, :2]
                if len(xyl) > max_roi:
                    xyl = xyl[rng.choice(len(xyl), max_roi, replace=False)]
                try:
                    c = _meanshift_bev(xyl, np.array([px, py], dtype=np.float64), min(bandwidth, rloc))
                    if _bev_iou((c[0], c[1], pr["w"], pr["l"], pr["yaw"]), pvb) > prev_iou_thresh:
                        new_center = c                    # a real cluster sits by the prev position
                except Exception:  # noqa: BLE001
                    new_center = None
            if new_center is not None:
                cbox = (new_center[0], new_center[1], pr["w"], pr["l"], pr["yaw"])
                clash = any(                              # would this re-seed steal ANOTHER track's blob?
                    o is not pr and o["center"] is not None
                    and _bev_iou(cbox, (o["center"][0], o["center"][1], o["w"], o["l"], o["yaw"])) > _col_thr
                    for o in prelim)
                if not clash:
                    pr["center"], pr["roi"], pr["xy"] = new_center, roi_loc, roi_loc[:, :2]
                # else: prev sat on a DIFFERENT track → keep this box's own Pass-1/2 cluster (escape it)
            else:
                pr["center"] = None                       # no local cluster → submission position (prior)

    # ---- Pass 3: finalize each box (gate → height → yaw → output) ----
    out, n_refined, shifts = [], 0, []
    for pr in prelim:
        b, cx, cy, cz = pr["b"], pr["cx"], pr["cy"], pr["cz"]
        w, l, h, yaw, forced, cid = pr["w"], pr["l"], pr["h"], pr["yaw"], pr["forced"], pr["cid"]
        roi, center = pr["roi"], pr["center"]
        gated = None
        if center is not None:
            gated = roi[np.linalg.norm(roi[:, :2] - center, axis=1) < radius]  # gate FULL roi
            if len(gated) < min_pts:
                center = None
        # tight-footprint (scene 27 persons): re-centre on the connectivity-dense CORE + drop loose
        # edge/ghost points, so the box hugs the body and the centre isn't pulled by surface noise.
        if center is not None and tight_footprint and not forced:
            res = _core_center_size(gated[:, :2])
            if res is not None:
                ctr2, in_core = res
                center = np.asarray(ctr2, dtype=np.float64)
                gated = gated[in_core]

        # Raw height-tracker measurements (z_top robust to spikes; foot = horiz extent).
        z_top_m, foot_m = None, None
        if center is not None:
            z_top_m = round(float(np.percentile(gated[:, 2], 95)) - floor_z, 3)
            foot_m = round(_foot_extent(gated[:, :2]), 3)

        yaw_out = yaw
        if center is None:
            cz_out = round(floor_z + h / 2.0, 3)        # EVERY box stands on the floor (bottom at z=0)
            world_xyz = [round(cx, 3), round(cy, 3), cz_out]
            size, src, shift, npts = [round(w, 3), round(l, 3), round(h, 3)], "prior", 0.0, int(len(roi))
        else:
            if forced:                                   # size is hard-fixed for this class
                size, hh = [round(w, 3), round(l, 3), round(h, 3)], h
            else:
                # box top = a high percentile of the cluster z (drops stray ghost points ABOVE the
                # head so the box matches the real cluster top, not a floating outlier).
                hh = float(np.clip(float(np.percentile(gated[:, 2], 98)) - floor_z, min_h, max_h))
                if keep_size:
                    size = [w, l, round(hh, 3)]
                else:                                       # robust extent (p2..p98) — ghost-resistant
                    cap = radius if tight_footprint else 1.0   # persons: max w,l = the radius gate (~0.35)
                    lo = min(0.25, cap)
                    ext = lambda a: float(np.clip(np.percentile(a, 98) - np.percentile(a, 2), lo, cap))
                    size = [round(ext(gated[:, 0]), 3), round(ext(gated[:, 1]), 3), round(hh, 3)]
            if estimate_yaw:                             # find real heading from the points (PCA)
                rpca = float(np.hypot(w, l)) / 2.0 + 0.1   # disk that contains the box at any angle
                near = roi[np.linalg.norm(roi[:, :2] - center, axis=1) < rpca]
                ye = _pca_yaw(near[:, :2], np.asarray(center, dtype=np.float64), w, l)
                if ye is not None:
                    yaw_out = ye
            world_xyz = [round(float(center[0]), 3), round(float(center[1]), 3),
                         round(floor_z + hh / 2.0, 3)]
            shift = round(float(np.hypot(center[0] - cx, center[1] - cy)), 3)
            src, npts = method, int(len(gated))
            n_refined += 1
            shifts.append(shift)

        out.append({
            "object_id": b.get("object_id"), "class_id": cid,
            "orig_xyz": [round(cx, 3), round(cy, 3), round(cz, 3)],
            "world_xyz": world_xyz, "size": size,
            "yaw": round(yaw_out, 4), "yaw_deg": round(float(np.degrees(yaw_out)), 1),
            "yaw_found": bool(estimate_yaw and abs(yaw_out - yaw) > 1e-6),
            "shift_m": shift, "n_points": npts, "source": src,
            "z_top": z_top_m, "foot": foot_m,
            "box_corners": gt._box_corners(world_xyz, size, [0.0, 0.0, yaw_out]),
        })

    return {
        "method": method, "frame_id": frame_id, "edges": gt.BOX_EDGES,
        "count": len(out), "n_refined": n_refined,
        "mean_shift_m": round(float(np.mean(shifts)), 3) if shifts else 0.0,
        "n_cloud": int(len(P)), "boxes": out,
    }
