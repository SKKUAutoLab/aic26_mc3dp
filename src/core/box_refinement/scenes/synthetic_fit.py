
import numpy as np

from ..engine import gt, submission


def subcrop(P, items, margin=1.5):
    """Crop P (once per frame) to the union of submission box footprints + margin, so each per-box ROI
    scan below runs over ~15-25% of the cloud instead of the full 3.3M points. items = (cx,cy,w,l).
    Chebyshev (square) so it contains every box's square ROI; xy-only (keeps all z for the raise)."""
    if not items or len(P) == 0:
        return P
    from scipy.spatial import cKDTree
    cc = np.asarray([[cx, cy] for cx, cy, w, l in items], dtype=np.float64)
    rr = np.asarray([max(w, l) / 2.0 + margin for cx, cy, w, l in items], dtype=np.float64)
    d, idx = cKDTree(cc).query(P[:, :2], p=np.inf)
    return P[d < rr[idx]]


def fit_one(P, seed_xy, ref_yaw, w, l, *, cid, floor_z=0.0, sub_yaw=None, anchor_xy=None, jump=False,
            stationary=False, jump_search=0.3, jump_cap=0.2, jump_z_hi=1.0, foot_cell=0.06, foot_kbg=1.8,
            foot_min_pts=12, still_radius=0.50, still_z_hi=2.0, still_tall=1.0, still_sat_frac=0.10,
            still_yaw_win_deg=40.0, still_yaw_step_deg=6.0,
            heading=None, near_pref=5.0, ang_pref=1.5, clu_vox=0.0, **_ignore):
    """Human-shaped fit (Person/Fourier/Agility). Jump = |submission − prev refined| > jump_gate (0.2 m, set
    upstream); only FOURIER(4) is jump-handled HERE. Person(0) and Agility(5) always FOLLOW the submission in
    this function (Agility's glitch handling is the SEPARATE hold-coast-return-jump state machine in synthetic.py,
    which OVERRIDES the x,y this returns — see the module docstring).
      • NOT jumping (Person/Agility always, Fourier when calm) → FOLLOW: x,y = submission. yaw = submission
        orientation (moving) or, when STATIONARY, the cloud shoulder-PCA (x,y still = submission).
      • FOURIER(4) JUMP → BEV density-excess SLIDE ONLY (clu_vox=0, NO 3D cluster): anchor at the prev refined,
        slide the fixed box onto the densest body coverage within ±jump_search, GLIDE ≤ jump_cap from prev.
        STANDING+jump holds x,y at the anchor with shoulder-PCA yaw. yaw else = submission orientation.
      A FOURIER track stuck > jump_reset_dist (0.2 m) from the submission for > jump_reset_frames (10) frames is
      snapped back to the submission in run_final (per-track reset) — the safety net for a frozen re-localize
      (Agility is EXCLUDED from that reset; its hold-coast state machine owns its own snap/timeout)."""
    sx, sy = float(seed_xy[0]), float(seed_xy[1])
    yaw = (float(sub_yaw) - np.pi / 2.0) if (sub_yaw is not None) else float(ref_yaw)
    if not (jump and anchor_xy is not None and cid == 4):  # FOLLOW → x,y = submission (only FOURIER(4) jumps
        #                                                    here; Person/Agility always FOLLOW)
        if stationary:                                   # standing → keep x,y at submission, refine yaw via PCA
            _x, _y, fyaw, loc, n, aniso = fit_human_feet(
                P, (sx, sy), float(yaw), w, l, floor_z=floor_z, foot_hi=still_z_hi, cell=foot_cell,
                kbg=foot_kbg, min_pts=foot_min_pts, sat_frac=still_sat_frac, pca_yaw=True,
                yaw_win_deg=still_yaw_win_deg, yaw_step_deg=still_yaw_step_deg, move=False,
                crop_r=still_radius, tall_gate=still_tall, clu_vox=clu_vox)
            return {"x": sx, "y": sy, "z": floor_z, "yaw": float(fyaw if loc else yaw),
                    "anchor": "follow", "yaw_conf": float(aniso if loc else 0.0),
                    "fit": float(n), "located": bool(loc), "n_roi": int(n)}
        return {"x": sx, "y": sy, "z": floor_z, "yaw": yaw, "anchor": "follow",
                "yaw_conf": 0.0, "fit": 0.0, "located": False, "n_roi": 0}
    # FOURIER(4) JUMP (|submission − prev refined| > jump_gate, set upstream). Fourier is VERY slow, so a
    # one-frame move of the submission beyond jump_gate is a GLITCH, not motion. (Only cid==4 reaches here.)
    px, py = float(anchor_xy[0]), float(anchor_xy[1])
    ax_a, ay_a = px, py                                  # anchor = prev refined = the position before the jump
    if stationary:                                       # FOURIER standing → hold x,y at the anchor; yaw = PCA
        _x, _y, fyaw, loc, n, aniso = fit_human_feet(
            P, (ax_a, ay_a), float(ref_yaw), w, l, floor_z=floor_z, foot_hi=still_z_hi, cell=foot_cell,
            kbg=foot_kbg, min_pts=foot_min_pts, sat_frac=still_sat_frac, pca_yaw=True,
            yaw_win_deg=still_yaw_win_deg, yaw_step_deg=still_yaw_step_deg, move=False,
            crop_r=still_radius, tall_gate=still_tall, clu_vox=0.0)
        fxx, fyy, yaw_out, yc = ax_a, ay_a, float(fyaw if loc else ref_yaw), float(aniso if loc else 0.0)
    else:                                                # FOURIER moving → BEV density-excess SLIDE ONLY (NO
        # 3D cluster, clu_vox=0). The minimal pre-cluster BEV method the user kept for Fourier: it pulls the
        # fixed w×l box onto the densest body/leg coverage near prev (good for the frame-~400 Fourier-18 case).
        _x, _y, fyaw, loc, n, aniso = fit_human_feet(
            P, (ax_a, ay_a), float(yaw), w, l, floor_z=floor_z, foot_hi=float(jump_z_hi),
            cell=foot_cell, kbg=foot_kbg, min_pts=foot_min_pts, search=float(jump_search), move=True,
            heading=heading, near_pref=near_pref, ang_pref=ang_pref,
            clu_vox=0.0)
        fxx, fyy = (float(_x), float(_y)) if loc else (ax_a, ay_a)
        yaw_out, yc = float(yaw), 0.0
    dx, dy = fxx - px, fyy - py                          # GLIDE: cap the move at jump_cap from prev (no teleport)
    st = float(np.hypot(dx, dy))
    if st > float(jump_cap):
        fxx, fyy = px + dx / st * float(jump_cap), py + dy / st * float(jump_cap)
    return {"x": float(fxx), "y": float(fyy), "z": floor_z, "yaw": yaw_out, "anchor": "jump",
            "yaw_conf": yc, "fit": float(n), "located": bool(loc), "n_roi": int(n)}


def _excess_keep(E, Ew, sx, sy, eps, min_cells, sat_frac, reach):
    """Grid connected-components (DBSCAN-like) on the BEV density-excess cells E (weights Ew), then KEEP
    only the substantial cluster(s) near the submission: drop tiny specks (< min_cells), weak GHOST
    satellites (cluster density < sat_frac of the strongest), and FAR clusters (centroid > reach from the
    submission → a neighbouring vehicle, not a ghost of this one). Returns a boolean keep-mask over E."""
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(E)
    pairs = cKDTree(E).query_pairs(eps, output_type="ndarray")
    if len(pairs):
        g = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
    else:
        lab = np.arange(n)
    sums = {c: Ew[lab == c].sum() for c in set(lab.tolist())}
    dmax = max(sums.values()) if sums else 1.0
    keep = np.zeros(n, bool)
    for c, sm in sums.items():
        m = lab == c
        if m.sum() < min_cells or sm < sat_frac * dmax:           # speck or weak ghost satellite
            continue
        cen = (E[m] * Ew[m, None]).sum(0) / sm
        if np.hypot(cen[0] - sx, cen[1] - sy) > reach:            # far → neighbour vehicle, not a ghost
            continue
        keep |= m
    return keep


def fit_vehicle(P, sub_xy, prev_yaw, w, l, h, *, cid, floor_z=0.0, yaw_win_deg=30.0,
                search=0.5, cell=0.08, kbg=2.5, eps_cells=2.2, min_cells=3, sat_frac=0.12,
                beta=1.0, gamma_frac=0.4, sstep=0.08, yaw_step_deg=10.0, stick_margin=0.18,
                min_excess=4, lock_yaw=None, **_ignore):
    """BEV density-excess, GHOST-AWARE fixed-box fit (scene-25 leader redesign v2). Validated on real
    scene-25 clouds (see output/aicity2026/visualization/vehicle_bev/*). NO z-band, NO occupancy, NO
    de-contaminate — z is dropped entirely (only x,y,yaw are estimated, the box is ALWAYS placed at floor_z).
      ① BEV DENSITY — drop z, bin (x,y) into `cell` (8cm) cells, count points/cell (multi-camera depth
         noise collapses top-down → the vehicle reveals itself as a density concentration).
      ② FLOOR by uniformity — bg = median density of the outer ring; the flat floor is an even low layer.
      ③ DENSITY-EXCESS — cells with density > `kbg`·bg = the vehicle (works for SHORT vehicles too, where
         a z-cut would lose them).
      ④ GHOST DENOISE — grid connected-components on the excess cells; drop tiny specks, weak ghost
         satellites, and far-neighbour clusters (`_excess_keep`).
      ⑤ FIXED-BOX FIT — YAW = the kept cluster's PCA principal axis (refined ±yaw_win_deg); the submission
         is ONLY the position seed (never the orientation). Slide the fixed w×l box (centre BOUNDED to
         submission ± `search`) over the refine window, MAX `density_inside − beta·density_stickout −
         gamma·dist(submission)`: stick-out enforces a SNUG single-ghost match (a box spanning two ghosts
         leaks → penalised); the dist term is the nearest-centre regulariser; the ± bound prevents drift.
    Too few excess cells → trust the submission + prev_yaw."""
    sx, sy = float(sub_xy[0]), float(sub_xy[1])
    D = max(w, l)                                         # box diagonal-ish reach (for crop/ring/keep radii)
    fb = {"x": sx, "y": sy, "z": floor_z,
          "yaw": float(lock_yaw if lock_yaw is not None else prev_yaw), "fit": 0.0,
          "located": False, "anchor": "submission", "n_roi": 0}
    # ① crop a column around the submission (ALL z kept — z is only dropped at the BEV projection)
    R = D / 2.0 + search + 0.1
    col = P[(np.abs(P[:, 0] - sx) < R) & (np.abs(P[:, 1] - sy) < R)]
    if len(col) < min_excess:
        return fb
    # ② BEV density: bin (x,y), count points per cell
    mn = col[:, :2].min(0)
    gi = np.floor((col[:, :2] - mn) / cell).astype(np.int64)
    keys, counts = np.unique(gi, axis=0, return_counts=True)
    cxy = mn + (keys + 0.5) * cell
    dens = counts.astype(np.float64)
    # ③ floor background (uniform low layer) = median ring density → excess cells = the vehicle
    rr = np.hypot(cxy[:, 0] - sx, cxy[:, 1] - sy)
    ring = dens[(rr > D / 2.0 + 0.2) & (rr < D / 2.0 + 0.5)]
    bg = float(np.median(ring)) if len(ring) else 1.0
    exm = dens > kbg * max(bg, 1.0)
    E, Ew = cxy[exm], dens[exm]
    if len(E) < min_excess:
        return fb
    # ④ drop ghost satellites / far neighbours
    keep = _excess_keep(E, Ew, sx, sy, cell * eps_cells, min_cells, sat_frac, search + D / 2.0)
    Ek, Ewk = E[keep], Ew[keep]
    if len(Ek) < min_excess:
        return fb
    # ⑤ fixed-box fit, centre bounded to submission ± search. Box frame: u along th gets HALF-WIDTH w/2,
    # v along th+90 gets HALF-LENGTH l/2 — i.e. EXACTLY the eval/output convention (scale=[w,l]: w at yaw,
    # l at yaw+90), so the output yaw=th renders the same rectangle the fit chose (NO 90° flip). The yaw
    # scan over [0,π) tries both perpendicular orientations, so a vehicle (l>w) auto-aligns its long l axis.
    gamma = gamma_frac * float(np.median(Ewk))
    offs = np.arange(-search, search + 1e-9, sstep)
    Wh, Lh = w / 2.0, l / 2.0
    best = (-1e18, sx, sy, float(prev_yaw))
    if lock_yaw is not None:
        # MOVING vehicle → TRUST the submission orientation (its yaw field is rock-stable & correct while
        # driving; the point-cloud PCA jitters). LOCK yaw = lock_yaw and only search the POSITION (slide the
        # fixed-orientation box to best cover the cluster). The PCA path below is used ONLY when slow/stopped.
        yaws = [float(lock_yaw)]
    else:
        # YAW = the kept cluster's PRINCIPAL DIRECTION (weighted PCA), refined by a windowed containment
        # search (used when SLOW/STOPPED, where the submission yaw can be stale). The box's long side (l)
        # lies at yaw+90, so yaw = pca_axis − 90; refine ±yaw_win around it.
        cen = (Ek * Ewk[:, None]).sum(0) / Ewk.sum()
        dd = Ek - cen
        cov = (dd.T * Ewk) @ dd / Ewk.sum()
        _, evecs = np.linalg.eigh(cov)
        pca_axis = float(np.arctan2(evecs[1, 1], evecs[0, 1]))       # cluster principal direction
        anchor = pca_axis - np.pi / 2.0
        yw = np.radians(yaw_win_deg)
        yaws = np.arange(anchor - yw, anchor + yw + 1e-9, np.radians(yaw_step_deg))
    for th in yaws:
        c, s = np.cos(-th), np.sin(-th)
        u = Ek[:, 0] * c - Ek[:, 1] * s
        v = Ek[:, 0] * s + Ek[:, 1] * c
        for dx in offs:
            cx = sx + dx
            for dy in offs:
                cy = sy + dy
                cu, cv = cx * c - cy * s, cx * s + cy * c
                du, dv = np.abs(u - cu), np.abs(v - cv)
                inb = (du < Wh) & (dv < Lh)
                near = (du < Wh + stick_margin) & (dv < Lh + stick_margin)
                fit = Ewk[inb].sum() - beta * Ewk[near & ~inb].sum() - gamma * np.hypot(dx, dy)
                if fit > best[0]:
                    best = (fit, cx, cy, float(th))
    return {"x": float(best[1]), "y": float(best[2]), "z": floor_z, "yaw": float(best[3]),
            "fit": float(len(Ek)), "located": True, "anchor": "vehicle", "n_roi": int(len(Ek))}


def _obj_cluster_mask(col, ax, ay, vox=0.12, min_pts=12):
    """3D SPATIAL cluster: voxelise the cropped points (``vox``), connect adjacent voxels (26-neighbourhood)
    into components, and return a per-point boolean mask for the substantial component whose centroid is
    NEAREST the anchor (ax,ay) = the box's OWN person. Separates it from another track crossing the crop (two
    disjoint 3D blobs) BEFORE the BEV step, so the box can't be dragged onto the neighbour. Falls back to
    all-True when there is a single blob / too few points."""
    if len(col) < min_pts:
        return np.ones(len(col), dtype=bool)
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    vi = np.floor(col / vox).astype(np.int64)
    keys, inv = np.unique(vi, axis=0, return_inverse=True)
    if len(keys) < 2:
        return np.ones(len(col), dtype=bool)
    vc = (keys.astype(np.float64) + 0.5) * vox
    pairs = cKDTree(vc).query_pairs(vox * 1.8, output_type="ndarray")   # adjacent voxels (incl. diagonals)
    if len(pairs):
        g = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(keys), len(keys)))
        ncomp, lab = connected_components(g, directed=False)
    else:
        ncomp, lab = len(keys), np.arange(len(keys))
    pcomp = lab[inv]                                        # 3D component label per point
    best, bd = -1, 1e18
    for c in range(ncomp):
        m = pcomp == c
        if int(m.sum()) < min_pts:                         # ignore tiny specks
            continue
        d = (float(col[m, 0].mean()) - ax) ** 2 + (float(col[m, 1].mean()) - ay) ** 2
        if d < bd:
            bd, best = d, c
    if best < 0:
        return np.ones(len(col), dtype=bool)
    return pcomp == best


def fit_human_feet(P, anchor_xy, yaw, w, l, *, floor_z=0.0, foot_lo=0.0, foot_hi=0.3,
                   search=0.40, cell=0.06, kbg=1.8, eps_cells=2.4, min_cells=2, sat_frac=0.15,
                   beta=0.5, gamma_frac=0.30, sstep=0.05, stick_margin=0.10, min_pts=12,
                   pca_yaw=False, yaw_win_deg=40.0, yaw_step_deg=10.0, move=True, crop_r=None,
                   tall_gate=0.0, heading=None, near_pref=1.0, ang_pref=1.5, clu_vox=0.12):
    """HUMAN-shaped BEV density-excess fit (scene-25) — the SAME machinery as ``fit_vehicle``, two modes:
      ① crop a column around ``anchor_xy`` (radius ``crop_r`` if given, else D/2+search), keep z∈[0, foot_hi].
         JUMP uses the FEET band (foot_hi≈0.3, the legs give a precise position). STANDING crops the FULL body
         (foot_hi≈2) and ``tall_gate`` keeps only cells whose COLUMN reaches > tall_gate (≈1 m) — the upright
         body, dropping floor / low clutter (offline test: the tall-column gate was the most stable yaw source).
      ② BEV DENSITY-EXCESS: bin (x,y), background = median ring density (flat floor = an even low layer),
         keep cells > kbg·bg = the person. Ghost-denoise with ``_excess_keep`` (keep the substantial
         cluster(s) within ``reach`` of the anchor; far neighbours / specks dropped).
      ③ YAW: ``pca_yaw=False`` (JUMP) keeps the fixed ``yaw``; ``pca_yaw=True`` (STANDING) takes the kept
         cluster's weighted PCA principal axis = the body's WIDER spread = the SHOULDER line = the box width
         w, so yaw = axis (NO −90, unlike the vehicle whose LENGTH lies along its principal axis). The PCA
         eigenvalue ratio is returned as ``aniso`` = a CONFIDENCE (≈1 round/uncertain, >2 clearly oriented).
      ④ POSITION: ``move=True`` (JUMP) slides the fixed w×l box over ±search around the anchor, MAX inside −
         beta·stick-out − gamma·dist(anchor) → x,y on the real feet. ``move=False`` (STANDING) LOCKS x,y at
         the anchor (the submission) and only rotates the fixed box ±yaw_win around the PCA axis.
    Returns (x, y, yaw, located, n, aniso). Too few points / no excess → fall back to anchor + input yaw."""
    ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
    D = max(w, l)
    R = float(crop_r) if crop_r is not None else (D / 2.0 + search + 0.1)
    col = P[(np.abs(P[:, 0] - ax) < R) & (np.abs(P[:, 1] - ay) < R)
            & (P[:, 2] > floor_z + foot_lo - 0.1) & (P[:, 2] < floor_z + foot_hi)]
    if len(col) < min_pts:
        return ax, ay, float(yaw), False, int(len(col)), 1.0
    mn = col[:, :2].min(0)
    gi = np.floor((col[:, :2] - mn) / cell).astype(np.int64)
    keys, inv = np.unique(gi, axis=0, return_inverse=True)
    dens = np.bincount(inv).astype(np.float64)
    cellz = np.full(len(keys), -1e9)
    np.maximum.at(cellz, inv, col[:, 2])                 # tallest point per column = the column HEIGHT
    cxy = mn + (keys + 0.5) * cell
    rr = np.hypot(cxy[:, 0] - ax, cxy[:, 1] - ay)
    ring = dens[(rr > R * 0.65) & (rr < R * 0.98)]       # floor background ring, just inside the crop edge
    bg = float(np.median(ring)) if len(ring) else 1.0
    exm = dens > kbg * max(bg, 1.0)
    if tall_gate > 0.0:                                  # STANDING: keep only TALL columns (the upright body)
        exm = exm & (cellz > floor_z + tall_gate)
    if clu_vox > 0.0:                                    # 3D SPATIAL cluster (combined with the BEV): keep only
        objm = _obj_cluster_mask(col, ax, ay, vox=clu_vox, min_pts=min_pts)   # the box's OWN person's points,
        in_obj = np.zeros(len(keys), dtype=bool)         # reject a crossing track's blob → no drag onto it
        np.logical_or.at(in_obj, inv, objm)              # a cell is kept if any of its points is in that blob
        exm = exm & in_obj
    E, Ew = cxy[exm], dens[exm]
    if len(E) < min_cells:
        return ax, ay, float(yaw), False, int(exm.sum()), 1.0
    reach = R if not move else (search + D / 2.0)        # STANDING: keep only the person's cluster near sub
    keep = _excess_keep(E, Ew, ax, ay, cell * eps_cells, min_cells, sat_frac, reach)
    Ek, Ewk = (E[keep], Ew[keep]) if keep.any() else (E, Ew)
    if len(Ek) < min_cells:
        Ek, Ewk = E, Ew
    aniso = 1.0
    if pca_yaw and len(Ek) >= 3:                          # STANDING → yaw from the body cluster's PCA axis
        cen = (Ek * Ewk[:, None]).sum(0) / Ewk.sum()
        dd = Ek - cen
        cov = (dd.T * Ewk) @ dd / Ewk.sum()
        evals, evecs = np.linalg.eigh(cov)
        # PCA principal axis = the body's WIDER horizontal spread = the SHOULDER line = the box WIDTH (w).
        # The box's w-side lies along ``yaw``, so the SHOULDER direction IS the yaw — NO −90° (that's the
        # vehicle convention, length along its principal axis). aniso = sqrt(λ1/λ2) = orientation confidence.
        base = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
        aniso = float(np.sqrt(max(evals[1], 1e-9) / max(evals[0], 1e-12)))
        yw = np.radians(yaw_win_deg)
        yaws = np.arange(base - yw, base + yw + 1e-9, np.radians(yaw_step_deg))
    else:
        base = float(yaw)
        yaws = [base]
    Wh, Lh = w / 2.0, l / 2.0
    if not move:
        # STANDING → x,y LOCKED at the submission. yaw = rotate the FIXED w×l box AROUND the PCA shoulder axis
        # (±yaw_win), keep the orientation that best MATCHES the body cluster (MAX inside − beta·stick-out).
        best = (-1e18, base)
        for th in yaws:
            c, s = np.cos(-th), np.sin(-th)
            u = Ek[:, 0] * c - Ek[:, 1] * s
            v = Ek[:, 0] * s + Ek[:, 1] * c
            cu, cv = ax * c - ay * s, ax * s + ay * c
            du, dv = np.abs(u - cu), np.abs(v - cv)
            inb = (du < Wh) & (dv < Lh)
            near = (du < Wh + stick_margin) & (dv < Lh + stick_margin)
            sc = float(Ewk[inb].sum() - beta * Ewk[near & ~inb].sum())
            if sc > best[0]:
                best = (sc, float(th))
        return ax, ay, best[1], True, int(len(Ek)), aniso
    # MOVING-jump scoring (the box can only sit within ±search of the anchor = prev refined). ALWAYS a STRONG
    # nearer-distance bias: score = coverage / (1 + near_pref·dist) → pick the NEAREST dense cluster to prev.
    # NO HARD CONE anymore (the 45° gate was removed — it dragged the box the wrong way when a person REVERSED
    # direction, e.g. Agility:2 @f988). The velocity (heading) is now only a SOFT preference (ang_pref, spread
    # over the full 180°): a cluster in the heading direction wins ties, but a clearly-denser cluster in ANY
    # direction still wins. heading=None → pure strong-near.
    offs = np.arange(-search, search + 1e-9, sstep)
    best = (-1e18, ax, ay, float(yaw))
    for th in yaws:
        c, s = np.cos(-th), np.sin(-th)
        u = Ek[:, 0] * c - Ek[:, 1] * s
        v = Ek[:, 0] * s + Ek[:, 1] * c
        for dx in offs:
            cx = ax + dx
            for dy in offs:
                d = float(np.hypot(dx, dy))
                ang_factor = 1.0
                if heading is not None and d > 1e-3:         # SOFT velocity preference (no rejection)
                    da = (float(np.arctan2(dy, dx)) - float(heading) + np.pi) % (2 * np.pi) - np.pi
                    ang_factor = 1.0 + ang_pref * (abs(da) / np.pi)   # 0 at heading, ang_pref at 180° opposite
                cy = ay + dy
                cu, cv = cx * c - cy * s, cx * s + cy * c
                du, dv = np.abs(u - cu), np.abs(v - cv)
                inb = (du < Wh) & (dv < Lh)
                near = (du < Wh + stick_margin) & (dv < Lh + stick_margin)
                base = float(Ewk[inb].sum() - beta * Ewk[near & ~inb].sum())
                fit = base / (1.0 + near_pref * d) / ang_factor   # STRONG near + heading-aligned preference
                if fit > best[0]:
                    best = (fit, cx, cy, float(th))
    return float(best[1]), float(best[2]), float(best[3]), True, int(len(Ek)), aniso
