"""Per-track person HEIGHT stabiliser for the Final batch pipeline.

mean-shift already fixes (x,y) and keeps (w,l); this module only stabilises HEIGHT.
A single-frame refine gives a noisy height because the DA3 cloud ghosts + partial views:
often only the legs are seen (z_top caps ~1.2 m), sometimes only the head, sometimes the
person genuinely bends to pick something up. We track each object_id across frames:

  * head visible (z_top reaches up to the head)  -> trust z_top (real top -> floor)
  * legs only (low z_top + COMPACT footprint)     -> HOLD the learned standing height
  * bending (low z_top + ELONGATED footprint, sustained) -> follow (allow < 1.6)
  * glitch (too few points)                       -> HOLD

H_stand (the standing height memory) is learned by an EWMA with a decaying-then-floored
gain, ONLY from head-visible standing frames (both directions, so init 1.7 can settle to a
true 1.62). See the design discussion. Pure-python + numpy-free so it is trivial to unit test.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HTParams:
    # NO fixed prior — the standing height is SEEDED from the first real observation of the
    # object (the first frame whose top is visible) and accumulated from there.
    h_min: float = 1.30        # loose sanity clamp for the learned standing height
    h_max: float = 2.10
    npmin: int = 40            # < this many gated points => unreliable frame
    head_abs: float = 1.40     # z_top >= this (and >= H-head_rel) => head/top visible
    head_rel: float = 0.20
    bend_drop: float = 0.03    # min per-frame z_top descent that counts as "head edging down"
    bend_max_step: float = 0.35  # a descent BIGGER than this in one frame = sudden occlusion, NOT a bend
    bend_k: int = 2            # consecutive descending frames -> confirm bending (small = fast)
    n0: int = 2                # accumulation strength (pseudo-observations) after the seed
    gain_cap: int = 13         # gain floors at 1/(gain_cap+n0) -> stays adaptive
    rate_up: float = 0.25      # max height RISE per frame (standing up smoothing)
    rate_dn: float = 0.20      # max height drop per frame when NOT bending
    bend_rate_dn: float = 0.50  # follow the head FAST while bending (video changes quickly)


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class HeightTracker:
    """Causal (online) per-track height filter. Feed frames in time order via step().

    No fixed prior: the baseline ``H`` is SEEDED from the first frame whose top (head) is
    visible — i.e. the REAL height first observed — then accumulated (EWMA) from clean
    standing frames so it tracks the true points. Bend is detected purely from the TOP
    point (z_top) descending gradually over consecutive frames (a crouch doesn't always
    widen the footprint): a sustained gradual descent => follow the head down fast; a single
    sudden drop (head occluded) is NOT a descent trend => HOLD the baseline (legs-only)."""

    def __init__(self, params: HTParams | None = None):
        self.p = params or HTParams()
        self.H = None              # standing-height baseline (None until seeded)
        self.out = None            # smoothed output height (None until first observation)
        self.z_prev = None         # previous reliable z_top (for the descent trend)
        self.n_head = 0            # count of head-visible standing frames (for the gain)
        self.desc = 0              # consecutive descending-frame counter
        self.bend_state = False    # currently in a bent posture

    def _ret(self, state, head_seen, is_bend, dz):
        h = self.out if self.out is not None else 0.0
        return {"height": round(h, 3), "state": state,
                "H_stand": round(self.H if self.H is not None else h, 3),
                "head_seen": bool(head_seen), "is_bend": bool(is_bend), "dz": round(dz, 3)}

    def step(self, z_top, foot, npts):
        p = self.p
        z_top = 0.0 if z_top is None else float(z_top)
        npts = 0 if npts is None else int(npts)
        reliable = npts >= p.npmin
        # before a baseline exists, "head visible" uses the absolute threshold only
        href = p.head_abs if self.H is None else max(p.head_abs, self.H - p.head_rel)
        head_seen = reliable and z_top >= href

        # ---- SEED: baseline = the REAL height the first time the top/head is observed ----
        if self.H is None:
            if head_seen:
                self.H = self.out = self.z_prev = z_top   # initial observed height = reference
                self.n_head = 1
                return self._ret("seed", True, False, 0.0)
            # not seeded yet → just echo the raw observation (track what we actually see)
            if reliable:
                self.out = z_top if self.out is None else self.out + _clip(z_top - self.out, -p.rate_dn, p.rate_up)
                self.z_prev = z_top
            return self._ret("wait", head_seen, False, 0.0)

        # ---- bend = sustained GRADUAL descent of the top point (head), no footprint ----
        dz = (z_top - self.z_prev) if (reliable and self.z_prev is not None) else 0.0
        if not self.bend_state:
            if reliable and -p.bend_max_step <= dz < -p.bend_drop:
                self.desc += 1                      # GRADUAL descent = head edging down (a bend)
            elif reliable and dz < -p.bend_max_step:
                self.desc = 0                       # SUDDEN big drop = head occluded, NOT a bend
            else:
                self.desc = max(0, self.desc - 1)
            if self.desc >= p.bend_k:               # enough consecutive descents -> bending
                self.bend_state = True
        elif head_seen:                             # head climbed back near standing -> stood up
            self.bend_state = False
            self.desc = 0
        is_bend = self.bend_state
        if reliable:
            self.z_prev = z_top

        # accumulate the baseline from clean standing frames (EWMA from the seed)
        if head_seen and not is_bend:
            self.n_head += 1
            a = 1.0 / (min(self.n_head, p.gain_cap) + p.n0)   # 1/3 -> 1/15 then flat
            self.H = _clip(self.H + a * (z_top - self.H), p.h_min, p.h_max)

        if is_bend:
            state, target, rdn = "bend", z_top, p.bend_rate_dn   # follow descending head, FAST
        elif head_seen:
            state, target, rdn = "head", z_top, p.rate_dn        # head visible -> trust points
        else:
            state, target, rdn = "hold", self.H, p.rate_dn       # legs-only / glitch -> HOLD

        self.out += _clip(target - self.out, -rdn, p.rate_up)
        return self._ret(state, head_seen, is_bend, dz)


_HT_INT = {"npmin", "bend_k", "n0", "gain_cap"}


def make_params(d):
    """HTParams from a dict, ignoring unknown keys (old fr_hi/af configs won't break)."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(HTParams)}
    kw = {}
    for k, v in (d or {}).items():
        if k in names and v is not None:
            kw[k] = int(v) if k in _HT_INT else float(v)
    return HTParams(**kw)


def track_sequence(obs, params: HTParams | None = None, forward_backward: bool = True):
    """Stabilise one track. ``obs`` = list of {frame_id, z_top, foot, npts} (any order).
    Returns the same list (sorted by frame_id) with height/state merged in. If
    ``forward_backward`` (offline), also runs a reverse pass and averages the heights —
    better recovery at occlusion boundaries (offline is allowed and scores ~2x online)."""
    p = params or HTParams()
    obs = sorted(obs, key=lambda o: o["frame_id"])
    t = HeightTracker(p)
    fwd = [t.step(o.get("z_top"), o.get("foot"), o.get("npts")) for o in obs]
    if not forward_backward:
        return [{**o, **r} for o, r in zip(obs, fwd)]
    t2 = HeightTracker(p)
    bwd = [t2.step(o.get("z_top"), o.get("foot"), o.get("npts")) for o in reversed(obs)]
    bwd.reverse()
    out = []
    for o, rf, rb in zip(obs, fwd, bwd):
        h = round(0.5 * (rf["height"] + rb["height"]), 3)
        st = rf["state"] if rf["state"] != "hold" else rb["state"]
        out.append({**o, **rf, "height": h, "state": st})
    return out
