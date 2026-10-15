"""Identify per-joint servo models from (command, response) episodes.

The estimator is a prediction-error method on the one-step model in model.py:
  1. grid search over fractional dead time with robust (IRLS-Huber) linear least
     squares for the gain/offset at each candidate delay,
  2. nonlinear refinement of (L, T, c, db, vmax) with a soft-L1 loss so that
     contact / stall segments (e.g. a gripper closed on an object) do not bias the
     free-space dynamics,
  3. episode-level bootstrap of the *whole* estimator for confidence intervals.

Dead time L and time constant T trade off against each other at low sampling
rates (both are often below two frames at 30 Hz).  Their sum, the effective lag
L + T (the mean delay of the FOPDT impulse response), is much better determined
and is the headline timing quantity we report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .model import ServoParams, simulate

COARSE_GRID = np.arange(-3.0, 8.01, 0.25)  # frames


def stall_mask(u, x, span, min_frames=4, err_frac=0.04):
    """Samples where the joint is held away from its command without moving.

    Signature of contact: a gripper closed on an object, an arm pressing on the
    table, or a servo that has hit its torque limit.  Returns a boolean mask.
    """
    e = np.abs(u - x)
    dx = np.abs(np.diff(x, append=x[-1]))
    raw = (e > err_frac * span) & (dx < 0.004 * span)
    if not raw.any():
        return raw
    # keep runs of at least min_frames, then dilate by 2 frames
    out = np.zeros_like(raw)
    k = 0
    n = len(raw)
    while k < n:
        if raw[k]:
            m = k
            while m < n and raw[m]:
                m += 1
            if m - k >= min_frames:
                out[max(0, k - 2):min(n, m + 2)] = True
            k = m
        else:
            k += 1
    return out


class Stacked:
    """All episodes of one joint concatenated, with per-sample episode bounds."""

    def __init__(self, episodes, j, exclude_stalls=True):
        us, xs, lo, hi, nxt = [], [], [], [], []
        off = 0
        self.span = float(max(np.ptp(np.concatenate([ep.x[:, j] for ep in episodes])), 0.0)) if episodes else 0.0
        n_stall = 0
        for ep in episodes:
            n = len(ep.u)
            us.append(ep.u[:, j])
            xs.append(ep.x[:, j])
            lo.append(np.full(n, off))
            hi.append(np.full(n, off + n - 1))
            m = np.ones(n, bool)
            m[-1] = False
            if exclude_stalls and self.span > 0:
                st = stall_mask(ep.u[:, j], ep.x[:, j], self.span)
                n_stall += int(st.sum())
                m &= ~st
                m[:-1] &= ~st[1:]
            nxt.append(m)
            off += n
        self.stall_fraction = n_stall / max(off, 1)
        self.u = np.concatenate(us)
        self.x = np.concatenate(xs)
        self.lo = np.concatenate(lo).astype(float)
        self.hi = np.concatenate(hi).astype(float)
        self.has_next = np.concatenate(nxt)
        self.idx = np.arange(len(self.u), dtype=float)
        self.k = np.flatnonzero(self.has_next)
        self.dx = self.x[self.k + 1] - self.x[self.k]

    def delayed(self, d):
        src = np.clip(self.idx - d, self.lo, self.hi)
        return np.interp(src, self.idx, self.u)

    def valid(self, d):
        """Samples whose delayed command is not clamped at an episode edge."""
        src = self.idx[self.k] - d
        return (src >= self.lo[self.k]) & (src <= self.hi[self.k])

    def predict_dx(self, p: ServoParams, h):
        ud = self.delayed(p.L / h)
        a = 1.0 - np.exp(-h / max(p.T, 1e-6))
        e = ud[self.k] - p.c - self.x[self.k]
        if p.db > 0:
            e = np.sign(e) * np.maximum(np.abs(e) - p.db, 0.0)
        dx = a * e
        if np.isfinite(p.vmax):
            dx = np.clip(dx, -p.vmax * h, p.vmax * h)
        return dx


@dataclass
class JointFit:
    name: str
    params: ServoParams
    r2_one_step: float          # fraction of per-step motion variance explained
    nrmse_free_run: float       # free-run RMSE / std(state)
    rmse_free_run: float        # free-run RMSE in native units
    baseline_rmse: float        # RMSE of the naive "state equals command" model
    n_samples: int
    ci: dict = field(default_factory=dict)  # 90% bootstrap intervals
    motion_fraction: float = 0.0
    range_: float = 0.0
    stall_fraction: float = 0.0  # fraction of samples held off-command without motion (contact)

    @property
    def lag(self) -> float:
        return self.params.L + self.params.T

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "params": self.params.to_dict(),
            "lag": float(self.lag),
            "r2_one_step": float(self.r2_one_step),
            "nrmse_free_run": float(self.nrmse_free_run),
            "rmse_free_run": float(self.rmse_free_run),
            "baseline_rmse": float(self.baseline_rmse),
            "n_samples": int(self.n_samples),
            "ci": {k: [float(v[0]), float(v[1])] for k, v in self.ci.items()},
            "motion_fraction": float(self.motion_fraction),
            "range": float(self.range_),
            "stall_fraction": float(self.stall_fraction),
        }


def _irls(e0, dx, iters=5):
    """Robust fit of dx = a*e0 + b. Returns (a, b, huber_cost)."""
    A = np.c_[e0, np.ones_like(e0)]
    w = np.ones_like(e0)
    coef = np.zeros(2)
    for _ in range(iters):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], dx * sw, rcond=None)
        r = dx - A @ coef
        k = 1.345 * (1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12)
        w = np.where(np.abs(r) <= k, 1.0, k / np.maximum(np.abs(r), 1e-300))
    r = np.abs(dx - A @ coef)
    k = 1.345 * (1.4826 * np.median(r) + 1e-12)
    cost = np.mean(np.where(r <= k, 0.5 * r**2, k * (r - 0.5 * k)))
    return coef[0], coef[1], cost


def _grid_init(S: Stacked, h):
    """Coarse-then-fine fractional delay search. Returns (d_frames, a, b) or None."""
    def evaluate(d):
        v = S.valid(d)
        if v.sum() < 20:
            return np.inf, None
        e0 = (S.delayed(d)[S.k] - S.x[S.k])[v]
        a, b, cost = _irls(e0, S.dx[v])
        if not (0 < a < 1):
            return np.inf, None
        # normalise by count so different valid-set sizes are comparable
        return cost, (a, b)

    best = (np.inf, None, None)
    for d in COARSE_GRID:
        c, s = evaluate(d)
        if c < best[0]:
            best = (c, d, s)
    if best[1] is None:
        return None
    for d in np.arange(best[1] - 0.25, best[1] + 0.251, 0.05):
        c, s = evaluate(d)
        if c < best[0]:
            best = (c, d, s)
    return best[1], best[2][0], best[2][1]


def _refine(S: Stacked, h, p0: ServoParams, rng_span, max_nfev=150):
    scale = np.std(S.dx) + 1e-12
    dxq = np.quantile(np.abs(S.dx), 0.995) / h + 1e-9
    db_max = 0.05 * rng_span + 1e-9
    c_max = 0.25 * rng_span + 1e-9

    def unpack(th):
        return ServoParams(L=th[0], T=float(np.exp(th[1])), c=th[2], db=th[3], vmax=float(np.exp(th[4])))

    def resid(th):
        q = unpack(th)
        return (S.dx - S.predict_dx(q, h)) / scale

    lo = np.array([COARSE_GRID[0] * h, np.log(2e-3), -c_max, 0.0, np.log(dxq * 0.5)])
    hi = np.array([COARSE_GRID[-1] * h, np.log(5.0), c_max, db_max, np.log(dxq * 20)])
    vm0 = p0.vmax if np.isfinite(p0.vmax) else dxq * 1.5
    th0 = np.array([p0.L, np.log(max(p0.T, 2e-3)), p0.c, p0.db, np.log(vm0)])
    th0 = np.clip(th0, lo + 1e-12, hi - 1e-12)
    sol = least_squares(resid, th0, bounds=(lo, hi), loss="soft_l1", f_scale=1.0,
                        x_scale="jac", max_nfev=max_nfev)
    return unpack(sol.x)


def _estimate(S: Stacked, h, rng_span, refine=True):
    init = _grid_init(S, h)
    if init is None:
        return None
    d, a, b = init
    p = ServoParams(L=d * h, T=-h / np.log(1 - a), c=float(np.clip(-b / a, -0.25 * rng_span, 0.25 * rng_span)))
    if refine:
        try:
            p = _refine(S, h, p, rng_span)
        except Exception:
            pass
    return p


def fit_joint(episodes, j, name="j", h=None, bootstrap=0, seed=0, refine=True) -> JointFit:
    h = h or episodes[0].h
    S = Stacked(episodes, j)
    span = float(np.ptp(S.x)) if len(S.x) else 0.0
    nan = ServoParams(L=np.nan, T=np.nan)
    if span <= 0 or np.std(S.dx) == 0:
        return JointFit(name, nan, np.nan, np.nan, np.nan, np.nan, len(S.x), range_=span)
    p = _estimate(S, h, span, refine)
    if p is None:
        return JointFit(name, nan, np.nan, np.nan, np.nan, np.nan, len(S.x), range_=span)

    pred = S.predict_dx(p, h)
    r2 = 1 - np.sum((S.dx - pred) ** 2) / max(np.sum((S.dx - S.dx.mean()) ** 2), 1e-12)
    fr, base = [], []
    for ep in episodes:
        u, x = ep.u[:, j], ep.x[:, j]
        fr.append(x - simulate(p, u, x[0], h))
        base.append(x - u)
    fr = np.concatenate(fr)
    base = np.concatenate(base)
    rmse = float(np.sqrt(np.mean(fr**2)))
    tol = 1e-3 * span
    mf = float(np.mean((np.abs(S.dx) > tol)))
    fit = JointFit(name, p, float(r2), rmse / (np.std(S.x) + 1e-12), rmse,
                   float(np.sqrt(np.mean(base**2))), len(S.x), motion_fraction=mf, range_=span,
                   stall_fraction=S.stall_fraction)

    if bootstrap and len(episodes) >= 4:
        rng = np.random.default_rng(seed)
        Ls, Ts, lags = [], [], []
        for _ in range(bootstrap):
            idx = rng.integers(0, len(episodes), len(episodes))
            Sb = Stacked([episodes[k] for k in idx], j)
            try:
                pb = _refine(Sb, h, p, span, max_nfev=60) if refine else _estimate(Sb, h, span, False)
            except Exception:
                pb = None
            if pb is not None and np.isfinite(pb.L):
                Ls.append(pb.L)
                Ts.append(pb.T)
                lags.append(pb.L + pb.T)
        if Ls:
            q = [0.05, 0.95]
            fit.ci = {"L": tuple(np.quantile(Ls, q)), "T": tuple(np.quantile(Ts, q)),
                      "lag": tuple(np.quantile(lags, q))}
    return fit


def fit_dataset(ds, joints=None, bootstrap=0, max_episodes=None, refine=True):
    eps = ds.episodes[:max_episodes] if max_episodes else ds.episodes
    joints = range(ds.n_joints) if joints is None else joints
    return [fit_joint(eps, j, ds.joint_names[j], ds.h, bootstrap=bootstrap, refine=refine) for j in joints]
