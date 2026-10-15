"""Per-joint servo response model.

A position-controlled joint driven by a recorded command stream is modelled as a
first-order-plus-dead-time (FOPDT) tracker with a rate limit, a deadband and a
constant offset:

    e[k]      = u(k*h - L) - c - x[k]                (tracking error vs delayed command)
    e_eff[k]  = sign(e) * max(|e| - db, 0)          (deadband)
    dx[k]     = clip(a * e_eff[k], -vmax*h, vmax*h) (first-order step, rate limited)
    x[k+1]    = x[k] + dx[k]

with a = 1 - exp(-h / T).  L (dead time) and T (time constant) are in seconds and
are invariant to the joint's unit convention (degrees, radians, normalized), which
is what makes them comparable across datasets recorded with different calibrations.
c, db and vmax are in the joint's native units.

L may be negative.  Physically it cannot be (state cannot respond before the
command is issued), so a negative estimate is evidence of a logging/alignment bug.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class ServoParams:
    L: float          # dead time [s]
    T: float          # time constant [s]
    c: float = 0.0    # steady offset, command minus settled state [units]
    vmax: float = np.inf  # rate limit [units/s]
    db: float = 0.0   # deadband half-width [units]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vmax"] = None if not np.isfinite(self.vmax) else float(self.vmax)
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}


def delayed(u: np.ndarray, delay_frames: float) -> np.ndarray:
    """Return u shifted by a fractional number of frames (positive = later).

    Linear interpolation between samples; edges are held.  Works on 1-D arrays.
    """
    n = len(u)
    src = np.arange(n, dtype=float) - delay_frames
    return np.interp(src, np.arange(n, dtype=float), u)


def simulate(p: ServoParams, u: np.ndarray, x0: float, h: float) -> np.ndarray:
    """Free-run simulation of the joint from x0 under command u (1-D)."""
    ud = delayed(u, p.L / h)
    a = 1.0 - np.exp(-h / max(p.T, 1e-6))
    step_lim = p.vmax * h if np.isfinite(p.vmax) else np.inf
    x = np.empty(len(u))
    x[0] = x0
    xk = x0
    for k in range(len(u) - 1):
        e = ud[k] - p.c - xk
        if p.db > 0:
            e = np.sign(e) * max(abs(e) - p.db, 0.0)
        dx = a * e
        if dx > step_lim:
            dx = step_lim
        elif dx < -step_lim:
            dx = -step_lim
        xk = xk + dx
        x[k + 1] = xk
    return x


def one_step(p: ServoParams, u: np.ndarray, x: np.ndarray, h: float) -> np.ndarray:
    """One-step-ahead prediction of x[k+1] from measured x[k] (vectorised)."""
    ud = delayed(u, p.L / h)
    a = 1.0 - np.exp(-h / max(p.T, 1e-6))
    e = ud[:-1] - p.c - x[:-1]
    if p.db > 0:
        e = np.sign(e) * np.maximum(np.abs(e) - p.db, 0.0)
    dx = a * e
    if np.isfinite(p.vmax):
        dx = np.clip(dx, -p.vmax * h, p.vmax * h)
    return x[:-1] + dx
