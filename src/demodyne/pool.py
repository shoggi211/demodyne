"""Pre-pooling compatibility check across datasets of the same robot family.

Community training sets (e.g. SmolVLA-style pretraining on hundreds of SO-100
datasets) concatenate datasets as if each action label meant the same thing.
The physics says otherwise: the frame-offset between an observation and the
command that moved it, and how many frames the arm takes to respond, differ by
dataset.  This module measures both, in *frames* (what a policy experiences),
and flags pairs that should not be pooled without re-alignment.
"""
from __future__ import annotations

import numpy as np

from .audit import alignment_offset, guess_units
from .fleet import canon_joint
from .ident import fit_dataset


def profile(ds, fits=None):
    fits = fits or fit_dataset(ds)
    good = [f for f in fits if np.isfinite(f.lag) and f.r2_one_step >= 0.8
            and not any(k in f.name.lower() for k in ("grip", "jaw"))]
    return {
        "dataset": ds.repo_id,
        "fps": ds.fps,
        "robot_type": ds.robot_type,
        "n_episodes": len(ds.episodes),
        "alignment_frames": alignment_offset(fits) / ds.h,
        "lag_frames": float(np.median([f.lag for f in good]) / ds.h) if good else float("nan"),
        "lag_frames_by_joint": {canon_joint(f.name): f.lag / ds.h for f in good},
        "units": {canon_joint(n): guess_units(np.concatenate([e.x[:, j] for e in ds.episodes]))
                  for j, n in enumerate(ds.joint_names)},
        "range": {canon_joint(n): [float(np.quantile(np.concatenate([e.x[:, j] for e in ds.episodes]), q))
                                   for q in (0.01, 0.99)] for j, n in enumerate(ds.joint_names)},
    }


def compare(profiles, align_tol=0.5, lag_tol=1.0):
    """Return list of (severity, message) issues for pooling `profiles`."""
    issues = []
    ref = profiles[0]
    fps = {p["fps"] for p in profiles}
    if len(fps) > 1:
        issues.append(("error", f"mixed fps {sorted(fps)}: action chunks span different wall-clock horizons"))
    al = np.array([p["alignment_frames"] for p in profiles])
    if np.isfinite(al).sum() >= 2 and np.nanmax(al) - np.nanmin(al) > align_tol:
        lo, hi = int(np.nanargmin(al)), int(np.nanargmax(al))
        issues.append(("error", f"action/observation alignment differs by {np.nanmax(al) - np.nanmin(al):.1f} frames "
                       f"({profiles[lo]['dataset']}: {al[lo]:+.2f}, {profiles[hi]['dataset']}: {al[hi]:+.2f}); "
                       f"shift actions to a common alignment before pooling"))
    lg = np.array([p["lag_frames"] for p in profiles])
    if np.isfinite(lg).sum() >= 2 and np.nanmax(lg) - np.nanmin(lg) > lag_tol:
        lo, hi = int(np.nanargmin(lg)), int(np.nanargmax(lg))
        issues.append(("warning", f"arm response lag differs by {np.nanmax(lg) - np.nanmin(lg):.1f} frames "
                       f"({profiles[lo]['dataset']}: {lg[lo]:.1f}, {profiles[hi]['dataset']}: {lg[hi]:.1f}); "
                       f"the same observation implies different future states"))
    for j in ref["units"]:
        us = {p["units"].get(j) for p in profiles if j in p["units"]}
        if len(us) > 1:
            issues.append(("warning", f"joint '{j}' unit/range class differs across datasets: {sorted(map(str, us))}"))
        rs = [p["range"][j] for p in profiles if j in p["range"]]
        if len(rs) >= 2:
            centers = np.array([(a + b) / 2 for a, b in rs])
            spans = np.array([b - a for a, b in rs])
            if np.ptp(centers) > 0.5 * np.median(spans) and np.median(spans) > 0:
                issues.append(("warning", f"joint '{j}' operating ranges are offset by {np.ptp(centers):.1f} units "
                               f"(median span {np.median(spans):.1f}): different calibration zeros?"))
    return issues
