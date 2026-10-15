"""Known-truth fault injection for benchmarking dataset validators.

Each corruption returns a new Dataset plus the ground-truth label the audit
should produce.  Faults mirror bugs reported in the wild (shifted action columns,
swapped motor IDs, inverted calibration, dead servos, stale sensor reads,
dropped frames, a single episode recorded under different latency).
"""
from __future__ import annotations

import copy

import numpy as np

from .io import Dataset, Episode
from .model import delayed


def _clone(ds: Dataset) -> Dataset:
    return Dataset(ds.repo_id, ds.fps, list(ds.joint_names),
                   [Episode(e.index, e.t.copy(), e.u.copy(), e.x.copy(), e.h) for e in ds.episodes],
                   copy.deepcopy(ds.info), list(ds.action_names), list(ds.state_names), list(ds.notes))


def shift_actions(ds, frames: int):
    """Mis-align the action column: u'[t] = u[t + frames].

    frames > 0: each row carries a command issued later -> apparent latency grows.
    frames < 0: each row carries a stale command, so the state appears to move
    before its command (the bug the causality check targets).
    """
    out = _clone(ds)
    for e in out.episodes:
        if len(e.u) <= abs(frames) + 1:
            continue
        e.u = np.roll(e.u, -frames, axis=0)
        if frames > 0:
            e.u[-frames:] = e.u[-frames - 1]
        elif frames < 0:
            e.u[:-frames] = e.u[-frames]
    return out, {"check": "causality" if frames < 0 else "latency_shift", "frames": frames}


def swap_channels(ds, i: int, j: int):
    out = _clone(ds)
    for e in out.episodes:
        e.x[:, [i, j]] = e.x[:, [j, i]]
    return out, {"check": "mapping", "joints": [ds.joint_names[i], ds.joint_names[j]]}


def invert_channel(ds, j: int):
    out = _clone(ds)
    for e in out.episodes:
        e.x[:, j] = 2 * np.median(e.x[:, j]) - e.x[:, j]
    return out, {"check": "mapping", "joints": [ds.joint_names[j]]}


def freeze_joint(ds, j: int, frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    out = _clone(ds)
    eps = rng.choice(len(out.episodes), max(1, int(frac * len(out.episodes))), replace=False)
    for k in eps:
        e = out.episodes[k]
        e.x[:, j] = e.x[0, j]
    return out, {"check": "frozen", "joints": [ds.joint_names[j]], "episodes": [out.episodes[k].index for k in eps]}


def stale_frames(ds, frac=0.05, seed=0):
    rng = np.random.default_rng(seed)
    out = _clone(ds)
    for e in out.episodes:
        k = rng.random(len(e.x)) < frac
        k[0] = False
        for i in np.flatnonzero(k):
            e.x[i] = e.x[i - 1]
    return out, {"check": "stale_frames"}


def drop_frames(ds, frac=0.03, seed=0):
    rng = np.random.default_rng(seed)
    out = _clone(ds)
    for e in out.episodes:
        keep = rng.random(len(e.t)) >= frac
        keep[0] = keep[-1] = True
        e.t, e.u, e.x = e.t[keep], e.u[keep], e.x[keep]
    return out, {"check": "timing"}


def episode_latency(ds, extra_s: float, n_eps=2, seed=0):
    """Re-time the response of a few episodes as if recorded with `extra_s` more
    command latency (shifting x relative to u by fractional frames)."""
    rng = np.random.default_rng(seed)
    out = _clone(ds)
    eps = rng.choice(len(out.episodes), n_eps, replace=False)
    for k in eps:
        e = out.episodes[k]
        for j in range(e.x.shape[1]):
            e.x[:, j] = delayed(e.x[:, j], extra_s / e.h)
    return out, {"check": "latency_outlier", "episodes": [out.episodes[k].index for k in eps]}


def future_state_actions(ds, k=1):
    out = _clone(ds)
    for e in out.episodes:
        e.u[:-k] = e.x[k:]
        e.u[-k:] = e.x[-1]
    return out, {"check": "action_is_future_state"}
