"""Re-align a dataset's action column to a target action/observation alignment.

Alignment is the median dead time across reliably identified joints, in frames
(`audit.alignment_offset / h`).  Shifting the action column by k frames,
u'[t] = u[t + k], changes the measured alignment by +k (validated in
tests/test_core.py::test_shift_moves_delay_by_one_frame), so re-aligning to a
target F uses k = F - measured.  Fractional k is applied by linear
interpolation within each episode; edges are held.

The output is a LeRobot-format copy: meta/ is copied (with action min/max/mean/std
in meta/stats.json recomputed when present) and every data parquet is rewritten
with the shifted action column.  Videos are copied only with copy_videos=True;
they are unaffected by the shift.  A provenance record is written to
meta/demodyne_realign.json.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import alignment_offset
from .ident import fit_dataset
from .io import Dataset, Episode, download, load
from .model import delayed

# Median alignment of current-LeRobot `record` datasets ('.pos' naming) in the
# 233-dataset census (experiments/v6_census_analysis.py).
CENSUS_RECORD_ALIGNMENT = 0.60


def shift_array(u: np.ndarray, k: float) -> np.ndarray:
    """u'[t] = u[t + k] for every column of a (n, d) array (fractional k)."""
    u = np.asarray(u, float)
    if u.ndim == 1:
        return delayed(u, -k)
    return np.stack([delayed(u[:, i], -k) for i in range(u.shape[1])], 1)


def shift_dataset(ds: Dataset, k: float) -> Dataset:
    """In-memory copy of `ds` with every episode's actions shifted by k frames."""
    eps = [Episode(e.index, e.t.copy(), shift_array(e.u, k), e.x.copy(), e.h) for e in ds.episodes]
    return Dataset(ds.repo_id, ds.fps, list(ds.joint_names), eps, ds.info, ds.action_names, ds.state_names,
                   list(ds.notes))


def measure_alignment(ds: Dataset, fits=None) -> float:
    """Measured action/observation alignment in frames (NaN if unidentifiable)."""
    fits = fits or fit_dataset(ds)
    return alignment_offset(fits) / ds.h


def _root(path_or_repo, cache_dir=None) -> Path:
    p = Path(str(path_or_repo).strip())
    if (p / "meta" / "info.json").exists():
        return p
    return download(str(path_or_repo).strip(), cache_dir)


def _update_stats(meta_dir: Path, actions: np.ndarray, key: str):
    f = meta_dir / "stats.json"
    if not f.exists():
        return
    stats = json.loads(f.read_text())
    if key not in stats:
        return
    s = stats[key]
    for name, val in (("min", actions.min(0)), ("max", actions.max(0)), ("mean", actions.mean(0)),
                      ("std", actions.std(0))):
        if name in s:
            s[name] = val.tolist()
    f.write_text(json.dumps(stats, indent=2))


def realign(path_or_repo, out_dir, target_frames: float | None = CENSUS_RECORD_ALIGNMENT,
            shift_frames: float | None = None, max_episodes: int | None = 60, copy_videos: bool = False,
            action_key: str = "action", cache_dir=None, episodes=None) -> dict:
    """Write a re-aligned copy of a LeRobot dataset. Returns the provenance record.

    Exactly one of `shift_frames` (apply this shift) or `target_frames` (measure,
    then shift to reach this alignment) determines the shift.  `episodes` (an
    iterable of episode indices) restricts both the measurement and the shift to
    those episodes, e.g. the part of a merged dataset that came from another
    recording pipeline; other episodes are copied unchanged.
    """
    ep_set = None if episodes is None else {int(e) for e in episodes}
    root = _root(path_or_repo, cache_dir)
    out = Path(out_dir)
    if out.resolve() == root.resolve():
        raise ValueError("out_dir must differ from the source dataset")
    measured = float("nan")
    if shift_frames is None:
        ds = load(str(root), max_episodes=None if ep_set else max_episodes, action_key=action_key)
        if ep_set is not None:
            ds = ds.subset([i for i, e in enumerate(ds.episodes) if e.index in ep_set])
        measured = measure_alignment(ds)
        if not np.isfinite(measured):
            raise ValueError("alignment is not identifiable for this dataset (too few well-fit joints)")
        shift_frames = float(target_frames) - measured
    k = float(shift_frames)

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(root / "meta", out / "meta")
    all_actions = []
    for f in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(f).reset_index(drop=True)
        if action_key not in df.columns:
            raise ValueError(f"{f}: no '{action_key}' column")
        dtype = np.asarray(df[action_key].iloc[0]).dtype
        new = [None] * len(df)
        for e, g in df.groupby("episode_index", sort=False):
            g = g.sort_values("frame_index")
            A = np.stack(g[action_key].values).astype(float)
            S = (shift_array(A, k) if ep_set is None or int(e) in ep_set else A).astype(dtype)
            for pos, row in zip(g.index, S):
                new[pos] = row
            all_actions.append(S)
        df[action_key] = new
        dest = out / f.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
    if all_actions:
        _update_stats(out / "meta", np.concatenate(all_actions), action_key)
    if copy_videos and (root / "videos").exists():
        shutil.copytree(root / "videos", out / "videos")
    from . import __version__
    rec = {"source": str(path_or_repo), "measured_alignment_frames": measured, "target_alignment_frames":
           target_frames if np.isfinite(measured) else None, "shift_frames": k,
           "convention": "action'[t] = action[t + shift_frames], linear interpolation, edges held",
           "videos_copied": bool(copy_videos and (root / "videos").exists()), "demodyne_version": __version__,
           "episodes": None if ep_set is None else sorted(ep_set)}
    (out / "meta" / "demodyne_realign.json").write_text(json.dumps(rec, indent=2))
    return rec
