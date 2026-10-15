"""Load LeRobot-format datasets (v2.0 / v2.1 / v3.0) as command/response episodes.

Only tabular data is read (meta/info.json + data/**/*.parquet); videos are never
downloaded, so a dataset of hundreds of episodes is typically a few MB.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Episode:
    index: int
    t: np.ndarray   # timestamps [s], shape (n,)
    u: np.ndarray   # command (action), shape (n, J)
    x: np.ndarray   # response (observation.state), shape (n, J)
    h: float        # nominal sample period [s]

    def __len__(self):
        return len(self.t)


@dataclass
class Dataset:
    repo_id: str
    fps: float
    joint_names: list
    episodes: list
    info: dict = field(default_factory=dict, repr=False)
    action_names: list = field(default_factory=list)
    state_names: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def h(self) -> float:
        return 1.0 / self.fps

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    @property
    def robot_type(self) -> str:
        return str(self.info.get("robot_type"))

    def subset(self, idx) -> "Dataset":
        return Dataset(self.repo_id, self.fps, self.joint_names, [self.episodes[i] for i in idx],
                       self.info, self.action_names, self.state_names, list(self.notes))


def _names(feat: dict, dim: int) -> list:
    n = feat.get("names")
    if isinstance(n, dict):
        for v in n.values():
            if isinstance(v, list) and len(v) == dim:
                n = v
                break
        else:
            n = None
    if isinstance(n, list) and len(n) == dim and all(isinstance(s, str) for s in n):
        return list(n)
    return [f"dim{i}" for i in range(dim)]


def _canon(name: str) -> str:
    s = name.lower()
    for pre in ("main_", "follower_", "leader_"):
        if s.startswith(pre):
            s = s[len(pre):]
    for suf in (".pos", "_pos", ".position"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def default_cache() -> Path:
    return Path(os.environ.get("DEMODYNE_CACHE", Path.home() / ".cache" / "demodyne"))


def _get(repo_id: str, fname: str, dest: Path, retries: int = 8):
    """Single GET per file (no HEAD), honouring 429 Retry-After."""
    import time
    import urllib.error
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{fname}"
    headers = {"User-Agent": "demodyne/0.1"}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    for k in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(r.read())
                tmp.replace(dest)
                return dest
        except urllib.error.HTTPError as e:
            if e.code == 429 and k < retries - 1:
                time.sleep(float(e.headers.get("Retry-After") or 30 * (k + 1)) + 1)
                continue
            raise


def download(repo_id: str, cache_dir: str | os.PathLike | None = None, max_bytes: int = 300_000_000,
             max_episode_files: int = 25) -> Path:
    """Fetch meta/info.json and tabular data of a Hub dataset (never videos).

    v3 datasets store many episodes per file and are fetched whole up to
    `max_bytes`; v2.x store one file per episode, of which the first
    `max_episode_files` are fetched.
    """
    import time

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    local = Path(cache_dir or default_cache()) / repo_id.replace("/", "__")
    if (local / ".complete").exists():
        return local
    for k in range(8):
        try:
            info = HfApi().dataset_info(repo_id, files_metadata=True)
            break
        except HfHubHTTPError as e:
            if getattr(e.response, "status_code", None) == 429 and k < 7:
                time.sleep(30 * (k + 1))
                continue
            raise
    data = sorted([s for s in info.siblings if s.rfilename.startswith("data/") and s.rfilename.endswith(".parquet")],
                  key=lambda s: s.rfilename)
    if not data:
        raise ValueError(f"{repo_id}: no parquet data files")
    if "episode_" in data[0].rfilename:
        data = data[:max_episode_files]
    else:
        keep, tot = [], 0
        for s in data:
            if tot + (s.size or 0) > max_bytes and keep:
                break
            keep.append(s)
            tot += s.size or 0
        data = keep
        if tot > max_bytes:
            raise ValueError(f"{repo_id}: first data file is {tot/1e6:.0f} MB > limit")
    for f in ["meta/info.json"] + [s.rfilename for s in data]:
        _get(repo_id, f, local / f)
    (local / ".complete").write_text("ok")
    return local


def load(path_or_repo: str, cache_dir=None, max_episodes: int | None = None,
         action_key: str = "action", state_key: str = "observation.state") -> Dataset:
    path_or_repo = str(path_or_repo).strip()
    root = Path(path_or_repo)
    repo_id = path_or_repo
    if not (root / "meta" / "info.json").exists():
        root = download(path_or_repo, cache_dir)
    info = json.loads((root / "meta" / "info.json").read_text())
    feats = info.get("features", {})
    if action_key not in feats or state_key not in feats:
        raise ValueError(f"dataset lacks '{action_key}' or '{state_key}' features")
    adim = int(np.prod(feats[action_key]["shape"]))
    sdim = int(np.prod(feats[state_key]["shape"]))
    anames, snames = _names(feats[action_key], adim), _names(feats[state_key], sdim)

    notes = []
    # Pair command and response channels by canonical name; fall back to position.
    ca, cs = [_canon(n) for n in anames], [_canon(n) for n in snames]
    pairs = [(i, cs.index(n)) for i, n in enumerate(ca) if n in cs and not n.startswith("dim")]
    if not pairs:
        if adim == sdim:
            pairs = [(i, i) for i in range(adim)]
            notes.append("channels paired by position (no matching names)")
        else:
            raise ValueError(f"cannot pair action({adim}) with state({sdim}) channels")
    elif len(pairs) < min(adim, sdim):
        notes.append(f"only {len(pairs)} of action({adim})/state({sdim}) channels paired by name")
    ai = [p[0] for p in pairs]
    si = [p[1] for p in pairs]
    joint_names = [anames[i] for i in ai]

    files = sorted((root / "data").rglob("*.parquet"))
    cols = ["timestamp", "episode_index", "frame_index", action_key, state_key]
    frames, seen = [], set()
    for f in files:
        df = pd.read_parquet(f, columns=[c for c in cols])
        frames.append(df)
        seen.update(df.episode_index.unique().tolist())
        if max_episodes and len(seen) > max_episodes:
            break
    df = pd.concat(frames, ignore_index=True)
    fps = float(info.get("fps") or 0) or 1.0 / np.median(np.diff(df.timestamp.values[:100]))
    episodes = []
    for e, g in df.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index")
        A = np.stack(g[action_key].values).astype(float)
        S = np.stack(g[state_key].values).astype(float)
        episodes.append(Episode(int(e), g.timestamp.values.astype(float), A[:, ai], S[:, si], 1.0 / fps))
        if max_episodes and len(episodes) >= max_episodes:
            break
    return Dataset(repo_id, fps, joint_names, episodes, info, anames, snames, notes)


def from_arrays(u_list, x_list, fps, joint_names=None, repo_id="arrays") -> Dataset:
    """Build a Dataset from in-memory command/response arrays (list of (n,J))."""
    eps = []
    for k, (u, x) in enumerate(zip(u_list, x_list)):
        u = np.asarray(u, float).reshape(len(u), -1)
        x = np.asarray(x, float).reshape(len(x), -1)
        eps.append(Episode(k, np.arange(len(u)) / fps, u, x, 1.0 / fps))
    J = eps[0].u.shape[1]
    return Dataset(repo_id, fps, joint_names or [f"j{i}" for i in range(J)], eps)
