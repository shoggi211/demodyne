"""Fleet census: identify and audit many datasets of the same robot family.

Every leader-follower teleop dataset is an unplanned system-identification
experiment on one physical arm.  Pooling hundreds of them gives a population
view of the hardware: typical latency and servo bandwidth, the spread across
units (the right domain-randomisation range), and arms that are unlike the rest.
"""
from __future__ import annotations

import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SO_FAMILY = {"so100", "so101", "so100_follower", "so101_follower", "so_follower", "so-100", "so-101",
             "so_100", "so_101", "so100follower", "so101follower"}


def discover(robot_types=SO_FAMILY, limit=20000, max_candidates=None, cache_dir=None, workers=16):
    """Return Hub repo ids of LeRobot datasets whose robot_type is in robot_types
    and whose action/state features have matching size."""
    import concurrent.futures as cf
    from huggingface_hub import HfApi, hf_hub_download

    from .io import default_cache

    cache = Path(cache_dir or default_cache()) / "_info"
    api = HfApi()
    ids = [d.id for d in api.list_datasets(filter="LeRobot", limit=limit, sort="downloads")]
    if max_candidates:
        ids = ids[:max_candidates]

    def get(r):
        try:
            p = hf_hub_download(r, "meta/info.json", repo_type="dataset", local_dir=cache / r.replace("/", "__"))
            info = json.loads(Path(p).read_text())
            rt = str(info.get("robot_type", "")).lower()
            f = info.get("features", {})
            if rt not in robot_types or "action" not in f or "observation.state" not in f:
                return None
            if f["action"].get("shape") != f["observation.state"].get("shape"):
                return None
            return r, rt, info.get("codebase_version"), info.get("fps"), info.get("total_episodes")
        except Exception:
            return None

    out = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for res in ex.map(get, ids):
            if res:
                out.append(res)
    return out


def analyze(repo_id, out_dir, max_episodes=60, bootstrap=0, cache_dir=None):
    """Identify + audit one dataset; write <out_dir>/<repo>.json. Never raises."""
    from . import load
    from .audit import audit
    from .ident import fit_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (repo_id.replace("/", "__") + ".json")
    if path.exists():
        return json.loads(path.read_text())
    rec = {"repo_id": repo_id, "ok": False}
    try:
        ds = load(repo_id, cache_dir=cache_dir, max_episodes=max_episodes)
        rec.update(robot_type=ds.robot_type, fps=ds.fps, n_episodes=len(ds.episodes),
                   n_frames=int(sum(len(e) for e in ds.episodes)), joint_names=ds.joint_names,
                   codebase_version=ds.info.get("codebase_version"), notes=ds.notes)
        fits = fit_dataset(ds, bootstrap=bootstrap)
        rec["fits"] = [f.to_dict() for f in fits]
        rec["findings"] = [f.to_dict() for f in audit(ds, fits)]
        rec["ok"] = True
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-1500:]
        transient = any(s in rec["error"] for s in ("429", "Timeout", "timed out", "URLError", "Connection"))
        if transient:
            return rec  # do not cache; a resumed run retries it
    path.write_text(json.dumps(rec, default=float))
    return rec


def run_census(repo_ids, out_dir, workers=8, **kw):
    results = []
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(analyze, r, out_dir, **kw): r for r in repo_ids}
        for i, f in enumerate(as_completed(futs)):
            try:
                rec = f.result()
            except Exception as e:  # worker crash
                rec = {"repo_id": futs[f], "ok": False, "error": repr(e)}
            results.append(rec)
            print(f"[{i+1}/{len(futs)}] {rec['repo_id']}: {'ok' if rec.get('ok') else rec.get('error','')[:80]}",
                  flush=True)
    return results


# ------------------------------------------------------------------ aggregate
CANON_SO = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def canon_joint(name: str) -> str:
    s = name.lower().replace(".pos", "")
    for c in CANON_SO:
        if c in s:
            return c
    return s


def load_results(out_dir):
    return [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("*.json"))]


def usable(rec, joint, min_r2=0.8, min_motion=0.15):
    for f in rec.get("fits", []):
        if canon_joint(f["name"]) == joint:
            p = f["params"]
            if (f["r2_one_step"] is not None and np.isfinite(f["r2_one_step"]) and f["r2_one_step"] >= min_r2
                    and f["motion_fraction"] >= min_motion and p["L"] is not None and np.isfinite(p["L"])):
                return f
    return None


def population(results, joints=CANON_SO[:5], **kw):
    """Per-joint population statistics of effective lag, dead time, time constant."""
    pop = {}
    for j in joints:
        rows = [(r["repo_id"], usable(r, j, **kw)) for r in results if r.get("ok")]
        rows = [(rid, f) for rid, f in rows if f]
        if not rows:
            continue
        lag = np.array([f["lag"] for _, f in rows])
        L = np.array([f["params"]["L"] for _, f in rows])
        T = np.array([f["params"]["T"] for _, f in rows])
        q = [0.05, 0.25, 0.5, 0.75, 0.95]
        med = np.median(lag)
        mad = 1.4826 * np.median(np.abs(lag - med))
        out = [rid for rid, f in rows if abs(f["lag"] - med) > 4 * mad and abs(f["lag"] - med) > 0.03]
        pop[j] = {"n": len(rows),
                  "lag_q": np.quantile(lag, q).tolist(), "L_q": np.quantile(L, q).tolist(),
                  "T_q": np.quantile(T, q).tolist(), "lag_mad": float(mad),
                  "outliers": out}
    return pop


def mujoco_actuator_suggestion(T, kp=50.0):
    """Overdamped position servo: closed-loop time constant ~ kv / kp (validated in
    experiments/v1_sim_ground_truth.py).  Returns kv for a chosen kp."""
    return {"kp": kp, "kv": kp * T}
