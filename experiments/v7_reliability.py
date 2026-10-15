"""V7: test-retest reliability of identified timing on real community data.

For every analysed census dataset with >= 10 episodes, fit odd and even episodes
separately.  Per joint, report the intraclass correlation ICC(1,1) of effective
lag and dead time across datasets, and the median |half A - half B| difference.
High ICC = between-arm differences dwarf measurement noise.
Usage: python experiments/v7_reliability.py   (uses the local dataset cache only)
"""
import json
from pathlib import Path

import numpy as np

import demodyne as dd
from demodyne.fleet import CANON_SO, canon_joint, load_results

RES = Path(__file__).parent / "results"


def icc1(a, b):
    x = np.c_[a, b]
    n, k = x.shape
    gm = x.mean()
    msb = k * np.sum((x.mean(1) - gm) ** 2) / (n - 1)
    msw = np.sum((x - x.mean(1, keepdims=True)) ** 2) / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)


def main():
    pairs = {j: {"lag": [], "L": [], "T": []} for j in CANON_SO[:5]}
    for r in load_results(RES / "census_final"):
        if not r.get("ok") or r["n_episodes"] < 10:
            continue
        try:
            ds = dd.load(r["repo_id"], max_episodes=60)
        except Exception:
            continue
        A = ds.subset(range(0, len(ds.episodes), 2))
        B = ds.subset(range(1, len(ds.episodes), 2))
        fa, fb = dd.fit_dataset(A), dd.fit_dataset(B)
        for x, y in zip(fa, fb):
            j = canon_joint(x.name)
            if j not in pairs:
                continue
            ok = all(np.isfinite(v) for v in (x.lag, y.lag)) and min(x.r2_one_step, y.r2_one_step) >= 0.8 \
                and min(x.motion_fraction, y.motion_fraction) >= 0.15
            if ok:
                pairs[j]["lag"].append((x.lag, y.lag))
                pairs[j]["L"].append((x.params.L, y.params.L))
                pairs[j]["T"].append((x.params.T, y.params.T))
        print(".", end="", flush=True)
    print()
    out = {}
    for j, d in pairs.items():
        if len(d["lag"]) < 5:
            continue
        res = {}
        for k, v in d.items():
            v = np.array(v)
            res[k] = {"n": len(v), "icc": float(icc1(v[:, 0], v[:, 1])),
                      "median_abs_diff_ms": float(np.median(np.abs(v[:, 0] - v[:, 1])) * 1e3),
                      "between_sd_ms": float(v.mean(1).std() * 1e3)}
        out[j] = res
        print(f"{j:14s} n={res['lag']['n']:3d}  lag ICC={res['lag']['icc']:.2f} (|dA-B| med {res['lag']['median_abs_diff_ms']:.1f} ms,"
              f" between-arm sd {res['lag']['between_sd_ms']:.1f} ms)   L ICC={res['L']['icc']:.2f}   T ICC={res['T']['icc']:.2f}")
    json.dump(out, open(RES / "v7_reliability.json", "w"), indent=1)


if __name__ == "__main__":
    main()
