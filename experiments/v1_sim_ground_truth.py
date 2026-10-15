"""V1/V2: identification against MuJoCo ground truth.

V1  inject known command dead time (0..100 ms) -> is it recovered (slope ~ 1)?
V2  sweep servo stiffness kp                     -> does T move monotonically?

Excitation = real leader trajectories from a public SO-100 dataset, mapped into
the Menagerie SO-ARM100 joint ranges.  Measurements are quantised to 4096
ticks/rev and get 0.05 deg noise.
Usage: python experiments/v1_sim_ground_truth.py <lerobot dataset path or repo>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import demodyne as dd
from demodyne.sim import ArmSim, SimServo, map_commands_to_model

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def simulate_set(sim, real, servo, n_eps, seed):
    rng = np.random.default_rng(seed)
    U, X = [], []
    for ep in real.episodes[:n_eps]:
        u, x = sim.rollout(map_commands_to_model(ep.u, sim.ctrlrange), real.fps, servo, rng)
        U.append(u)
        X.append(x)
    return dd.from_arrays(U, X, real.fps, real.joint_names)


def main(src):
    real = dd.load(src)
    sim = ArmSim()
    N = 20
    res = {"v1": [], "v2": []}

    delays = [0.0, 0.010, 0.020, 0.033, 0.050, 0.067, 0.100]
    for i, L in enumerate(delays):
        ds = simulate_set(sim, real, SimServo(dead_time=L, noise_deg=0.05), N, seed=i)
        fits = dd.fit_dataset(ds, bootstrap=20)
        for f in fits:
            res["v1"].append({"true_L": L, "joint": f.name, "L": f.params.L, "T": f.params.T,
                              "lag": f.lag, "ci_L": f.ci.get("L"), "ci_lag": f.ci.get("lag"),
                              "r2": f.r2_one_step})
        print(f"L_true={L*1e3:5.1f}ms  " + "  ".join(f"{f.name[-5:]}:{f.params.L*1e3:5.1f}/{f.lag*1e3:5.1f}" for f in fits), flush=True)

    kps = [12.5, 25, 50, 100, 200]
    for i, kp in enumerate(kps):
        ds = simulate_set(sim, real, SimServo(kp=kp, kv=5.0, noise_deg=0.05), N, seed=100 + i)
        fits = dd.fit_dataset(ds)
        for f in fits:
            res["v2"].append({"kp": kp, "joint": f.name, "L": f.params.L, "T": f.params.T,
                              "lag": f.lag, "c": f.params.c, "r2": f.r2_one_step})
        print(f"kp={kp:6.1f}  " + "  ".join(f"{f.name[-5:]}:T={f.params.T*1e3:5.1f}" for f in fits), flush=True)

    # --- summarise
    summary = {"v1": {}, "v2": {}}
    joints = real.joint_names
    for j in joints:
        rows = [r for r in res["v1"] if r["joint"] == j]
        tl = np.array([r["true_L"] for r in rows])
        for key in ("L", "lag"):
            est = np.array([r[key] for r in rows])
            slope, icpt = np.polyfit(tl, est, 1)
            resid = est - (slope * tl + icpt)
            # error after removing the plant's apparent dead time (estimated at L_true=0)
            delta_err = (est - est[0]) - (tl - tl[0])
            summary["v1"].setdefault(j, {})[key] = {
                "slope": slope, "intercept_ms": icpt * 1e3,
                "resid_rms_ms": float(np.sqrt(np.mean(resid**2)) * 1e3),
                "delta_max_abs_err_ms": float(np.max(np.abs(delta_err)) * 1e3)}
        rows = [r for r in res["v2"] if r["joint"] == j]
        kp = np.array([r["kp"] for r in rows])
        T = np.array([r["T"] for r in rows])
        lag = np.array([r["lag"] for r in rows])
        summary["v2"][j] = {"spearman_kp_T": float(spearmanr(kp, T)[0]),
                            "spearman_kp_lag": float(spearmanr(kp, lag)[0]),
                            "T_ms": (T * 1e3).round(1).tolist(), "lag_ms": (lag * 1e3).round(1).tolist()}
    json.dump({"raw": res, "summary": summary}, open(OUT / "v1_v2.json", "w"), indent=1, default=float)
    print("\nV1 summary (dead time recovery):")
    for j, s in summary["v1"].items():
        print(f"  {j:20s} L: slope={s['L']['slope']:.3f} icpt={s['L']['intercept_ms']:5.1f}ms resid={s['L']['resid_rms_ms']:.1f}ms max|dErr|={s['L']['delta_max_abs_err_ms']:.1f}ms"
              f" | lag: slope={s['lag']['slope']:.3f} resid={s['lag']['resid_rms_ms']:.1f}ms")
    print("V2 summary (stiffness):")
    for j, s in summary["v2"].items():
        print(f"  {j:20s} spearman(kp,T)={s['spearman_kp_T']:.2f} T_ms={s['T_ms']} lag_ms={s['lag_ms']}")


if __name__ == "__main__":
    main(sys.argv[1])
