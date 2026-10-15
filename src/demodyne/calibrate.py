"""Calibrate a MuJoCo arm's actuators to a real dataset by fingerprint matching.

Identify the real arm's per-joint (L, T).  Then repeatedly simulate the model
under the *same* recorded commands, identify the simulator with the same
estimator, and correct:  kv <- kv * T_real / T_sim,  delay <- delay + (L_real - L_sim).
Because both sides go through the identical estimator, estimator biases (e.g.
the plant's apparent dead time at 30 fps) cancel.  Converges in a few iterations.
"""
from __future__ import annotations

import numpy as np

from .ident import fit_dataset
from .io import from_arrays
from .sim import ArmSim, CommandMap, SimServo


def simulate_dataset(sim: ArmSim, ds, servo: SimServo, cmap: CommandMap, episodes=None):
    eps = ds.episodes if episodes is None else episodes
    U, X = [], []
    for ep in eps:
        u_rad = cmap.to_model(ep.u)
        u_rad = np.clip(u_rad, sim.ctrlrange[:, 0], sim.ctrlrange[:, 1])
        _, x_deg = sim.rollout(u_rad, ds.fps, servo)
        U.append(ep.u)
        X.append(cmap.to_data(np.deg2rad(x_deg)))
    return from_arrays(U, X, ds.fps, ds.joint_names)


def calibrate(sim: ArmSim, ds, target_fits=None, iters=4, kp=None, episodes=None, verbose=False):
    """Return (SimServo, history).  `ds` channels must map 1:1 onto actuators."""
    eps = ds.episodes if episodes is None else episodes
    cmap = CommandMap(np.concatenate([e.u for e in eps]), sim.ctrlrange)
    J = sim.nu
    tgt = target_fits or fit_dataset(ds.subset(range(len(eps))) if episodes is None else
                                     type(ds)(ds.repo_id, ds.fps, ds.joint_names, eps))
    kp = np.full(J, sim.base_gain[0] if kp is None else kp, float)
    kv = -sim.base_bias[:, 2].copy()
    dead = np.zeros(J)
    best = np.full(J, np.inf)
    best_kv, best_dead = kv.copy(), dead.copy()
    hist = []
    relax = 0.7
    for it in range(iters):
        servo = SimServo(dead_time=dead.copy(), kp=kp.copy(), kv=kv.copy())
        sf = fit_dataset(simulate_dataset(sim, ds, servo, cmap, eps))
        row = []
        for j in range(J):
            t, s = tgt[j], sf[j]
            if not (np.isfinite(t.params.T) and np.isfinite(s.params.T)) or t.r2_one_step < 0.5:
                continue
            # fingerprint distance: effective lag dominates (well identified), dead time secondary
            dist = abs(s.lag - t.lag) + 0.5 * abs(s.params.L - t.params.L)
            if dist < best[j]:
                best[j], best_kv[j], best_dead[j] = dist, kv[j], dead[j]
            # dead time: move toward target; if the plant's own apparent dead time is
            # already longer than the target (dead would go < 0), give the excess to T
            new_dead = dead[j] + relax * (t.params.L - s.params.L)
            T_goal = t.params.T + min(new_dead, 0.0)
            dead[j] = np.clip(new_dead, 0.0, 0.3)
            kv[j] = np.clip(kv[j] * (max(T_goal, 2e-3) / s.params.T) ** relax, 1e-3, 1e3)
            row.append((ds.joint_names[j], s.params.L, s.params.T, t.params.L, t.params.T))
        hist.append(row)
        if verbose:
            print(f"iter {it}: " + "  ".join(f"{n[-6:]}: L {ls*1e3:.0f}->{lt*1e3:.0f} T {ts*1e3:.0f}->{tt*1e3:.0f}"
                                             for n, ls, ts, lt, tt in row), flush=True)
    return SimServo(dead_time=best_dead, kp=kp, kv=best_kv), hist
