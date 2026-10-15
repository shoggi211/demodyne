"""V8: does action/observation alignment matter for a learned policy, and does
demodyne's re-alignment fix it?

Controlled simulation: the ONLY difference between datasets A and B is label timing.
  task     joint-space target tracking on the MuJoCo Menagerie SO-ARM100 (30 fps,
           observe-then-act).  Observation = [x, dx, g, dg] (state, target, and
           their one-frame differences).  Expert: u = g(t + LEAD) + KP*(g(t) - x(t)).
  A        recorded observe-then-act (native alignment).
  B        same physics and expert, but the logged action is s frames stale:
           u_logged[t] = u[t - s]  (the pipeline variation found in the census).
Training sets (single-step MLP behaviour cloning, 5 seeds):
  A only | naive pool A+B | A + B re-aligned by demodyne using *measured*
  alignments | oracle pool A+B with true labels.
Metrics: closed-loop tracking RMSE |x - g| on held-out targets (arm joints, after
the first second) and held-out one-step action error against true labels.
Usage: python experiments/v8_realign.py [n_seeds]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import demodyne as dd
from demodyne.realign import measure_alignment, shift_array
from demodyne.sim import ArmSim, SimServo

RES = Path(__file__).parent / "results"
FPS = 30.0
N_FRAMES = 300            # 10 s episodes
LEAD, KP = 0.12, 0.5      # expert anticipation [s] and feedback gain
ARM = [0, 1, 2, 3, 4]
N_A, N_B, N_TEST = 20, 20, 10
SHIFTS = [0.5, 1.0, 1.5, 2.0, 3.0]
torch.set_num_threads(4)


def targets(sim, rng):
    """Smooth random joint-space target g(t) in degrees, sampled finely for lookahead."""
    lo, hi = np.rad2deg(sim.ctrlrange[:, 0]), np.rad2deg(sim.ctrlrange[:, 1])
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    J = sim.nu
    f = rng.uniform(0.15, 0.7, (3, J))
    ph = rng.uniform(0, 2 * np.pi, (3, J))
    a = rng.uniform(0.3, 1.0, (3, J))
    a = a / a.sum(0) * 0.35 * half
    c = mid + rng.uniform(-0.15, 0.15, J) * half

    def g(t):
        t = np.asarray(t, float)[..., None]
        return c + np.sum(a * np.sin(2 * np.pi * f * t[..., None, :] + ph), axis=-2)
    return g


def obs(x_hist, g, k):
    x = x_hist[k]
    dx = x - x_hist[k - 1] if k > 0 else np.zeros_like(x)
    gk = g(k / FPS)
    dg = gk - g((k - 1) / FPS)
    return np.concatenate([x, dx, gk, dg])


def expert_episode(sim, rng):
    g = targets(sim, rng)

    def pol(k, X):
        t = k / FPS
        return g(t + LEAD) + KP * (g(t) - X[k])
    U, X = sim.rollout_policy(pol, N_FRAMES, FPS, SimServo(), g(0.0))
    O = np.stack([obs(X, g, k) for k in range(N_FRAMES)])
    return {"U": U, "X": X, "O": O, "g": g}


class MLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, d_out))

    def forward(self, x):
        return self.net(x)


def train(O, U, seed, epochs=60):
    torch.manual_seed(seed)
    mo, so = O.mean(0), O.std(0) + 1e-6
    mu, su = U.mean(0), U.std(0) + 1e-6
    Xt = torch.tensor((O - mo) / so, dtype=torch.float32)
    Yt = torch.tensor((U - mu) / su, dtype=torch.float32)
    net = MLP(O.shape[1], U.shape[1])
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()

    def act(o):
        with torch.no_grad():
            y = net(torch.tensor((np.atleast_2d(o) - mo) / so, dtype=torch.float32)).numpy()
        return y * su + mu
    return act


def closed_loop(sim, act, test_eps):
    errs = []
    for ep in test_eps:
        g = ep["g"]
        _, X = sim.rollout_policy(lambda k, Xh: act(obs(Xh, g, k))[0], N_FRAMES, FPS, SimServo(), g(0.0))
        G = g(np.arange(N_FRAMES) / FPS)
        errs.append((X - G)[int(FPS):, ARM])
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))


def to_ds(eps, labels):
    return dd.from_arrays([labels(e) for e in eps], [e["X"] for e in eps], FPS, [f"j{i}" for i in range(6)])


def main(n_seeds=5):
    t0 = time.time()
    sim = ArmSim()
    rng = np.random.default_rng(0)
    A = [expert_episode(sim, rng) for _ in range(N_A)]
    B = [expert_episode(sim, rng) for _ in range(N_B)]
    T = [expert_episode(sim, rng) for _ in range(N_TEST)]
    print(f"generated data in {time.time() - t0:.0f}s", flush=True)
    expert_err = float(np.sqrt(np.mean(np.concatenate(
        [(e["X"] - e["g"](np.arange(N_FRAMES) / FPS))[int(FPS):, ARM] for e in T]) ** 2)))
    O_T = np.concatenate([e["O"] for e in T])
    U_T = np.concatenate([e["U"] for e in T])

    a_A = measure_alignment(to_ds(A, lambda e: e["U"]))
    print(f"expert closed-loop RMSE {expert_err:.2f} deg; measured alignment of A = {a_A:+.2f} frames", flush=True)

    def stack(eps, lab):
        return np.concatenate([e["O"] for e in eps]), np.concatenate([lab(e) for e in eps])

    results = {"expert_rmse": expert_err, "alignment_A": a_A, "shifts": {}}
    cache = {}
    for s in SHIFTS:
        stale = lambda e, s=s: shift_array(e["U"], -s)  # u_logged[t] = u[t - s]
        a_B = measure_alignment(to_ds(B, stale))
        k = a_A - a_B                                    # demodyne's correction, from measurements only
        fixed = lambda e, s=s, k=k: shift_array(shift_array(e["U"], -s), k)
        conds = {
            "A_only": stack(A, lambda e: e["U"]),
            "naive_pool": tuple(np.concatenate(p) for p in zip(stack(A, lambda e: e["U"]), stack(B, stale))),
            "realigned_pool": tuple(np.concatenate(p) for p in zip(stack(A, lambda e: e["U"]), stack(B, fixed))),
            "oracle_pool": tuple(np.concatenate(p) for p in zip(stack(A, lambda e: e["U"]), stack(B, lambda e: e["U"]))),
        }
        row = {"true_shift": s, "measured_alignment_B": a_B, "applied_correction": k, "conditions": {}}
        for name, (O, U) in conds.items():
            key = name if name in ("A_only", "oracle_pool") else (name, s)
            if key in cache:
                row["conditions"][name] = cache[key]
                continue
            cl, ol = [], []
            for seed in range(n_seeds):
                act = train(O, U, seed)
                ol.append(float(np.sqrt(np.mean((act(O_T)[:, ARM] - U_T[:, ARM]) ** 2))))
                cl.append(closed_loop(sim, act, T))
            cache[key] = {"closed_loop_rmse": cl, "action_rmse": ol}
            row["conditions"][name] = cache[key]
        results["shifts"][str(s)] = row
        c = row["conditions"]
        print(f"shift {s:.1f} fr (measured B {a_B:+.2f}, correction {k:+.2f}):  " + "  ".join(
            f"{n}: CL {np.mean(v['closed_loop_rmse']):.2f}±{np.std(v['closed_loop_rmse']):.2f} "
            f"act {np.mean(v['action_rmse']):.2f}" for n, v in c.items()), flush=True)
        json.dump(results, open(RES / "v8_realign.json", "w"), indent=1)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
