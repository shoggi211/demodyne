"""V5: does fingerprint calibration make MuJoCo track the real arm better?

Calibrate the Menagerie SO-ARM100 actuators on half of a real dataset's episodes,
then replay held-out real leader commands and compare simulated follower joint
trajectories against the real follower.  Reported per arm joint (gripper is
excluded: grasp contact is not simulated):
  rmse      raw RMSE [deg]
  rmse_dyn  RMSE after removing each episode's median error (isolates dynamics
            from static gravity-sag / calibration-zero differences)
Usage: python experiments/v5_sim_calibration.py <dataset> [n_train] [n_test]
"""
import json
import sys
from pathlib import Path

import numpy as np

import demodyne as dd
from demodyne.calibrate import calibrate, simulate_dataset
from demodyne.sim import ArmSim, CommandMap, SimServo

OUT = Path(__file__).parent / "results"


def score(real_eps, sim_ds, joints):
    raw, dyn = [], []
    for re, se in zip(real_eps, sim_ds.episodes):
        e = re.x[:, joints] - se.x[:, joints]
        raw.append(e)
        dyn.append(e - np.median(e, axis=0))
    raw, dyn = np.concatenate(raw), np.concatenate(dyn)
    return np.sqrt((raw**2).mean(0)), np.sqrt((dyn**2).mean(0))


def main(src, n_train=15, n_test=15):
    ds = dd.load(src, max_episodes=n_train + n_test)
    if ds.n_joints != 6:
        raise SystemExit("expects a 6-channel SO-100/101 dataset")
    eps = ds.episodes
    train, test = eps[:n_train], eps[n_train:n_train + n_test]
    arm = [0, 1, 2, 3, 4]
    sim = ArmSim()
    cmap = CommandMap(np.concatenate([e.u for e in train]), sim.ctrlrange)

    servo_cal, hist = calibrate(sim, ds, episodes=train, iters=4, verbose=True)
    res = {}
    for name, servo in [("menagerie_default", SimServo()), ("demodyne_calibrated", servo_cal)]:
        sd = simulate_dataset(sim, ds, servo, cmap, test)
        r, d = score(test, sd, arm)
        res[name] = {"rmse": r.tolist(), "rmse_dyn": d.tolist()}
    # non-physics references
    naive = dd.from_arrays([e.u for e in test], [e.u for e in test], ds.fps)
    r, d = score(test, naive, arm)
    res["naive_state_equals_command"] = {"rmse": r.tolist(), "rmse_dyn": d.tolist()}
    fits = dd.fit_dataset(type(ds)(ds.repo_id, ds.fps, ds.joint_names, train))
    fop = dd.from_arrays([e.u for e in test],
                         [np.stack([dd.simulate(fits[j].params, e.u[:, j], e.x[0, j], ds.h) for j in range(6)], 1)
                          for e in test], ds.fps)
    r, d = score(test, fop, arm)
    res["demodyne_fopdt_model"] = {"rmse": r.tolist(), "rmse_dyn": d.tolist()}

    names = [ds.joint_names[j] for j in arm]
    print(f"\nheld-out episodes: {len(test)}   joints: {names}")
    for k, v in res.items():
        print(f"  {k:28s} mean rmse={np.mean(v['rmse']):.2f}  mean rmse_dyn={np.mean(v['rmse_dyn']):.2f}   "
              f"per-joint dyn={np.round(v['rmse_dyn'], 2).tolist()}")
    out = {"dataset": str(src), "joints": names, "results": res,
           "calibrated": {"kv": np.asarray(servo_cal.kv).tolist(), "kp": np.asarray(servo_cal.kp).tolist(),
                          "dead_time": np.asarray(servo_cal.dead_time).tolist()},
           "history": [[list(map(lambda v: v if isinstance(v, str) else float(v), r)) for r in h] for h in hist]}
    tag = Path(str(src)).name.replace("/", "__")
    json.dump(out, open(OUT / f"v5_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(int(v) for v in a[1:]))
