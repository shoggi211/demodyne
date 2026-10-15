import json

import numpy as np
import pandas as pd
import pytest

import demodyne as dd
from demodyne import corrupt as C
from demodyne.audit import audit
from demodyne.model import ServoParams, simulate

FPS = 30.0


def teleop_like(n, rng, amp=30.0):
    """Smooth, bursty command: low-passed noise with pauses, like a human leader arm."""
    w = rng.normal(0, 1, n)
    w[rng.random(n) < 0.3] = 0
    v = np.convolve(w, np.ones(9) / 9, mode="same")
    u = np.cumsum(v)
    u = u - u.mean()
    return amp * u / (np.abs(u).max() + 1e-9)


def make_ds(params, n_eps=12, n=300, J=4, seed=0, noise=0.02, quant=0.088):
    rng = np.random.default_rng(seed)
    U, X = [], []
    for _ in range(n_eps):
        u = np.stack([teleop_like(n, rng) + 10 * j for j in range(J)], 1)
        x = np.stack([simulate(params[j], u[:, j], u[0, j], 1 / FPS) for j in range(J)], 1)
        x = np.round((x + rng.normal(0, noise, x.shape)) / quant) * quant
        U.append(u)
        X.append(x)
    return dd.from_arrays(U, X, FPS, [f"joint{j}" for j in range(J)])


PARAMS = [ServoParams(L=0.020, T=0.060, c=0.3, db=0.2, vmax=200),
          ServoParams(L=0.035, T=0.090, c=-0.5, db=0.1, vmax=150),
          ServoParams(L=0.010, T=0.050, c=0.0, db=0.4, vmax=250),
          ServoParams(L=0.050, T=0.120, c=1.0, db=0.0, vmax=300)]


@pytest.fixture(scope="module")
def ds():
    return make_ds(PARAMS)


def test_parameter_recovery(ds):
    fits = dd.fit_dataset(ds, bootstrap=10)
    for f, p in zip(fits, PARAMS):
        assert abs(f.params.L - p.L) < 4e-3, (f.name, f.params)
        assert abs(f.params.T - p.T) / p.T < 0.1, (f.name, f.params)
        assert abs(f.lag - (p.L + p.T)) < 4e-3
        assert abs(f.params.c - p.c) < 0.3
        assert f.r2_one_step > 0.95
        lo, hi = f.ci["lag"]
        assert lo <= hi
        assert f.rmse_free_run < f.baseline_rmse


def test_delay_is_unit_invariant(ds):
    """Rescaling a joint (deg -> rad) must not change timing estimates."""
    scaled = C._clone(ds)
    for e in scaled.episodes:
        e.u[:, 0] = np.deg2rad(e.u[:, 0])
        e.x[:, 0] = np.deg2rad(e.x[:, 0])
    a = dd.fit_joint(ds.episodes, 0, h=ds.h)
    b = dd.fit_joint(scaled.episodes, 0, h=ds.h)
    assert abs(a.lag - b.lag) < 2e-3
    assert abs(a.params.L - b.params.L) < 3e-3


def test_clean_has_no_alarms(ds):
    bad = [f for f in audit(ds) if f.severity in ("error", "warning")]
    assert not bad, [f.message for f in bad]


@pytest.mark.parametrize("fault,check", [
    (lambda d: C.shift_actions(d, -2), "causality"),
    (lambda d: C.swap_channels(d, 0, 2), "mapping"),
    (lambda d: C.invert_channel(d, 1), "mapping"),
    (lambda d: C.freeze_joint(d, 3, frac=0.2), "frozen"),
    (lambda d: C.stale_frames(d, 0.05), "stale_frames"),
    (lambda d: C.drop_frames(d, 0.03), "timing"),
    (lambda d: C.episode_latency(d, 0.1, n_eps=1), "latency_outlier"),
    (lambda d: C.future_state_actions(d, 1), "action_is_future_state"),
])
def test_fault_detected(ds, fault, check):
    bad, truth = fault(ds)
    found = {f.check for f in audit(bad) if f.severity in ("error", "warning")}
    assert check in found, found


def test_shift_moves_delay_by_one_frame(ds):
    a = dd.fit_joint(ds.episodes, 0, h=ds.h)
    b, _ = C.shift_actions(ds, 1)
    fb = dd.fit_joint(b.episodes, 0, h=ds.h)
    assert abs((fb.params.L - a.params.L) - 1 / FPS) < 3e-3


def test_stall_excluded_from_fit():
    """Gripper-like joint that stalls on an object must still be identified."""
    rng = np.random.default_rng(1)
    p = ServoParams(L=0.02, T=0.06)
    U, X = [], []
    for _ in range(10):
        u = teleop_like(300, rng, amp=40) + 50
        x = simulate(p, u, u[0], 1 / FPS)
        # object blocks closing below 40
        x = np.maximum(x, 40.0)
        U.append(u[:, None])
        X.append(x[:, None])
    d = dd.from_arrays(U, X, FPS, ["gripper"])
    f = dd.fit_joint(d.episodes, 0, h=d.h)
    assert f.stall_fraction > 0.05
    assert abs(f.lag - 0.08) < 0.01


def test_load_lerobot_v3_layout(tmp_path, ds):
    root = tmp_path / "toy"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    names = ["main_shoulder_pan", "main_gripper"]
    info = {"codebase_version": "v3.0", "robot_type": "so100", "fps": 30,
            "features": {"action": {"dtype": "float32", "shape": [2], "names": names},
                         "observation.state": {"dtype": "float32", "shape": [2], "names": names}}}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    rows = []
    for k, e in enumerate(ds.episodes[:3]):
        for i in range(len(e)):
            rows.append({"timestamp": e.t[i], "episode_index": k, "frame_index": i,
                         "action": e.u[i, :2].astype(np.float32), "observation.state": e.x[i, :2].astype(np.float32)})
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    d = dd.load(str(root))
    assert d.n_joints == 2 and len(d.episodes) == 3 and d.fps == 30
    assert d.robot_type == "so100"


def test_pool_check_flags_alignment_and_lag(ds):
    from demodyne.pool import compare, profile
    shifted, _ = C.shift_actions(ds, -2)
    slow = make_ds([ServoParams(L=p.L, T=p.T * 3, c=p.c, db=p.db, vmax=p.vmax) for p in PARAMS], seed=3)
    same = compare([profile(ds), profile(make_ds(PARAMS, seed=5))])
    assert not [m for s, m in same if s == "error"], same
    msgs = " ".join(m for _, m in compare([profile(ds), profile(shifted)]))
    assert "alignment differs" in msgs
    msgs = " ".join(m for _, m in compare([profile(ds), profile(slow)]))
    assert "response lag differs" in msgs


def test_no_false_alarms_for_slow_motion_and_unused_joint():
    """Regressions from real census data: slow motion inside the servo deadband
    must not read as stale frames; command jitter on a barely-used joint must not
    read as a frozen servo."""
    rng = np.random.default_rng(7)
    params = PARAMS[:3] + [ServoParams(L=0.02, T=0.06, db=1.0)]
    U, X = [], []
    for _ in range(12):
        u = np.stack([teleop_like(300, rng, amp=3.0) for _ in range(3)]
                     + [0.008 * rng.standard_normal(300).cumsum() + 2.0], 1)  # joint 3: tiny jitter (~2% of others)
        x = np.stack([simulate(params[j], u[:, j], u[0, j], 1 / FPS) for j in range(4)], 1)
        x = np.round(x / 0.088) * 0.088
        U.append(u)
        X.append(x)
    d = dd.from_arrays(U, X, FPS)
    found = {(f.check, f.joint) for f in audit(d) if f.severity in ("error", "warning")}
    assert not any(c in ("stale_frames", "frozen") for c, _ in found), found


def write_lerobot(root, ds, layout="v3", n_eps=None):
    """Write `ds` as a minimal LeRobot dataset (v3 chunked or v2 per-episode)."""
    names = list(ds.joint_names)
    J = len(names)
    info = {"codebase_version": "v3.0" if layout == "v3" else "v2.1", "robot_type": "so100", "fps": int(ds.fps),
            "features": {"action": {"dtype": "float32", "shape": [J], "names": names},
                         "observation.state": {"dtype": "float32", "shape": [J], "names": names}}}
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "stats.json").write_text(json.dumps({"action": {"min": [0] * J, "max": [0] * J,
                                                                     "mean": [0] * J, "std": [0] * J}}))
    (root / "data" / "chunk-000").mkdir(parents=True)
    frames = []
    for k, e in enumerate(ds.episodes[:n_eps]):
        rows = [{"timestamp": e.t[i], "episode_index": k, "frame_index": i, "index": len(e) * k + i,
                 "action": e.u[i].astype(np.float32), "observation.state": e.x[i].astype(np.float32)}
                for i in range(len(e))]
        if layout == "v2":
            pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / f"episode_{k:06d}.parquet")
        frames += rows
    if layout == "v3":
        pd.DataFrame(frames).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")


def test_realign_in_memory_recovers_alignment(ds):
    from demodyne.realign import measure_alignment, shift_dataset
    a0 = measure_alignment(ds)
    bad = shift_dataset(ds, -1.7)
    a1 = measure_alignment(bad)
    assert abs((a1 - a0) + 1.7) < 0.15
    fixed = shift_dataset(bad, a0 - a1)
    assert abs(measure_alignment(fixed) - a0) < 0.1


@pytest.mark.parametrize("layout", ["v3", "v2"])
def test_realign_files_round_trip(tmp_path, ds, layout):
    from demodyne.realign import measure_alignment, realign, shift_dataset
    src = tmp_path / "src"
    write_lerobot(src, shift_dataset(ds, -2.0), layout)
    out = tmp_path / "out"
    rec = realign(str(src), str(out), target_frames=0.6)
    assert abs(rec["shift_frames"] - (0.6 - rec["measured_alignment_frames"])) < 1e-9
    d2 = dd.load(str(out))
    assert abs(measure_alignment(d2) - 0.6) < 0.15
    # observations, timestamps and episode structure untouched
    d1 = dd.load(str(src))
    assert len(d1.episodes) == len(d2.episodes)
    for e1, e2 in zip(d1.episodes, d2.episodes):
        assert np.array_equal(e1.x, e2.x) and np.array_equal(e1.t, e2.t)
    assert (out / "meta" / "demodyne_realign.json").exists()
    stats = json.loads((out / "meta" / "stats.json").read_text())
    assert stats["action"]["max"][0] != 0


def test_realign_explicit_shift_is_exact(tmp_path, ds):
    from demodyne.realign import realign
    src = tmp_path / "src"
    write_lerobot(src, ds, "v3", n_eps=2)
    realign(str(src), str(tmp_path / "out"), shift_frames=1.0)
    a, b = dd.load(str(src)), dd.load(str(tmp_path / "out"))
    e1, e2 = a.episodes[0], b.episodes[0]
    assert np.allclose(e2.u[:-1], e1.u[1:], atol=1e-5)


def test_closed_loop_rollout_order_and_tracking():
    """observe-then-act: u[k] is computed from x[0..k]; a constant target is reached."""
    pytest.importorskip("mujoco")
    from demodyne.sim import DEFAULT_XML, ArmSim, SimServo
    if not DEFAULT_XML.exists():
        pytest.skip("Menagerie SO-ARM100 asset not present")
    sim = ArmSim()
    q0 = np.rad2deg(sim.ctrlrange.mean(1))
    goal = q0 + 10.0
    seen = []

    def pol(k, X):
        seen.append(len(X))
        return goal
    U, X = sim.rollout_policy(pol, 60, 30.0, SimServo(), q0)
    assert seen == list(range(1, 61))          # policy at step k sees k+1 observations
    assert np.allclose(X[0], q0, atol=0.1)     # first observation precedes any command
    assert np.abs(X[-1] - goal)[[0, 2, 3, 4]].max() < 1.5


def test_plain_summary_verdicts(ds):
    from demodyne.plain import summary
    fits = dd.fit_dataset(ds)
    assert "healthy" in summary(ds, fits, audit(ds, fits))
    bad, _ = C.swap_channels(ds, 0, 2)
    fb = dd.fit_dataset(bad)
    assert "Problems found" in summary(bad, fb, audit(bad, fb))


def test_realign_subset_of_episodes(tmp_path, ds):
    from demodyne.realign import measure_alignment, realign
    src = tmp_path / "src"
    write_lerobot(src, ds, "v3")
    mixed = tmp_path / "mixed"
    late = list(range(6, 12))
    realign(str(src), str(mixed), shift_frames=-2.0, episodes=late)       # simulate a second pipeline
    m = dd.load(str(mixed))
    a_first = measure_alignment(m.subset(range(0, 6)))
    a_late = measure_alignment(m.subset(range(6, 12)))
    assert abs((a_first - a_late) - 2.0) < 0.2
    fixed = tmp_path / "fixed"
    realign(str(mixed), str(fixed), target_frames=a_first, episodes=late)  # repair only that part
    f = dd.load(str(fixed))
    assert abs(measure_alignment(f.subset(range(6, 12))) - a_first) < 0.15
    for e1, e2 in zip(m.episodes[:6], f.episodes[:6]):
        assert np.array_equal(e1.u, e2.u)                                  # untouched episodes
