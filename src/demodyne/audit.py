"""Physics-level integrity checks for command/response robot datasets.

Metadata linters check that files agree with each other.  These checks ask
whether the recorded signals are physically consistent with a robot following
its commands, which catches bugs that leave every file well-formed:

  timing        irregular / duplicated / missing timestamps
  stale_frames  observation rows repeated while the robot is moving
  frozen        a joint does not respond although it is being commanded
  mapping       command channel i drives state channel j != i (swapped wiring /
                mislabelled names) or drives it with inverted sign
  causality     state responds *before* its command (negative dead time): the
                action column is shifted relative to observations
  action_is_future_state  actions are literally later observations
  latency_outlier  individual episodes whose response lag departs from the rest
  stalls        joints held off-command without moving (contact, overload)
  idle          long motionless heads/tails of episodes
  units         unit convention guess per joint (radians / degrees / normalized)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .ident import Stacked, _grid_init, stall_mask, fit_joint

SEVERITY = {"error": 3, "warning": 2, "info": 1}


@dataclass
class Finding:
    check: str
    severity: str
    message: str
    joint: str | None = None
    episodes: list = field(default_factory=list)
    value: float | None = None

    def to_dict(self):
        d = asdict(self)
        d["episodes"] = [int(e) for e in self.episodes[:50]]
        return d


def _span(ds, j):
    return float(np.ptp(np.concatenate([ep.x[:, j] for ep in ds.episodes])))


# --------------------------------------------------------------------- timing
def check_timing(ds):
    out = []
    h = ds.h
    bad_eps, gaps, dups, nonmono, tot = [], 0, 0, 0, 0
    drift_eps = []
    for ep in ds.episodes:
        dt = np.diff(ep.t)
        tot += len(dt)
        g = int(np.sum(dt > 1.5 * h))
        d = int(np.sum(dt < 0.25 * h))
        nm = int(np.sum(dt < 0))
        gaps += g
        dups += d
        nonmono += nm
        if g or d or nm:
            bad_eps.append(ep.index)
        # timestamps vs nominal clock: cumulative drift
        if len(ep.t) > 2:
            drift = (ep.t[-1] - ep.t[0]) - (len(ep.t) - 1) * h
            if abs(drift) > 2 * h:
                drift_eps.append(ep.index)
    short = [ep.index for ep in ds.episodes if len(ep.t) < 10]
    if short:
        out.append(Finding("timing", "warning", f"{len(short)} degenerate episodes shorter than 10 frames",
                           episodes=short))
    if nonmono:
        out.append(Finding("timing", "error", f"{nonmono} non-monotonic timestamp steps", episodes=bad_eps))
    if gaps:
        out.append(Finding("timing", "warning", f"{gaps} gaps > 1.5 frame periods ({gaps / max(tot,1):.2%} of steps)",
                           episodes=bad_eps, value=gaps / max(tot, 1)))
    if dups:
        out.append(Finding("timing", "warning", f"{dups} near-duplicate timestamps", episodes=bad_eps))
    if drift_eps:
        out.append(Finding("timing", "warning",
                           f"{len(drift_eps)} episodes whose duration disagrees with frame count at {ds.fps:g} fps",
                           episodes=drift_eps))
    return out


# --------------------------------------------------------------- stale frames
def check_stale_frames(ds, thresh=0.01, min_eligible=50):
    """Observation vector exactly repeated in the middle of fast motion.

    A physical arm moving at >1% of its range per frame before and after a step
    does not stop dead for exactly one frame; a stale sensor read does.  Slow
    motion (where every joint can legitimately sit inside its deadband for a
    frame) is excluded."""
    spans = np.array([max(_span(ds, j), 1e-12) for j in range(ds.n_joints)])
    # encoder quantum per joint: small quantile of the non-zero step sizes
    q = []
    for j in range(ds.n_joints):
        d = np.abs(np.concatenate([np.diff(ep.x[:, j]) for ep in ds.episodes if len(ep) > 1]))
        d = d[d > 0]
        q.append(np.quantile(d, 0.05) if len(d) else 0.0)
    thr = np.maximum(0.01 * spans, 3 * np.array(q))
    rep, eligible, eps = 0, 0, []
    for ep in ds.episodes:
        if len(ep) < 4:
            continue
        dx = np.diff(ep.x, axis=0)
        fast = np.abs(dx) > thr
        elig = np.zeros(len(dx), bool)
        elig[1:-1] = np.any(fast[:-2] & fast[2:], axis=1)
        stale = elig & np.all(dx == 0, axis=1)
        rep += int(stale.sum())
        eligible += int(elig.sum())
        if stale.sum() >= 2:
            eps.append(ep.index)
    frac = rep / max(eligible, 1)
    if eligible >= min_eligible and frac > thresh:
        return [Finding("stale_frames", "warning",
                        f"{frac:.1%} of mid-motion steps repeat the previous observation exactly "
                        f"(stale sensor reads / duplicated frames)", episodes=eps, value=frac)]
    return []


# --------------------------------------------------------------------- frozen
def check_frozen(ds):
    """Joint does not respond to a substantial command (or moves without one).
    Joints the dataset barely uses are skipped: a small command inside the servo
    deadband legitimately produces no motion."""
    out = []
    span_u_all = np.array([np.ptp(np.concatenate([ep.u[:, j] for ep in ds.episodes])) for j in range(ds.n_joints)])
    ref = np.median(span_u_all) + 1e-12
    for j, name in enumerate(ds.joint_names):
        span_u = span_u_all[j]
        if span_u < 0.1 * ref:
            continue
        min_cmd = max(0.25 * span_u, 0.1 * ref)
        frozen, dead_cmd = [], []
        for ep in ds.episodes:
            pu, px = np.ptp(ep.u[:, j]), np.ptp(ep.x[:, j])
            if pu > min_cmd and px < 0.01 * pu:
                frozen.append(ep.index)
            if px > min_cmd and pu < 0.01 * px:
                dead_cmd.append(ep.index)
        if frozen:
            out.append(Finding("frozen", "error", f"follower does not move while commanded in {len(frozen)} episodes "
                               f"(torque off / disconnected servo)", name, frozen, len(frozen) / len(ds.episodes)))
        if dead_cmd:
            out.append(Finding("frozen", "warning", f"state moves while command is constant in {len(dead_cmd)} episodes "
                               f"(dead leader channel or robot moved externally)", name, dead_cmd,
                               len(dead_cmd) / len(ds.episodes)))
    return out


# -------------------------------------------------------------------- mapping
def mapping_matrix(ds, lags=4):
    """E[i, j]: total standardised effect of command increments on channel i on
    state increments of channel j, from a joint multivariate regression over all
    commands at lags 0..lags-1.  Unlike pairwise correlation this separates a
    joint's own command from other commands the operator moves together with it
    (e.g. shoulder and elbow while reaching)."""
    J = ds.n_joints
    rows_u, rows_x = [], []
    for ep in ds.episodes:
        if len(ep) <= lags + 2:
            continue
        du = np.diff(ep.u, axis=0)
        dx = np.diff(ep.x, axis=0)
        n = len(du)
        rows_u.append(np.hstack([du[lags - 1 - l: n - l] for l in range(lags)]))  # (n-lags+1, J*lags)
        rows_x.append(dx[lags - 1:])
    if not rows_u:
        return np.zeros((J, J))
    A = np.concatenate(rows_u)
    Y = np.concatenate(rows_x)
    sa = A.std(0) + 1e-12
    sy = Y.std(0) + 1e-12
    An = (A - A.mean(0)) / sa
    Yn = (Y - Y.mean(0)) / sy
    lam = 1e-3 * len(An)
    coef = np.linalg.solve(An.T @ An + lam * np.eye(An.shape[1]), An.T @ Yn)  # (J*lags, J)
    # sum lag coefficients per command channel; regressor (l, i) sits at column l*J + i
    return coef.reshape(lags, J, J).sum(0)


_POSE = {"x", "y", "z", "qx", "qy", "qz", "qw"}


def _is_pose(name):
    tok = name.lower().replace(".pos", "").replace(".", "_").split("_")[-1]
    return tok in _POSE


def check_mapping(ds):
    out = []
    E = mapping_matrix(ds)
    # per-episode centred positions: between-episode offsets must not dilute the
    # position-level correlation used to confirm a verdict
    U = np.concatenate([ep.u - np.median(ep.u, axis=0) for ep in ds.episodes])
    X = np.concatenate([ep.x - np.median(ep.x, axis=0) for ep in ds.episodes])
    for j, name in enumerate(ds.joint_names):
        if _is_pose(name):
            continue  # Cartesian channels are coupled by kinematics; out of scope
        col = E[:, j]
        i = int(np.argmax(np.abs(col)))
        if abs(col[i]) < 0.3:
            continue  # nothing drives this joint; the frozen check covers it
        own_pos = np.corrcoef(U[:, j], X[:, j])[0, 1] if np.std(U[:, j]) > 0 and np.std(X[:, j]) > 0 else 0.0
        if i != j and abs(col[i]) > 2.0 * abs(col[j]) and abs(own_pos) < 0.5:
            out.append(Finding("mapping", "error",
                               f"state '{name}' is driven by command '{ds.joint_names[i]}' (effect {col[i]:+.2f}) "
                               f"rather than its own command ({col[j]:+.2f}): swapped or mislabelled channels",
                               name, value=float(col[i])))
        elif col[j] < 0 and own_pos < -0.5:
            out.append(Finding("mapping", "error",
                               f"state moves opposite to its own command (effect {col[j]:+.2f}): inverted sign / "
                               f"calibration direction between leader and follower", name, value=float(col[j])))
    return out


def check_range(ds, lo=0.7, hi=1.4):
    """Follower range vs commanded range (1st-99th percentile).  A ratio far from
    1 means leader and follower are calibrated to different ranges, or the
    follower is clipped at a joint/software limit the leader does not have."""
    out = []
    U = np.concatenate([ep.u for ep in ds.episodes])
    X = np.concatenate([ep.x for ep in ds.episodes])
    su = np.quantile(U, 0.99, axis=0) - np.quantile(U, 0.01, axis=0)
    sx = np.quantile(X, 0.99, axis=0) - np.quantile(X, 0.01, axis=0)
    ref = np.median(su) + 1e-12
    for j, name in enumerate(ds.joint_names):
        if su[j] < 0.1 * ref or _is_pose(name) or any(k in name.lower() for k in ("grip", "jaw", "hand")):
            continue
        r = sx[j] / su[j]
        if r < lo or r > hi:
            out.append(Finding("range_mismatch", "warning",
                               f"follower covers {r:.0%} of the commanded range ({sx[j]:.1f} vs {su[j]:.1f}): "
                               f"leader/follower calibration mismatch or follower clipped at a limit", name, value=float(r)))
    return out


# ------------------------------------------------------------------ causality
def alignment_offset(fits):
    """Median dead time across reliably identified non-gripper joints [s]."""
    Ls = [f.params.L for f in fits if np.isfinite(f.params.L) and f.r2_one_step >= 0.5
          and not any(k in f.name.lower() for k in ("grip", "jaw"))]
    return float(np.median(Ls)) if len(Ls) >= 2 else float("nan")


def check_causality(ds, fits=None):
    """Action/observation alignment from the physics.

    With an observe-then-act loop (current LeRobot `record`), a servo's response
    to u[t] first appears in x[t+1] and the apparent dead time is positive
    (census: 5-24 ms interquartile for v3 '.pos' datasets).  A write-then-read
    loop shifts it by up to one frame earlier; a shifted action column by whole
    frames.  So: median dead time < 0  -> alignment convention differs from
    observe-then-act (warning: pooling with other datasets mixes label timing);
    < -1.5 frames -> beyond any loop-ordering convention (error: action column
    shifted / mis-joined).  Note timestamps in LeRobot are frame_index/fps, so
    "ms" here means dataset time."""
    out = []
    h = ds.h
    if fits is not None:
        med = alignment_offset(fits)
        if np.isfinite(med) and med < 0:
            frames = -med / h
            if med < -1.5 * h:
                out.append(Finding("causality", "error",
                                   f"states respond {-med*1e3:.0f} ms ({frames:.1f} frames) before their commands "
                                   f"(median over joints): beyond any record-loop ordering; action column is likely "
                                   f"shifted relative to observations", value=med))
            else:
                out.append(Finding("causality", "warning",
                                   f"states respond {-med*1e3:.0f} ms ({frames:.1f} frames) before their commands: "
                                   f"actions are aligned differently from the observe-then-act convention "
                                   f"(write-then-read loop or 1-frame shift); pooling with other datasets mixes "
                                   f"action-label timing", value=med))
        return out  # dataset-level verdict supersedes per-joint ones
    for j, name in enumerate(ds.joint_names):
        if fits is not None:
            f = fits[j]
            L, lag = f.params.L, f.lag
            if not np.isfinite(L) or f.r2_one_step < 0.3:
                continue
        else:
            init = _grid_init(Stacked(ds.episodes, j), h)
            if init is None:
                continue
            L = init[0] * h
            lag = L
        if L < -0.5 * h:
            out.append(Finding("causality", "error",
                               f"state responds {-L*1e3:.0f} ms *before* its command: action column is "
                               f"time-shifted relative to observations", name, value=float(L)))
        elif L > 0.3:
            out.append(Finding("causality", "warning", f"very long dead time {L*1e3:.0f} ms", name, value=float(L)))
    return out


def check_action_is_future_state(ds):
    """Datasets that build 'actions' from later observations (not commands)."""
    best = None
    for k in range(0, 4):
        errs = []
        for ep in ds.episodes:
            if len(ep) <= k + 1:
                continue
            errs.append(np.abs(ep.u[: len(ep) - k] - ep.x[k:]).max(axis=1))
        if not errs:
            continue
        frac_exact = float(np.mean(np.concatenate(errs) < 1e-6))
        if best is None or frac_exact > best[1]:
            best = (k, frac_exact)
    if best and best[1] > 0.9:
        return [Finding("action_is_future_state", "warning",
                        f"action[t] equals observation.state[t+{best[0]}] in {best[1]:.0%} of frames: actions are "
                        f"derived from logged states, not recorded commands; command dynamics are unobservable",
                        value=best[1])]
    return []


# ----------------------------------------------------------- latency outliers
def episode_lags(ds, j, p):
    """Per-episode effective lag for joint j.

    Only (L, T) are re-estimated; offset, deadband and rate limit are held at the
    dataset-level fit `p`, which keeps single-episode estimates well posed.
    """
    from scipy.optimize import least_squares
    from .model import ServoParams

    h = ds.h
    lags = np.full(len(ds.episodes), np.nan)
    for k, ep in enumerate(ds.episodes):
        S = Stacked([ep], j)
        if len(S.k) < 30 or np.std(S.dx) == 0 or np.mean(np.abs(S.dx) > 1e-3 * S.span) < 0.2:
            continue
        sc = np.std(S.dx)

        def resid(th):
            q = ServoParams(L=th[0], T=float(np.exp(th[1])), c=p.c, db=p.db, vmax=p.vmax)
            return (S.dx - S.predict_dx(q, h)) / sc

        try:
            sol = least_squares(resid, [p.L, np.log(p.T)], bounds=([-3 * h, np.log(2e-3)], [8 * h, np.log(2.0)]),
                                loss="soft_l1", max_nfev=60)
            lags[k] = sol.x[0] + np.exp(sol.x[1])
        except Exception:
            pass
    return lags


def check_latency_outliers(ds, fits, z=6.0, min_dev_ms=15.0):
    out = []
    if len(ds.episodes) < 8 or fits is None:
        return out
    per = []
    for j, f in enumerate(fits):
        if (not np.isfinite(f.params.L) or f.r2_one_step < 0.8 or f.stall_fraction > 0.1
                or "grip" in f.name.lower() or "jaw" in f.name.lower()):
            continue
        per.append(episode_lags(ds, j, f.params))
    if not per:
        return out
    # combine joints: median lag per episode across joints is robust to one bad joint
    L = np.nanmedian(np.vstack(per), axis=0)
    ok = np.isfinite(L)
    if ok.sum() < 8:
        return out
    med = np.median(L[ok])
    mad = 1.4826 * np.median(np.abs(L[ok] - med)) + 1e-4
    dev = np.abs(L - med)
    bad = [ds.episodes[k].index for k in np.flatnonzero(ok & (dev > z * mad) & (dev > min_dev_ms / 1e3))]
    if bad:
        worst = float(np.nanmax(np.abs(L[ok] - med)))
        out.append(Finding("latency_outlier", "warning",
                           f"{len(bad)} episodes with response lag far from the dataset median {med*1e3:.0f} ms "
                           f"(max deviation {worst*1e3:.0f} ms): different setup, USB/bus load or clock",
                           episodes=bad, value=worst))
    return out


# --------------------------------------------------------------------- stalls
def check_stalls(ds, fits=None):
    """Joints held away from their command without moving.  For arm joints the
    identified steady offset (gravity sag, calibration mismatch) is removed first
    and the episode must stall for >= 0.25 s, so a loaded joint resting with a
    constant error is not mistaken for contact."""
    out = []
    for j, name in enumerate(ds.joint_names):
        span = _span(ds, j)
        if span == 0:
            continue
        is_grip = any(k in name.lower() for k in ("grip", "jaw", "hand"))
        c = 0.0
        if fits is not None and not is_grip and np.isfinite(fits[j].params.c):
            c = fits[j].params.c
        eps, tot, n = [], 0, 0
        min_frames = 4 if is_grip else max(4, int(round(0.25 * ds.fps)))
        for ep in ds.episodes:
            s = stall_mask(ep.u[:, j] - c, ep.x[:, j], span, min_frames=min_frames,
                           err_frac=0.04 if is_grip else 0.08)
            tot += s.sum()
            n += len(s)
            if s.any():
                eps.append(ep.index)
        frac = tot / max(n, 1)
        if is_grip and frac > 0.02:
            out.append(Finding("stalls", "info", f"gripper held off-command in {frac:.0%} of frames "
                               f"({len(eps)} episodes): grasp contact; excluded from dynamics fit", name, eps, frac))
        elif not is_grip and frac > 0.05:
            out.append(Finding("stalls", "warning", f"joint held >8% of its range off-command without moving in "
                               f"{frac:.0%} of frames: contact with environment, overload or torque limit",
                               name, eps, frac))
    return out


# ----------------------------------------------------------------------- idle
def check_idle(ds, thresh=0.3):
    fr, eps = [], []
    for ep in ds.episodes:
        span = np.ptp(ep.x, axis=0) + 1e-12
        mv = np.any(np.abs(np.diff(ep.x, axis=0)) > 0.01 * span, axis=1)
        if not mv.any():
            fr.append(1.0)
            eps.append(ep.index)
            continue
        first, last = np.argmax(mv), len(mv) - np.argmax(mv[::-1])
        f = (first + (len(mv) - last)) / len(mv)
        fr.append(f)
        if f > thresh:
            eps.append(ep.index)
    if eps:
        return [Finding("idle", "info", f"{len(eps)} episodes are >{thresh:.0%} motionless at head/tail "
                        f"(median idle {np.median(fr):.0%})", episodes=eps, value=float(np.median(fr)))]
    return []


# ---------------------------------------------------------------------- units
def guess_units(values: np.ndarray) -> str:
    lo, hi = np.quantile(values, [0.001, 0.999])
    span = hi - lo
    if span == 0:
        return "constant"
    if max(abs(lo), abs(hi)) <= 3.3 and span <= 6.3:
        return "radians?" if span > 0.02 else "unknown"
    if lo >= -0.5 and hi <= 100.5 and lo < 15:
        return "0..100 normalized or degrees"
    if lo >= -100.5 and hi <= 100.5:
        return "-100..100 normalized or degrees"
    if max(abs(lo), abs(hi)) <= 360:
        return "degrees"
    return "raw ticks?"


def check_units(ds):
    units = {n: guess_units(np.concatenate([ep.x[:, j] for ep in ds.episodes])) for j, n in enumerate(ds.joint_names)}
    out = [Finding("units", "info", "unit guess per joint: " + ", ".join(f"{k}={v}" for k, v in units.items()))]
    if any(v == "raw ticks?" for v in units.values()):
        out[0].severity = "warning"
    return out


# ---------------------------------------------------------------------- run
SIGNAL_CHECKS = [check_timing, check_stale_frames, check_frozen, check_mapping, check_range,
                 check_action_is_future_state, check_idle, check_units]
DYNAMICS_CHECKS = [check_causality, check_latency_outliers, check_stalls]


def audit(ds, fits=None, dynamics=True):
    """Run all checks. `fits` (from ident.fit_dataset) is computed if needed."""
    findings = []
    for c in SIGNAL_CHECKS:
        try:
            findings += c(ds)
        except Exception as e:  # a broken check must not hide the others
            findings.append(Finding(c.__name__.replace("check_", ""), "info", f"check failed: {e!r}"))
    if dynamics:
        if fits is None:
            from .ident import fit_dataset
            fits = fit_dataset(ds)
        for c in DYNAMICS_CHECKS:
            try:
                findings += c(ds, fits)
            except Exception as e:
                findings.append(Finding(c.__name__.replace("check_", ""), "info", f"check failed: {e!r}"))
    findings.sort(key=lambda f: -SEVERITY[f.severity])
    return findings
