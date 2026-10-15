"""MuJoCo ground-truth generator for validating identification.

Drives a position-servoed arm model (default: MuJoCo Menagerie SO-ARM100) with
*recorded* leader trajectories, so excitation statistics match real teleop, while
the servo gain, damping, torque limit, command dead time and encoder resolution
are known.  Sampling follows the LeRobot record loop: observe x[k], then send
u[k], which is held until the next frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_XML = Path(__file__).resolve().parents[2] / "assets" / "menagerie" / "trs_so_arm100" / "so_arm100.xml"


@dataclass
class SimServo:
    """Scalars apply to all joints; arrays (len = n actuators) set per-joint values."""
    dead_time: float | np.ndarray = 0.0     # extra command latency [s] (e.g. bus + driver)
    kp: float | np.ndarray | None = None    # position gain override [N m / rad]
    kv: float | np.ndarray | None = None    # damping override [N m s / rad]
    force_limit: float | np.ndarray | None = None  # torque limit override [N m]
    ticks_per_rev: int = 4096  # encoder quantisation (STS3215 = 4096)
    noise_deg: float = 0.0     # additive measurement noise std [deg]


class CommandMap:
    """Affine map between recorded joint units (deg, arbitrary zero) and model
    radians: centre each joint in its control range, preserve amplitude unless it
    would exceed `fill` of the half range."""

    def __init__(self, u_deg: np.ndarray, ctrlrange: np.ndarray, fill=0.8):
        self.med = np.median(u_deg, axis=0)
        rad = np.deg2rad(u_deg - self.med)
        self.mid = ctrlrange.mean(axis=1)
        half = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2
        amp = np.max(np.abs(rad), axis=0) + 1e-9
        self.s = np.minimum(1.0, fill * half / amp)

    def to_model(self, deg):
        return self.mid + np.deg2rad(deg - self.med) * self.s

    def to_data(self, rad):
        return self.med + np.rad2deg((rad - self.mid) / self.s)


def map_commands_to_model(u_deg: np.ndarray, ctrlrange: np.ndarray, fill=0.8) -> np.ndarray:
    return CommandMap(u_deg, ctrlrange, fill).to_model(u_deg)


class ArmSim:
    def __init__(self, xml_path: str | Path = DEFAULT_XML, physics_hz: int = 600):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = 1.0 / physics_hz
        self.data = mujoco.MjData(self.model)
        self.physics_hz = physics_hz
        self.base_gain = self.model.actuator_gainprm[:, 0].copy()
        self.base_bias = self.model.actuator_biasprm[:, :3].copy()
        self.base_force = self.model.actuator_forcerange.copy()
        self.ctrlrange = self.model.actuator_ctrlrange.copy()
        self.nu = self.model.nu

    def configure(self, servo: SimServo, joints=None):
        m = self.model
        joints = range(self.nu) if joints is None else joints
        m.actuator_gainprm[:, 0] = self.base_gain
        m.actuator_biasprm[:, :3] = self.base_bias
        m.actuator_forcerange[:] = self.base_force
        pick = lambda v, j: float(np.broadcast_to(v, (self.nu,))[j])
        for j in joints:
            if servo.kp is not None:
                m.actuator_gainprm[j, 0] = pick(servo.kp, j)
                m.actuator_biasprm[j, 1] = -pick(servo.kp, j)
            if servo.kv is not None:
                m.actuator_biasprm[j, 2] = -pick(servo.kv, j)
            if servo.force_limit is not None:
                fl = pick(servo.force_limit, j)
                m.actuator_forcerange[j] = [-fl, fl]

    def rollout(self, u_rad: np.ndarray, fps: float, servo: SimServo, rng=None, joints=None):
        """Simulate one episode. Returns (u_deg, x_deg) sampled at fps."""
        mj = self.mj
        rng = rng or np.random.default_rng(0)
        self.configure(servo, joints)
        m, d = self.model, self.data
        mj.mj_resetData(m, d)
        d.qpos[: self.nu] = u_rad[0]
        d.ctrl[:] = u_rad[0]
        mj.mj_forward(m, d)
        n = len(u_rad)
        h = 1.0 / fps
        dt = m.opt.timestep
        x = np.empty_like(u_rad)
        q = 2 * np.pi / servo.ticks_per_rev if servo.ticks_per_rev else 0.0
        t = 0.0
        dead = np.broadcast_to(np.asarray(servo.dead_time, float), (self.nu,))
        cols = np.arange(self.nu)
        for k in range(n):
            # observe at frame time k*h
            x[k] = d.qpos[: self.nu]
            t_end = (k + 1) * h
            while t < t_end - 1e-12:
                # command in force at physical time t is the one sent at t - dead_time
                i = np.clip(np.floor((t - dead) / h + 1e-9).astype(int), 0, n - 1)
                d.ctrl[:] = u_rad[i, cols]
                mj.mj_step(m, d)
                t += dt
        if q:
            x = np.round(x / q) * q
        xd = np.rad2deg(x)
        if servo.noise_deg:
            xd = xd + rng.normal(0, servo.noise_deg, xd.shape)
        return np.rad2deg(u_rad), xd

    def rollout_policy(self, policy, n: int, fps: float, servo: SimServo, q0_deg: np.ndarray):
        """Closed loop in the LeRobot record order: observe x[k] (degrees, encoder
        quantised), call u[k] = policy(k, x_history_deg), hold u[k] for one frame.
        Returns (u_deg, x_deg), each (n, nu)."""
        mj = self.mj
        self.configure(servo)
        m, d = self.model, self.data
        mj.mj_resetData(m, d)
        q0 = np.deg2rad(np.asarray(q0_deg, float))
        d.qpos[: self.nu] = q0
        d.ctrl[:] = q0
        mj.mj_forward(m, d)
        h = 1.0 / fps
        steps = int(round(h / m.opt.timestep))
        q = 2 * np.pi / servo.ticks_per_rev if servo.ticks_per_rev else 0.0
        dead = np.broadcast_to(np.asarray(servo.dead_time, float), (self.nu,))
        lag_steps = np.round(dead / m.opt.timestep).astype(int)
        hist = []      # commands per physics step, for dead time
        X, U = [], []
        for k in range(n):
            xr = d.qpos[: self.nu].copy()
            if q:
                xr = np.round(xr / q) * q
            X.append(np.rad2deg(xr))
            u = np.clip(np.deg2rad(np.asarray(policy(k, X), float)), self.ctrlrange[:, 0], self.ctrlrange[:, 1])
            U.append(np.rad2deg(u))
            for _ in range(steps):
                hist.append(u)
                idx = np.maximum(len(hist) - 1 - lag_steps, 0)
                d.ctrl[:] = [hist[i][j] for j, i in enumerate(idx)]
                mj.mj_step(m, d)
        return np.array(U), np.array(X)
