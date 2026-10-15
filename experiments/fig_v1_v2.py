"""Figure for V1 (dead-time recovery) and V2 (stiffness sweep) from results/v1_v2.json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = Path(__file__).parent / "results"
d = json.load(open(RES / "v1_v2.json"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
joints = sorted({r["joint"] for r in d["raw"]["v1"]})
for j in joints:
    rows = [r for r in d["raw"]["v1"] if r["joint"] == j]
    t = np.array([r["true_L"] for r in rows]) * 1e3
    lag = np.array([r["lag"] for r in rows]) * 1e3
    ax[0].plot(t, lag - lag[0], "o-", ms=3, lw=1, label=j.replace("main_", ""))
ax[0].plot([0, 100], [0, 100], "k:", lw=1, label="ideal (slope 1)")
ax[0].set_xlabel("injected command dead time (ms)")
ax[0].set_ylabel("Δ identified effective lag (ms)")
ax[0].set_title("V1: MuJoCo SO-ARM100, real leader trajectories\nslope 1.003–1.005, residual 0.4–0.5 ms")
ax[0].legend(fontsize=7)
for j in joints:
    rows = [r for r in d["raw"]["v2"] if r["joint"] == j]
    kp = np.array([r["kp"] for r in rows])
    ax[1].loglog(kp, np.array([r["lag"] for r in rows]) * 1e3, "o-", ms=3, lw=1)
kp = np.array([12.5, 25, 50, 100, 200])
ax[1].loglog(kp, 5.0 / kp * 1e3, "k:", label="kv/kp (overdamped servo)")
ax[1].axhline(1e3 / 30, color="gray", lw=0.8, ls="--")
ax[1].text(13, 36, "1 frame @30 fps", fontsize=7, color="gray")
ax[1].set_xticks(kp, [f"{k:g}" for k in kp]); ax[1].minorticks_off()
ax[1].set_xlabel("servo stiffness kp (kv = 5 fixed)")
ax[1].set_ylabel("identified effective lag (ms)")
ax[1].set_title("V2: lag tracks kv/kp; resolution limit below ~1 frame")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(RES / "v1_v2.png", dpi=130)
print("wrote", RES / "v1_v2.png")
