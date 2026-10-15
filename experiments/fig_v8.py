"""Figure for V8 (results/v8_realign.json)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = Path(__file__).parent / "results"
d = json.load(open(RES / "v8_realign.json"))
rows = [d["shifts"][k] for k in sorted(d["shifts"], key=float)]
s = [r["true_shift"] for r in rows]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
style = {"naive_pool": ("#c0504d", "naive pool A+B"), "realigned_pool": ("#4f81bd", "A + B re-aligned by demodyne"),
         "oracle_pool": ("k", "oracle pool (true labels)"), "A_only": ("gray", "A only (half the data)")}
for name, (c, lab) in style.items():
    for i, key in enumerate(["closed_loop_rmse", "action_rmse"]):
        m = [np.mean(r["conditions"][name][key]) for r in rows]
        e = [np.std(r["conditions"][name][key]) for r in rows]
        ls = ":" if name in ("oracle_pool", "A_only") else "-"
        ax[i].errorbar(s, m, yerr=e, color=c, ls=ls, marker="o", ms=4, capsize=2, label=lab)
ax[0].axhline(d["expert_rmse"], color="green", lw=0.8, ls="--", label="expert")
ax[0].set_ylabel("closed-loop tracking RMSE (deg)")
ax[1].set_ylabel("held-out action RMSE vs true labels (deg)")
for a in ax:
    a.set_xlabel("action-column staleness of dataset B (frames @30 fps)")
ax[0].set_title("V8: policy quality vs label misalignment (5 seeds)")
ax[1].set_title("measured correction = true shift ± 0.05 frames")
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(RES / "v8_realign.png", dpi=130)
print("wrote", RES / "v8_realign.png")
