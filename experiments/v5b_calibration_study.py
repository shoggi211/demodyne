"""V5b: calibration benefit vs. how far an arm is from the default sim model.

Selects census datasets with >=5 well-identified arm joints, 6 channels, >=20
episodes and no alignment error, spread over the observed lag range; runs V5 on
each and relates held-out improvement to |lag_real - lag_default_sim|.
Usage: python experiments/v5b_calibration_study.py [n_datasets]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from demodyne.fleet import CANON_SO, load_results, usable

RES = Path(__file__).parent / "results"
sys.path.insert(0, str(Path(__file__).parent))
from v5_sim_calibration import main as run_v5  # noqa: E402

DEFAULT_SIM_LAG = 0.095  # Menagerie SO-ARM100 effective lag under real excitation (V1, L_true=0)


def pick(n):
    cands = []
    for r in load_results(RES / "census_final"):
        if not r.get("ok") or len(r["joint_names"]) != 6 or r["n_episodes"] < 20:
            continue
        if any(f["check"] == "causality" for f in r["findings"]):
            continue
        fs = [usable(r, j) for j in CANON_SO[:5]]
        if not all(fs):
            continue
        cands.append((float(np.median([f["lag"] for f in fs])), r["repo_id"]))
    cands.sort()
    if len(cands) <= n:
        return cands
    idx = np.unique(np.round(np.linspace(0, len(cands) - 1, n)).astype(int))
    return [cands[i] for i in idx]


def main(n=12):
    rows = []
    for lag, rid in pick(n):
        tag = Path(rid).name
        f = RES / f"v5_{tag}.json"
        if not f.exists():
            print(f"\n=== {rid} (lag {lag*1e3:.0f} ms)", flush=True)
            try:
                run_v5(rid, 10, 10)
            except Exception as e:
                print("failed:", e)
                continue
        if not f.exists():
            continue
        d = json.loads(f.read_text())["results"]
        a = np.mean(d["menagerie_default"]["rmse_dyn"])
        b = np.mean(d["demodyne_calibrated"]["rmse_dyn"])
        rows.append({"repo_id": rid, "lag": lag, "default": a, "calibrated": b,
                     "fopdt": float(np.mean(d["demodyne_fopdt_model"]["rmse_dyn"])),
                     "naive": float(np.mean(d["naive_state_equals_command"]["rmse_dyn"])),
                     "rel_improvement": (a - b) / a})
    dist = np.array([abs(r["lag"] - DEFAULT_SIM_LAG) for r in rows])
    imp = np.array([r["rel_improvement"] for r in rows])
    print("\nrepo | lag ms | default | calibrated | improvement")
    for r in sorted(rows, key=lambda r: r["lag"]):
        print(f"{r['repo_id'][:55]:55s} {r['lag']*1e3:5.0f} {r['default']:6.2f} {r['calibrated']:6.2f} "
              f"{100*r['rel_improvement']:+6.1f}%")
    rho, p = spearmanr(dist, imp)
    print(f"\nn={len(rows)}  median improvement={100*np.median(imp):+.1f}%  improved in {np.sum(imp>0)}/{len(imp)}")
    print(f"Spearman(|lag - default|, improvement) = {rho:.2f} (p={p:.3f})")
    far = imp[dist > 0.025]
    near = imp[dist <= 0.025]
    print(f"arms >25 ms from default: n={len(far)} median {100*np.median(far) if len(far) else float('nan'):+.1f}%;"
          f"  within 25 ms: n={len(near)} median {100*np.median(near) if len(near) else float('nan'):+.1f}%")
    json.dump({"rows": rows, "spearman": [rho, p]}, open(RES / "v5b_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
