"""V6: population-level findings from the census directory.

  * prevalence of each audit finding across community datasets
  * distribution of effective lag / dead time / time constant per joint
  * between-arm vs within-arm variability of lag (is the spread real?)
  * figure: lag distribution per joint + finding prevalence
Usage: python experiments/v6_census_analysis.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from demodyne.fleet import CANON_SO, load_results, population, usable

RES = Path(__file__).parent / "results"


def main():
    import sys
    R = load_results(RES / (sys.argv[1] if len(sys.argv) > 1 else "census_final"))
    ok = [r for r in R if r.get("ok")]
    print(f"datasets attempted={len(R)}  analysed={len(ok)}")
    print("failure reasons:", Counter(r.get("error", "")[:50] for r in R if not r.get("ok")).most_common(6))
    print("robot_type:", Counter(r["robot_type"] for r in ok).most_common(8))
    print("fps:", Counter(r["fps"] for r in ok).most_common(6))
    print("codebase:", Counter(r.get("codebase_version") for r in ok).most_common())

    # ---- data sufficiency: which datasets carry identifiable command dynamics
    ident = [r for r in ok if sum(bool(usable(r, j)) for j in CANON_SO[:5]) >= 3]
    print(f"datasets with >=3 identifiable arm joints (R2>=0.8, motion>=15%): {len(ident)}/{len(ok)}")

    # ---- prevalence of findings (dataset-level)
    prev = defaultdict(set)
    for r in ok:
        for f in r["findings"]:
            if f["severity"] in ("error", "warning"):
                prev[(f["check"], f["severity"])].add(r["repo_id"])
    any_err = {r["repo_id"] for r in ok if any(f["severity"] == "error" for f in r["findings"])}
    any_warn = {r["repo_id"] for r in ok if any(f["severity"] in ("error", "warning") for f in r["findings"])}
    print(f"\nfindings prevalence over {len(ok)} analysed datasets:")
    for (c, s), v in sorted(prev.items(), key=lambda kv: -len(kv[1])):
        print(f"  {c:24s} {s:8s} {len(v):4d}  ({len(v) / len(ok):.1%})")
    print(f"  ANY error: {len(any_err)} ({len(any_err)/len(ok):.1%});  ANY error/warning: {len(any_warn)} "
          f"({len(any_warn)/len(ok):.1%})")
    causal = sorted(prev.get(("causality", "error"), set()))
    print("  causality errors:", causal[:15])

    # ---- population dynamics
    pop = population(ident)
    print("\neffective lag L+T [ms] by joint (datasets passing quality gate):")
    for j, p in pop.items():
        q = np.round(np.array(p["lag_q"]) * 1e3).astype(int)
        Lq = np.round(np.array(p["L_q"]) * 1e3).astype(int)
        Tq = np.round(np.array(p["T_q"]) * 1e3).astype(int)
        print(f"  {j:14s} n={p['n']:3d}  lag p5/p25/p50/p75/p95 = {q.tolist()}  L median={Lq[2]}  T median={Tq[2]}"
              f"  outliers={len(p['outliers'])}")

    # ---- is between-arm spread real? compare to within-dataset bootstrap width
    # each dataset's lag CI is not stored without bootstrap; use per-episode-free proxy:
    # spread of dataset medians vs median |lag_j - lag_k| across joints within dataset.
    meds, within = [], []
    for r in ident:
        lags = [usable(r, j)["lag"] for j in CANON_SO[:5] if usable(r, j)]
        meds.append(np.median(lags))
        within.append(np.std(lags))
    meds = np.array(meds)
    print(f"\nper-dataset median lag: p5={np.quantile(meds,.05)*1e3:.0f} p50={np.median(meds)*1e3:.0f} "
          f"p95={np.quantile(meds,.95)*1e3:.0f} ms; ratio p95/p5 = {np.quantile(meds,.95)/np.quantile(meds,.05):.2f}")
    print(f"between-dataset std of median lag = {meds.std()*1e3:.1f} ms; mean within-dataset std across joints "
          f"= {np.mean(within)*1e3:.1f} ms")

    # ---- fps / policy-vs-teleop split
    def is_eval(r):
        return "eval" in r["repo_id"].lower()
    for label, grp in [("teleop", [r for r in ident if not is_eval(r)]), ("policy eval", [r for r in ident if is_eval(r)])]:
        m = [np.median([usable(r, j)["lag"] for j in CANON_SO[:5] if usable(r, j)]) for r in grp]
        if m:
            print(f"  {label:12s} n={len(m):3d} median lag={np.median(m)*1e3:.0f} ms  IQR=[{np.quantile(m,.25)*1e3:.0f},"
                  f"{np.quantile(m,.75)*1e3:.0f}]")

    json.dump({"n_attempted": len(R), "n_ok": len(ok), "n_ident": len(ident),
               "prevalence": {f"{c}/{s}": sorted(v) for (c, s), v in prev.items()},
               "any_error": sorted(any_err), "population": pop,
               "dataset_median_lag": dict(zip([r["repo_id"] for r in ident], meds.tolist()))},
              open(RES / "v6_census_summary.json", "w"), indent=1)
    plot(ident, prev, len(ok))


def plot(ident, prev, n_ok):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    data = [[usable(r, j)["lag"] * 1e3 for r in ident if usable(r, j)] for j in CANON_SO[:5]]
    ax[0].boxplot(data, showfliers=True)
    ax[0].set_xticks(range(1, 6), [j.replace("_", "\n") for j in CANON_SO[:5]])
    ax[0].set_ylabel("effective command→response lag L+T (ms)")
    ax[0].set_title(f"SO-100/101 community arms (n={len(ident)} datasets)")
    ax[0].axhline(100, ls=":", c="gray")
    ax[0].text(5.4, 101, "Menagerie\ndefault ≈100", fontsize=7, va="bottom", ha="right", color="gray")
    items = sorted(((f"{c} ({s})", len(v) / n_ok * 100) for (c, s), v in prev.items()), key=lambda t: t[1])
    ax[1].barh([i[0] for i in items], [i[1] for i in items], color="#c0504d")
    ax[1].set_xlabel(f"% of {n_ok} datasets")
    ax[1].set_title("Physics-level findings (error / warning)")
    fig.tight_layout()
    fig.savefig(RES / "v6_census.png", dpi=130)
    print("wrote", RES / "v6_census.png")


if __name__ == "__main__":
    main()
