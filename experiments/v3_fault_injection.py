"""V3: detection rate of injected faults and false-alarm rate on clean data.

For each real dataset: audit the clean copy (any error/warning = false alarm,
counted per check), then inject each fault and record whether the audit raises a
finding of the expected check (naming the expected joint, when applicable).
Usage: python experiments/v3_fault_injection.py <dataset> [<dataset> ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import demodyne as dd
from demodyne import corrupt as C
from demodyne.audit import audit

OUT = Path(__file__).parent / "results"


def faults(ds):
    J = ds.n_joints
    return {
        "action_shift_-1": lambda: C.shift_actions(ds, -1),
        "action_shift_-2": lambda: C.shift_actions(ds, -2),
        "action_shift_-3": lambda: C.shift_actions(ds, -3),
        "swap_0_1": lambda: C.swap_channels(ds, 0, 1),
        "swap_2_3": lambda: C.swap_channels(ds, 2, 3),
        "invert_1": lambda: C.invert_channel(ds, 1),
        "invert_3": lambda: C.invert_channel(ds, 3),
        "freeze_2": lambda: C.freeze_joint(ds, 2, frac=0.1),
        "stale_5pct": lambda: C.stale_frames(ds, 0.05),
        "stale_2pct": lambda: C.stale_frames(ds, 0.02),
        "drop_3pct": lambda: C.drop_frames(ds, 0.03),
        "ep_latency_+50ms": lambda: C.episode_latency(ds, 0.05),
        "ep_latency_+100ms": lambda: C.episode_latency(ds, 0.10),
        "future_state_actions": lambda: C.future_state_actions(ds, 1),
    }


def detected(findings, truth, clean=()):
    """True if the expected check fires (naming the right joint/episodes) and the
    same finding was not already present in the clean dataset."""
    def key(f):
        return (f.check, f.joint)
    pre = {key(f) for f in clean if f.severity in ("error", "warning")}
    hits = [f for f in findings if f.check == truth["check"] and f.severity in ("error", "warning")
            and key(f) not in pre]
    if "joints" in truth:
        hits = [f for f in hits if f.joint is None or f.joint in truth["joints"]]
    if "episodes" in truth and truth["check"] == "latency_outlier":
        hits = [f for f in hits if set(truth["episodes"]) & set(f.episodes)]
    return bool(hits)


def main(paths, max_episodes=40):
    rows = []
    fa = defaultdict(int)
    n_clean = 0
    for p in paths:
        try:
            ds = dd.load(p, max_episodes=max_episodes)
        except Exception as e:
            print("skip", p, e)
            continue
        clean = audit(ds)
        n_clean += 1
        alarms = sorted({f.check for f in clean if f.severity in ("error", "warning")})
        for a in alarms:
            fa[a] += 1
        print(f"\n{p}: clean alarms={alarms}", flush=True)
        pre_checks = {f.check for f in clean if f.severity in ("error", "warning")}
        for name, make in faults(ds).items():
            bad, truth = make()
            if truth["check"] in pre_checks and truth["check"] not in ("mapping", "frozen"):
                # dataset-level finding already present: cannot attribute a detection
                print(f"  {name:22s} (n/a: '{truth['check']}' already fires on clean data)")
                continue
            det = detected(audit(bad), truth, clean)
            rows.append({"dataset": str(p), "fault": name, "detected": det, "clean_alarms": alarms})
            print(f"  {name:22s} {'DETECTED' if det else 'missed'}", flush=True)
    summ = defaultdict(list)
    for r in rows:
        summ[r["fault"]].append(r["detected"])
    print("\nDetection rate per fault:")
    for k, v in summ.items():
        print(f"  {k:22s} {np.mean(v):.0%} ({sum(v)}/{len(v)})")
    print(f"Clean (uncorrupted) datasets with an alarm, per check (of {n_clean}): {dict(fa)}")
    json.dump({"rows": rows, "false_alarms": fa, "n_clean": n_clean,
               "rates": {k: float(np.mean(v)) for k, v in summ.items()}},
              open(OUT / "v3_fault_injection.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
