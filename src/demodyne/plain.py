"""Plain-language summary of an identification + audit, for non-specialists."""
from __future__ import annotations

import numpy as np

from .audit import alignment_offset

_EXPLAIN = {
    "causality": "Timing of actions vs. observations: the robot appears to move before it was told to. "
                 "The action column is offset in time relative to what the robot saw.",
    "mapping": "Wiring/labels: a joint follows another joint's command, or moves the opposite way. "
               "Channels are probably swapped or a direction is inverted.",
    "frozen": "Dead motor: a joint did not move although it was commanded (motor off or unplugged).",
    "range_mismatch": "Range: the follower arm only reaches part of the range the operator commanded "
                      "(calibration differs between the two arms, or the joint hits a limit).",
    "stale_frames": "Repeated sensor readings: the recorded position sometimes freezes for one frame in "
                    "the middle of fast motion, which a real arm cannot do.",
    "timing": "Timestamps: frames are missing, duplicated or very short episodes exist.",
    "latency_outlier": "Inconsistent setup: a few episodes respond much slower or faster than the rest.",
    "action_is_future_state": "Actions are copies of later positions rather than real commands.",
    "stalls": "Joint held away from its target without moving (pressing on something, or overloaded).",
}


def summary(ds, fits, findings) -> str:
    errors = [f for f in findings if f.severity == "error"]
    warns = [f for f in findings if f.severity == "warning"]
    if errors:
        verdict = "🔴 **Problems found.** Fix or exclude this dataset before training on it."
    elif warns:
        verdict = "🟡 **Usable, with caveats.** Read the notes below before mixing it with other data."
    else:
        verdict = "🟢 **Looks healthy.** No physical inconsistencies detected."
    good = [f for f in fits if np.isfinite(f.lag) and f.r2_one_step >= 0.8
            and not any(k in f.name.lower() for k in ("grip", "jaw"))]
    lines = [verdict, ""]
    if good:
        lag = np.median([f.lag for f in good]) * 1e3
        lines.append(f"- **Arm reaction time:** about **{lag:.0f} ms** from command to movement "
                     f"(typical SO-100/101 arms: 54–123 ms; median 88 ms).")
    else:
        lines.append("- **Arm reaction time:** could not be measured (too little movement or no "
                     "command/response pairing).")
    al = alignment_offset(fits) / ds.h if fits else float("nan")
    if np.isfinite(al):
        lines.append(f"- **Action timing alignment:** {al:+.2f} frames (current LeRobot recordings are "
                     f"around +0.6). If you combine datasets, `demodyne realign` can bring this one in line.")
    seen = set()
    for f in errors + warns:
        if f.check in seen:
            continue
        seen.add(f.check)
        joint = f" (joint: {f.joint})" if f.joint else ""
        lines.append(f"- {'⛔' if f.severity == 'error' else '⚠️'} {_EXPLAIN.get(f.check, f.message)}{joint}")
    return "\n".join(lines)
