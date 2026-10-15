"""demodyne command line.

  demodyne ident  <dataset>        per-joint latency / bandwidth model
  demodyne audit  <dataset>        physics-level data integrity checks
  demodyne report <dataset> -o f   both, as Markdown + JSON
  demodyne sim-params <dataset>    MuJoCo position-actuator kp/kv + dead time
  demodyne fleet <results_dir>     population summary of a census directory
  demodyne pool-check <ds> <ds>..  can these datasets be trained on together?
  demodyne realign <dataset> -o d  copy with actions shifted to a common alignment

<dataset> is a local LeRobot dataset directory or a Hugging Face repo id.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import load
from .audit import audit
from .ident import fit_dataset


def _ms(v):
    return "  n/a" if v is None or not np.isfinite(v) else f"{v * 1e3:5.1f}"


def ident_table(ds, fits) -> str:
    lines = [f"# Dynamics: {ds.repo_id}",
             f"robot_type={ds.robot_type}  fps={ds.fps:g}  episodes={len(ds.episodes)}  "
             f"frames={sum(len(e) for e in ds.episodes)}", "",
             "| joint | dead time L (ms) | time const T (ms) | eff. lag L+T (ms) [90% CI] | offset | deadband | "
             "rate limit (/s) | R2 step | free-run RMSE | naive RMSE | stall % |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in fits:
        p = f.params
        ci = f.ci.get("lag")
        cis = f" [{_ms(ci[0]).strip()}, {_ms(ci[1]).strip()}]" if ci else ""
        lines.append(f"| {f.name} | {_ms(p.L)} | {_ms(p.T)} | {_ms(f.lag)}{cis} | {p.c:.2f} | {p.db:.2f} | "
                     f"{p.vmax:.0f} | {f.r2_one_step:.3f} | {f.rmse_free_run:.2f} | {f.baseline_rmse:.2f} | "
                     f"{100 * f.stall_fraction:.0f} |")
    return "\n".join(lines)


def audit_table(findings) -> str:
    if not findings:
        return "## Audit\nNo findings."
    icon = {"error": "ERROR", "warning": "WARN", "info": "info"}
    lines = ["## Audit"]
    for f in findings:
        eps = f" (episodes {f.episodes[:10]}{'...' if len(f.episodes) > 10 else ''})" if f.episodes else ""
        j = f" [{f.joint}]" if f.joint else ""
        lines.append(f"- **{icon[f.severity]}** `{f.check}`{j}: {f.message}{eps}")
    return "\n".join(lines)


def sim_params(fits, kp=50.0):
    """Map identified FOPDT to an overdamped MuJoCo position servo (T ~ kv/kp),
    plus the dead time to add as a command buffer.  The plant's own apparent
    dead time (~20 ms for the Menagerie SO-ARM100 at 30 fps, experiment V1) is
    already produced by the simulated dynamics and is subtracted."""
    out = {}
    for f in fits:
        if not np.isfinite(f.params.L) or f.r2_one_step < 0.5:
            continue
        out[f.name] = {"kp": kp, "kv": round(kp * f.params.T, 4),
                       "extra_command_delay_s": round(max(0.0, f.params.L - 0.020), 4),
                       "identified": f.to_dict()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="demodyne", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ident", "audit", "report", "sim-params"):
        s = sub.add_parser(name)
        s.add_argument("dataset")
        s.add_argument("--max-episodes", type=int, default=None)
        s.add_argument("--bootstrap", type=int, default=30 if name in ("ident", "report") else 0)
        s.add_argument("--json", type=str, default=None, help="also write machine-readable JSON here")
        if name == "report":
            s.add_argument("-o", "--out", type=str, default=None)
        if name == "sim-params":
            s.add_argument("--kp", type=float, default=50.0)
    s = sub.add_parser("fleet")
    s.add_argument("results_dir")
    s = sub.add_parser("pool-check", help="can these datasets be pooled for training?")
    s.add_argument("datasets", nargs="+")
    s.add_argument("--max-episodes", type=int, default=40)
    s = sub.add_parser("realign", help="write a copy with actions shifted to a target alignment")
    s.add_argument("dataset")
    s.add_argument("-o", "--out", required=True)
    g = s.add_mutually_exclusive_group()
    g.add_argument("--target-frames", type=float, default=None,
                   help="target alignment in frames (default: census median of current LeRobot record, +0.60)")
    g.add_argument("--shift-frames", type=float, default=None, help="apply this shift instead of measuring")
    s.add_argument("--max-episodes", type=int, default=60, help="episodes used to measure alignment")
    s.add_argument("--copy-videos", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "realign":
        from .realign import CENSUS_RECORD_ALIGNMENT, realign
        tgt = CENSUS_RECORD_ALIGNMENT if a.target_frames is None else a.target_frames
        rec = realign(a.dataset, a.out, target_frames=tgt, shift_frames=a.shift_frames,
                      max_episodes=a.max_episodes, copy_videos=a.copy_videos)
        print(json.dumps(rec, indent=1))
        return 0

    if a.cmd == "pool-check":
        from .pool import compare, profile
        profs = []
        for d in a.datasets:
            p = profile(load(d, max_episodes=a.max_episodes))
            profs.append(p)
            print(f"{d[:60]:60s} fps={p['fps']:g} align={p['alignment_frames']:+.2f} fr  "
                  f"lag={p['lag_frames']:.2f} fr  episodes={p['n_episodes']}")
        issues = compare(profs)
        print("\n" + ("\n".join(f"{s.upper()}: {m}" for s, m in issues) if issues else "OK: no pooling hazards found"))
        return 2 if any(s == "error" for s, _ in issues) else 0

    if a.cmd == "fleet":
        from .fleet import load_results, population
        res = load_results(a.results_dir)
        pop = population(res)
        ok = sum(r.get("ok", False) for r in res)
        print(f"{ok}/{len(res)} datasets analysed")
        for j, p in pop.items():
            q = [round(v * 1e3) for v in p["lag_q"]]
            print(f"{j:14s} n={p['n']:3d}  effective lag ms  p5={q[0]} p25={q[1]} median={q[2]} p75={q[3]} p95={q[4]}"
                  f"  outliers={len(p['outliers'])}")
        return 0

    ds = load(a.dataset, max_episodes=a.max_episodes)
    fits = fit_dataset(ds, bootstrap=a.bootstrap)
    payload = {"dataset": ds.repo_id, "robot_type": ds.robot_type, "fps": ds.fps,
               "fits": [f.to_dict() for f in fits]}
    if a.cmd == "ident":
        print(ident_table(ds, fits))
    elif a.cmd == "audit":
        fs = audit(ds, fits)
        payload["findings"] = [f.to_dict() for f in fs]
        print(audit_table(fs))
    elif a.cmd == "report":
        fs = audit(ds, fits)
        payload["findings"] = [f.to_dict() for f in fs]
        md = ident_table(ds, fits) + "\n\n" + audit_table(fs) + "\n"
        if a.out:
            Path(a.out).write_text(md, encoding="utf-8")
            print(f"wrote {a.out}")
        else:
            print(md)
    elif a.cmd == "sim-params":
        sp = sim_params(fits, a.kp)
        payload["sim_params"] = sp
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "identified"} for k, v in sp.items()}, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(payload, indent=1, default=float))
    errors = [f for f in payload.get("findings", []) if f["severity"] == "error"]
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
