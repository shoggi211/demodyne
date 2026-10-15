"""Re-run identification + audit with the current code on every dataset the
census downloaded (local cache only), into results/census_final/."""
import json
from pathlib import Path

from demodyne.fleet import load_results, run_census

RES = Path(__file__).parent / "results"

if __name__ == "__main__":
    ids = [r["repo_id"] for r in load_results(RES / "census") if r.get("ok")]
    print(len(ids), "datasets to re-audit", flush=True)
    run_census(ids, RES / "census_final", workers=8, max_episodes=60)
