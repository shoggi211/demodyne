"""V4: fleet census of SO-100/SO-101 community datasets on the Hugging Face Hub.

Samples tagged datasets, identifies every joint and audits every dataset.
Results land in experiments/results/census/<repo>.json (resumable).
Usage: python experiments/v4_census.py [n_datasets] [workers]
"""
import json
import random
import sys
from pathlib import Path

from huggingface_hub import HfApi

from demodyne.fleet import run_census

OUT = Path(__file__).parent / "results"


def candidates():
    p = OUT / "so_tagged.json"
    if p.exists():
        return json.loads(p.read_text())
    api = HfApi()
    ids = set()
    for tag in ("so100", "so101"):
        ids.update(d.id for d in api.list_datasets(filter=["LeRobot", tag], limit=20000))
    ids = sorted(ids)
    p.write_text(json.dumps(ids))
    return ids


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ids = candidates()
    random.Random(0).shuffle(ids)
    print(f"{len(ids)} tagged so100/so101 datasets; analysing {n}", flush=True)
    run_census(ids[:n], OUT / "census", workers=workers, max_episodes=60)
