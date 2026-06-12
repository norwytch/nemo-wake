"""Shared setup for nemo-core replication scripts."""
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEMO_DIR = PROJECT_ROOT / "nemo-core"
RESULTS_ROOT = PROJECT_ROOT / "results"


def ensure_hash_seed(seed: int = 0) -> None:
    """If PYTHONHASHSEED isn't set to `seed`, re-exec the current script with it.

    Hash randomization perturbs set/dict-of-string iteration order, which
    propagates into RNG consumption order in brain.project() and breaks
    reproducibility even when numpy/random are seeded. Must be called as the
    first action in main() so the re-exec happens before any RNG work.
    """
    target = str(seed)
    if os.environ.get("PYTHONHASHSEED") == target:
        return
    os.environ["PYTHONHASHSEED"] = target
    os.execvp(sys.executable, [sys.executable, *sys.argv])


def add_nemo_to_path() -> None:
    sys.path.insert(0, str(NEMO_DIR))


def setup_seeded(seed: int) -> None:
    """Seed numpy, Python random, and patch brain.Brain to use the seed.

    Must be called *after* add_nemo_to_path() and *before* importing learner /
    word_order_int. brain.Brain.__init__'s seed default is monkey-patched
    because nemo-core's LearnBrain subclasses call brain.Brain.__init__(self, p)
    without forwarding a seed.
    """
    np.random.seed(seed)
    random.seed(seed)
    import brain
    _orig_init = brain.Brain.__init__
    captured = seed

    def patched_init(self, p, save_size=True, save_winners=False, seed=captured):
        _orig_init(self, p, save_size=save_size, save_winners=save_winners, seed=seed)

    brain.Brain.__init__ = patched_init


def make_run_dir(experiment: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_ROOT / f"{experiment}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def open_tee_log(run_dir: Path):
    """Install a Tee on sys.stdout that mirrors output to run_dir/run.log.

    Returns the log_file. On interpreter shutdown, sys.stdout is restored to
    its original before the log file is closed, so atexit flushes don't
    hit a closed handle.
    """
    import atexit

    log_path = run_dir / "run.log"
    log_file = open(log_path, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    def restore():
        sys.stdout = original_stdout
        log_file.close()

    atexit.register(restore)
    return log_file


def save_json(run_dir: Path, name: str, data: dict) -> None:
    with open(run_dir / name, "w") as f:
        json.dump(data, f, indent=2)
