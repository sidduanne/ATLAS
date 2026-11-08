import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve()
RESULTS_DIR = ROOT / "experiments" / "results"
ALLOWED_LABELS = ("relevant", "duplicate", "malformed", "out_of_domain", "irrelevant")


def data_path(relative: str) -> Path:
    path = (DATA_DIR / relative).resolve()
    if not path.is_relative_to(DATA_DIR):
        raise ValueError("Invalid asset path")
    return path
