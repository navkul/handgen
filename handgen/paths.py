from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PACKAGE_ROOT / "data"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
DB_PATH = DATA_DIR / "handgen.sqlite3"

DEFAULT_DIFFBRUSH_ROOT = PACKAGE_ROOT / "models" / "diffbrush" / "third_party_repo"
DEFAULT_DIFFBRUSH_CHECKPOINT = DEFAULT_DIFFBRUSH_ROOT / "model_zoo" / "DiffBrush-ckpt.pt"
