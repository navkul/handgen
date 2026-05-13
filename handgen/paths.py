from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PACKAGE_ROOT / "data"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
DB_PATH = DATA_DIR / "handgen.sqlite3"

DEFAULT_DIFFBRUSH_ROOT = RESEARCH_ROOT / "04_cpu_neural_scout" / "vendor" / "DiffBrush"
DEFAULT_DIFFBRUSH_CHECKPOINT = DEFAULT_DIFFBRUSH_ROOT / "model_zoo" / "DiffBrush-ckpt.pt"
DEFAULT_DIFFBRUSH_RUNNER = RESEARCH_ROOT / "04_cpu_neural_scout" / "code" / "run_diffbrush_single.py"
DEFAULT_DIFFBRUSH_PYTHON = RESEARCH_ROOT / "04_cpu_neural_scout" / ".venv" / "bin" / "python"
