from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "cifar-10"
CHECKPOINT_ROOT = PYTHON_DIR / "checkpoints"
