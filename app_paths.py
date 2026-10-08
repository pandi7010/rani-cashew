"""All file locations of the grader, relative to this folder (never to the current directory)."""
import logging
import logging.handlers
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CFG_PATH = BASE_DIR / "pneumatic_config.json"
CLASS_MAPPING_PATH = BASE_DIR / "cashew_class_mapping.json"
MODELS_DIR = BASE_DIR / "saved_models"
CNN_MODEL_PATH = MODELS_DIR / "cashew_color_cnn_new.pth"
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DISABLED_DIR = BASE_DIR / "dataset_disabled"
GOLDEN_DIR = BASE_DIR / "cnn_final" / "golden_crops"
FULL_FRAME_DIR = BASE_DIR / "cnn_final" / "golden_full_frame"
ASSETS_DIR = BASE_DIR / "assets"
NAV_ICONS_DIR = ASSETS_DIR / "nav_icons"
RUNS_DIR = BASE_DIR / "runs"
LOGS_DIR = BASE_DIR / "logs"
FFC_GAIN_MAP_PATH = BASE_DIR / "ffc_gain_map.png"
AUTO_LABELER_SCRIPT = BASE_DIR / "ai_auto_labeler.py"

# The magic wand and TCM Studio tools live in another project, outside this folder.
# Set RANIVISION_DIR to point somewhere else; features that need them are skipped when it is missing.
EXTERNAL_TOOLS_DIR = Path(os.environ.get("RANIVISION_DIR", "/home/phoenix/Documents/RaniVision_Latest"))
TCM_STUDIO_SCRIPT = EXTERNAL_TOOLS_DIR / "tcm_studio_pro.py"


def ensure_dirs():
    for folder in (MODELS_DIR, DATASET_DIR, GOLDEN_DIR, FULL_FRAME_DIR, LOGS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def resolve(path) -> Path:
    """A path stored in the config (relative to this folder, or absolute) as an absolute path."""
    p = Path(str(path))
    return p if p.is_absolute() else BASE_DIR / p


def relative(path) -> str:
    """How a path is stored in the config: relative to this folder when it is inside it."""
    p = Path(str(path)).resolve()
    try:
        return str(p.relative_to(BASE_DIR))
    except ValueError:
        return str(p)


def setup_logging(name="grader") -> logging.Logger:
    """Console plus logs/<name>.log (5 files of 2 MB). Safe to call more than once."""
    root = logging.getLogger()
    if getattr(root, "_cashew_configured", False):
        return logging.getLogger(name)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    file_handler = logging.handlers.RotatingFileHandler(LOGS_DIR / f"{name}.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console)
    root._cashew_configured = True
    return logging.getLogger(name)
