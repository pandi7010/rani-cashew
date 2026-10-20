"""All file locations of the grader, relative to this folder (never to the current directory)."""
import logging
import logging.handlers
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CFG_PATH = BASE_DIR / "pneumatic_config.json"
CLASS_MAPPING_PATH = BASE_DIR / "cashew_class_mapping.json"
MODELS_DIR = BASE_DIR / "saved_models"
CNN_MODEL_PATH = MODELS_DIR / "cashew_color_cnn_new.pth"      # the CNN used until another one is activated
ACTIVE_MODELS_PATH = MODELS_DIR / "active_models.json"            # {"cnn_model": file name} set on AI Model Select
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DISABLED_DIR = BASE_DIR / "dataset_disabled"
GOLDEN_DIR = BASE_DIR / "cnn_final" / "golden_crops"
FULL_FRAME_DIR = BASE_DIR / "cnn_final" / "golden_full_frame"
ASSETS_DIR = BASE_DIR / "assets"
NAV_ICONS_DIR = ASSETS_DIR / "nav_icons"
RUNS_DIR = BASE_DIR / "runs"
LOGS_DIR = BASE_DIR / "logs"
CAPTURES_DIR = BASE_DIR / "captures"                     # captures/undefined/<date>/, captures/double/<date>/ (not in git)
DETECTS_DIR = BASE_DIR / "detects"                       # detects/<class or grade>/<date>/: every fired cashew (not in git)
AUTO_LABELER_SCRIPT = BASE_DIR / "ai_auto_labeler.py"
# The active program's data (5-track layout: programs/<name>/dataset, testset, inbox). One program until (e3).
PROGRAM_NAME = "Default"
PROGRAM_DIR = BASE_DIR / "programs" / PROGRAM_NAME
CLEAN_DATASET_DIR = PROGRAM_DIR / "dataset"            # one folder per final class name, used for training
TESTSET_DIR = PROGRAM_DIR / "testset"                  # every 10th confirmed image; never trained on
INBOX_DIR = PROGRAM_DIR / "inbox"                      # production capture, unlabelled
MACHINE_CONFIG_PATH = BASE_DIR / "machine_config.json" # belongs to the machine, not to a program (not in git)
FFC_GAIN_MAP_PATH = BASE_DIR / "ffc_gain_map.png"      # white calibration (flat-field) map
RECORDINGS_DIR = BASE_DIR / "recordings"                # camera frames recorded for replay tests

# Copies of the magic wand and TCM Studio tools (see external_tools/README.md).
EXTERNAL_TOOLS_DIR = BASE_DIR / "external_tools"
TCM_STUDIO_SCRIPT = EXTERNAL_TOOLS_DIR / "tcm_studio_pro.py"
FASTSAM_PATH = BASE_DIR / "FastSAM-s.pt"


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
