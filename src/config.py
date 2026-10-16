"""Project-wide paths and constants."""
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[1]
WESAD_DIR     = ROOT / "WESAD"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR    = ROOT / "outputs"
MODEL_DIR     = ROOT / "models"

# WESAD has 15 usable subjects (S1 and S12 were discarded by the authors)
SUBJECTS = [f"S{i}" for i in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17)]

FS_RAW   = 700            # RespiBAN chest sampling rate (Hz)
FS       = 100            # after anti-aliased decimation (factor 7)
CHANNELS = ["ECG", "EDA", "EMG", "Temp", "Resp"]

WINDOW_S = 60             # window length (seconds)
STEP_S   = 15             # hop between windows (seconds) -> 75% overlap

# WESAD protocol labels we keep (0 = transient, 5-7 = ignore)
LABELS      = {1: "Baseline", 2: "Stress", 3: "Amusement", 4: "Meditation"}
CLASS_NAMES = list(LABELS.values())

# Classification tasks: original class index (0-3) -> task class index
TASKS = {
    "4class": {"classes": ["Baseline", "Stress", "Amusement", "Meditation"],
               "map": {0: 0, 1: 1, 2: 2, 3: 3}},
    "3class": {"classes": ["Baseline", "Stress", "Amusement"],
               "map": {0: 0, 1: 1, 2: 2}},
    # Binary setup from the WESAD paper: stress vs. non-stress (baseline + amusement)
    "binary": {"classes": ["Non-stress", "Stress"],
               "map": {0: 0, 1: 1, 2: 0}},
}

PROCESSED_FILE = PROCESSED_DIR / f"wesad_{FS}hz_{WINDOW_S}s_step{STEP_S}.npz"
