"""Tiny WESAD-format pickles with class-dependent signals, for fast pipeline tests."""
import pickle

import numpy as np

FS = 700
# (protocol label, seconds): mirrors the real study order, heavily shortened
BLOCKS = [(0, 10), (1, 150), (0, 10), (2, 100), (0, 10), (4, 70), (3, 70), (0, 10), (4, 70)]
HEART_RATE = {0: 70, 1: 70, 2: 95, 3: 78, 4: 62}


def make_subject(seed):
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.full(d * FS, lab, np.int32) for lab, d in BLOCKS])
    n = len(labels)
    t = np.arange(n) / FS
    phase = 2 * np.pi * np.cumsum(np.array([HEART_RATE[l] for l in labels]) / 60) / FS
    chest = {
        "ECG": np.sin(phase) ** 31 + 0.05 * rng.standard_normal(n),
        "EDA": 2 + seed + 0.8 * (labels == 2) + 0.01 * rng.standard_normal(n),
        "EMG": rng.standard_normal(n) * (0.02 + 0.03 * (labels == 3)),
        "Temp": 33 + 0.5 * seed - 0.3 * (labels == 2) + 0.001 * t,
        "Resp": np.sin(2 * np.pi * np.where(labels == 4, 0.1, 0.25) * t),
        "ACC": np.zeros((n, 3)),
    }
    chest = {k: v.reshape(n, -1) for k, v in chest.items()}
    return {"signal": {"chest": chest, "wrist": {}}, "label": labels}


def write_fake_wesad(root, subjects=("S2", "S3", "S4", "S6", "S15")):
    for i, s in enumerate(subjects):
        (root / s).mkdir(parents=True, exist_ok=True)
        with open(root / s / f"{s}.pkl", "wb") as f:
            pickle.dump({**make_subject(i), "subject": s}, f)
    return root
