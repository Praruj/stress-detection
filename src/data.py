"""Load WESAD subjects, clean the chest signals and cut them into labelled windows.

Run as a script to build the processed dataset:
    python -m src.data
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import decimate

from .config import (CHANNELS, FS, FS_RAW, LABELS, PROCESSED_FILE, STEP_S,
                     SUBJECTS, TASKS, WESAD_DIR, WINDOW_S)


def load_subject(subject, wesad_dir=WESAD_DIR):
    """Return raw chest signals (n_samples, n_channels) and per-sample labels."""
    with open(wesad_dir / subject / f"{subject}.pkl", "rb") as f:
        d = pickle.load(f, encoding="latin1")
    chest = d["signal"]["chest"]
    sig = np.column_stack([np.asarray(chest[c]).reshape(-1) for c in CHANNELS])
    return sig.astype(np.float32), np.asarray(d["label"]).astype(np.int8)


def preprocess_subject(sig, labels):
    """Anti-aliased downsampling to FS, then per-subject z-scoring.

    Normalising each subject by their own statistics removes person-specific
    offsets (skin conductance level, body temperature, electrode gain), which
    is what made the v1 model fail to generalise to unseen subjects.
    """
    q = FS_RAW // FS
    sig = decimate(sig, q, axis=0, zero_phase=True).astype(np.float32)
    labels = labels[::q][: len(sig)]
    mask = np.isin(labels, list(LABELS))
    mu, sd = sig[mask].mean(0), sig[mask].std(0) + 1e-8
    return (sig - mu) / sd, labels


def make_windows(sig, labels, window_s=WINDOW_S, step_s=STEP_S, fs=FS):
    """Slide a window over each contiguous block of a single protocol label."""
    win, step = window_s * fs, step_s * fs
    bounds = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
    X, y = [], []
    for s, e in zip(bounds[:-1], bounds[1:]):
        lab = int(labels[s])
        if lab not in LABELS:
            continue
        for start in range(s, e - win + 1, step):
            X.append(sig[start:start + win])
            y.append(lab - 1)                    # 1..4 -> 0..3
    if not X:
        return np.empty((0, win, sig.shape[1]), np.float32), np.empty(0, np.int64)
    return np.stack(X), np.asarray(y, np.int64)


def build_dataset(wesad_dir=WESAD_DIR, out_file=PROCESSED_FILE, subjects=SUBJECTS):
    """Process every available subject and save X, y, subjects to one .npz."""
    Xs, ys, subs = [], [], []
    for s in subjects:
        if not (wesad_dir / s / f"{s}.pkl").exists():
            print(f"  {s}: not found, skipping")
            continue
        sig, labels = preprocess_subject(*load_subject(s, wesad_dir))
        X, y = make_windows(sig, labels)
        counts = np.bincount(y, minlength=len(LABELS))
        print(f"  {s}: {len(y):4d} windows  " +
              "  ".join(f"{n[:4]}={c}" for n, c in zip(LABELS.values(), counts)))
        Xs.append(X.astype(np.float16))          # halves disk/RAM, plenty of precision for z-scores
        ys.append(y)
        subs += [s] * len(y)
    if not Xs:
        raise FileNotFoundError(f"No WESAD subjects found in {wesad_dir}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    np.savez(out_file, X=X, y=y, subjects=np.asarray(subs),
             meta=json.dumps({"fs": FS, "window_s": WINDOW_S, "step_s": STEP_S,
                              "channels": CHANNELS}))
    print(f"Saved {X.shape} -> {out_file}")
    return out_file


def load_dataset(task="4class", path=PROCESSED_FILE):
    """Load the processed windows and remap labels for the chosen task."""
    d = np.load(path)
    X, y, subjects = d["X"], d["y"], d["subjects"]
    mapping = TASKS[task]["map"]
    keep = np.isin(y, list(mapping))
    y = np.vectorize(mapping.get)(y[keep]).astype(np.int64)
    return X[keep], y, subjects[keep], TASKS[task]["classes"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wesad-dir", type=Path, default=WESAD_DIR)
    ap.add_argument("--out", type=Path, default=PROCESSED_FILE)
    args = ap.parse_args()
    build_dataset(args.wesad_dir, args.out)
