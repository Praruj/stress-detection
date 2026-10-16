"""Train and evaluate stress classifiers on WESAD.

Examples
--------
    python -m src.train --model rf      --task 4class --eval loso
    python -m src.train --model cnnlstm --task 4class --eval loso
    python -m src.train --model cnnlstm --task binary --eval split
    python -m src.train --model hybrid  --task 4class --eval loso
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import FS, MODEL_DIR, OUTPUT_DIR, PROCESSED_FILE, TASKS
from .data import load_dataset
from .features import extract_features
from .models import MODELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed split kept identical to the v1 notebook so results are comparable
SPLIT = {"val": ["S3", "S6"], "test": ["S2", "S15"]}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_folds(subjects, mode, n_val=1):
    """Yield (name, train_subjects, val_subjects, test_subjects)."""
    present = sorted(set(subjects), key=lambda s: int(s[1:]))
    if mode == "split":
        test = [s for s in SPLIT["test"] if s in present]
        val = [s for s in SPLIT["val"] if s in present]
        yield "split", [s for s in present if s not in test + val], val, test
        return
    # Leave-one-subject-out; the next n_val subjects (cyclically) are used for early stopping
    for i, test_s in enumerate(present):
        val = [present[(i + k) % len(present)] for k in range(1, n_val + 1)]
        yield test_s, [s for s in present if s != test_s and s not in val], val, [test_s]


# ----------------------------------------------------------------- deep models

def batches(X, y, idx, batch_size, shuffle, args=None, F=None):
    """Yield (x, features, y) batches. With `args`, apply training-time augmentation.

    `F` is an optional (n_windows, n_features) tensor for the hybrid model; features
    stay un-augmented because they summarise the whole 60 s window.
    """
    if shuffle:
        idx = np.random.permutation(idx)
    for i in range(0, len(idx), batch_size):
        b = np.sort(idx[i:i + batch_size])
        xb = X[torch.from_numpy(b).to(X.device)].float() if torch.is_tensor(X) \
            else torch.from_numpy(X[b].astype(np.float32)).to(DEVICE)
        if args is not None:
            # random per-channel gain + sensor noise
            xb = xb * (1 + 0.1 * torch.randn(len(b), 1, xb.shape[2], device=DEVICE))
            xb = xb + 0.05 * torch.randn_like(xb)
            if args.crop:     # random sub-window; attention pooling handles any length
                n = int(args.crop * FS)
                start = np.random.randint(0, xb.shape[1] - n + 1)
                xb = xb[:, start:start + n]
            if args.chan_drop:  # blank whole sensor channels so no single signal is relied on
                keep = torch.rand(len(b), 1, xb.shape[2], device=DEVICE) >= args.chan_drop
                xb = xb * keep
        fb = F[torch.from_numpy(b).to(F.device)] if F is not None else None
        yield xb, fb, torch.from_numpy(y[b]).to(DEVICE)


def forward(model, xb, fb):
    return model(xb) if fb is None else model(xb, fb)


def predict_proba(model, X, idx, F=None, batch_size=64):
    model.eval()
    out = []
    with torch.no_grad():
        for xb, fb, _ in batches(X, np.zeros(len(X), np.int64), idx, batch_size, False, F=F):
            out.append(torch.softmax(forward(model, xb, fb), 1).cpu().numpy())
    return np.concatenate(out)


def predict(model, X, idx, F=None):
    return predict_proba(model, X, idx, F).argmax(1)


def build_model(args, n_channels, n_classes, n_features=0):
    if args.model == "hybrid":
        return MODELS["hybrid"](n_channels, n_classes, n_features, dropout=args.dropout, width=args.width)
    if args.model == "cnnlstm":
        return MODELS["cnnlstm"](n_channels, n_classes, dropout=args.dropout, width=args.width)
    return MODELS[args.model](n_channels, n_classes)


def train_deep(args, X, y, tr, va, n_classes, F=None):
    model = build_model(args, X.shape[2], n_classes, 0 if F is None else F.shape[1]).to(DEVICE)
    counts = np.bincount(y[tr], minlength=n_classes).astype(float)
    w = (1 / np.maximum(counts, 1)); w = w / w.sum() * n_classes
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE),
                                    label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = {k: [] for k in ("train_loss", "train_f1", "val_f1")}
    best_f1, best_state, best_epoch, bad = -1, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, preds, trues = [], [], []
        for xb, fb, yb in batches(X, y, tr, args.batch_size, True, args=args, F=F):
            logits = forward(model, xb, fb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item() * len(yb))
            preds.append(logits.argmax(1).cpu().numpy()); trues.append(yb.cpu().numpy())
        sched.step()
        tr_f1 = f1_score(np.concatenate(trues), np.concatenate(preds), average="macro")
        va_f1 = f1_score(y[va], predict(model, X, va, F), average="macro")
        history["train_loss"].append(sum(losses) / len(tr))
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)
        if args.verbose:
            print(f"    ep {epoch:3d}  loss {history['train_loss'][-1]:.4f}  "
                  f"train F1 {tr_f1:.3f}  val F1 {va_f1:.3f}")
        if va_f1 > best_f1:
            best_f1, best_epoch, bad = va_f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_f1


# ------------------------------------------------------------------ RF baseline

def get_features(X, cache):
    if cache.exists():
        F = np.load(cache)
        if len(F) == len(X):
            return F
    print("Extracting hand-crafted features...")
    F = extract_features(X)
    np.save(cache, F)
    return F


# ---------------------------------------------------------------------- output

def plot_confusion(cm, classes, title, path):
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    fig, axes = plt.subplots(1, 2, figsize=(4 * len(classes) + 3, 0.9 * len(classes) + 2.5))
    fig.suptitle(title, fontweight="bold")
    for ax, (data, fmt, t) in zip(axes, [(cm, "d", "Counts"), (cm_norm, ".1f", "Row-normalised (%)")]):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", xticklabels=classes,
                    yticklabels=classes, ax=ax, linewidths=0.5, cbar=False)
        ax.set_title(t); ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_history(history, best_epoch, path):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, history["train_loss"], color="steelblue"); axes[0].set_title("Train loss")
    axes[1].plot(ep, history["train_f1"], label="Train", color="steelblue")
    axes[1].plot(ep, history["val_f1"], label="Val", color="crimson"); axes[1].set_title("Macro F1")
    for ax in axes:
        ax.axvline(best_epoch, color="gray", ls="--", alpha=0.6); ax.grid(alpha=0.3)
        ax.set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["cnnlstm", "hybrid", "lstm", "rf"], default="cnnlstm")
    ap.add_argument("--task", choices=list(TASKS), default="4class")
    ap.add_argument("--eval", choices=["loso", "split"], default="loso")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", type=Path, default=PROCESSED_FILE)
    ap.add_argument("--verbose", action="store_true", help="print every epoch")
    # regularisation / generalisation knobs (CNN-LSTM)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--width", type=int, default=32, help="channels of the first conv layer")
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--crop", type=float, default=0, help="random training crop length in seconds (0 = off)")
    ap.add_argument("--chan-drop", type=float, default=0.0, help="probability of blanking a channel")
    ap.add_argument("--n-val", type=int, default=1, help="LOSO: subjects held out for early stopping")
    ap.add_argument("--ensemble", type=int, default=1, help="average this many models trained with different seeds")
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    X, y, subjects, classes = load_dataset(args.task, args.data)
    n_classes = len(classes)
    run = f"{args.model}_{args.task}_{args.eval}" + (f"_{args.tag}" if args.tag else "")
    out_dir = OUTPUT_DIR / run
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run {run} | device {DEVICE} | X {X.shape} | classes {classes}")

    F = get_features(X, args.data.with_name(args.data.stem + f"_{args.task}_features.npy")) \
        if args.model in ("rf", "hybrid") else None
    # On GPU keep the whole (float16) dataset in device memory: ~150 MB, avoids per-batch copies
    Xd = torch.from_numpy(X).to(DEVICE) if DEVICE.type == "cuda" and args.model != "rf" else X

    y_true, y_pred, per_subject = [], [], []
    t0 = time.time()
    for name, tr_s, va_s, te_s in make_folds(subjects, args.eval, args.n_val):
        tr = np.flatnonzero(np.isin(subjects, tr_s))
        va = np.flatnonzero(np.isin(subjects, va_s))
        te = np.flatnonzero(np.isin(subjects, te_s))
        val_f1 = None
        if args.model == "rf":
            clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                         n_jobs=-1, random_state=args.seed)
            clf.fit(F[np.r_[tr, va]], y[np.r_[tr, va]])
            pred, best_epoch = clf.predict(F[te]), None
        else:
            Fd = None
            if F is not None:   # hybrid: standardise features with this fold's training subjects only
                Fz = np.nan_to_num(F)
                mu, sd = Fz[tr].mean(0), Fz[tr].std(0) + 1e-6
                Fd = torch.tensor((Fz - mu) / sd, dtype=torch.float32, device=DEVICE)
            probs, epochs, val_f1s = 0, [], []
            for k in range(args.ensemble):
                set_seed(args.seed + 1000 * k)
                model, history, best_epoch, best_val = train_deep(args, Xd, y, tr, va, n_classes, Fd)
                probs = probs + predict_proba(model, Xd, te, Fd)
                epochs.append(best_epoch); val_f1s.append(best_val)
            pred, best_epoch, val_f1 = probs.argmax(1), int(np.median(epochs)), float(np.mean(val_f1s))
            if args.eval == "split":
                plot_history(history, best_epoch, out_dir / "training_curves.png")
                MODEL_DIR.mkdir(exist_ok=True)
                torch.save(model.state_dict(), MODEL_DIR / f"{args.model}_{args.task}.pt")

        acc = accuracy_score(y[te], pred)
        f1 = f1_score(y[te], pred, average="macro", labels=range(n_classes), zero_division=0)
        per_subject.append({"fold": name, "test": te_s, "n": int(len(te)), "accuracy": round(acc, 4),
                            "macro_f1": round(f1, 4), "best_epoch": best_epoch,
                            "val_f1": None if val_f1 is None else round(val_f1, 4)})
        y_true.append(y[te]); y_pred.append(pred)
        print(f"  fold {name:>5}: acc {acc:.3f}  macro-F1 {f1:.3f}"
              + (f"  (best ep {best_epoch}, val F1 {val_f1:.3f})" if best_epoch else "")
              + f"  [{time.time() - t0:.0f}s]")

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=range(n_classes), zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, labels=range(n_classes), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    subj_f1 = [p["macro_f1"] for p in per_subject]
    subj_acc = [p["accuracy"] for p in per_subject]

    results = {
        "run": run, "model": args.model, "task": args.task, "eval": args.eval,
        "classes": classes, "n_windows": int(len(y)), "input_shape": list(X.shape[1:]),
        "accuracy": round(float(acc), 4), "macro_f1": round(float(macro_f1), 4),
        "per_subject_accuracy_mean": round(float(np.mean(subj_acc)), 4),
        "per_subject_accuracy_std": round(float(np.std(subj_acc)), 4),
        "per_subject_macro_f1_mean": round(float(np.mean(subj_f1)), 4),
        "per_subject_macro_f1_std": round(float(np.std(subj_f1)), 4),
        # mean early-stopping validation F1: used to pick hyper-parameters without looking at test subjects
        "val_macro_f1_mean": round(float(np.mean([p["val_f1"] for p in per_subject])), 4)
                             if args.model != "rf" else None,
        "per_class_f1": {c: round(float(f), 4) for c, f in zip(classes, per_class)},
        "confusion_matrix": cm.tolist(), "folds": per_subject,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    plot_confusion(cm, classes, f"{run}  (acc {acc*100:.1f}%, macro-F1 {macro_f1:.3f})",
                   out_dir / "confusion_matrix.png")

    print("\n" + classification_report(y_true, y_pred, labels=range(n_classes),
                                       target_names=classes, zero_division=0))
    print(f"Pooled accuracy {acc*100:.1f}% | macro-F1 {macro_f1:.3f} | "
          f"per-subject F1 {np.mean(subj_f1):.3f} +/- {np.std(subj_f1):.3f}")
    print(f"Saved -> {out_dir}")
    return results


if __name__ == "__main__":
    main()
