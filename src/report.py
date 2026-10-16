"""Collect every outputs/<run>/results.json into summary.csv and the README figures.

    python -m src.report
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR

FIG_DIR = OUTPUT_DIR / "figures"

# Runs shown in the figures: (results folder, label). One colour per model family,
# taken in fixed order from a colour-blind-validated categorical palette.
MODELS = [
    ("rf", "Random Forest (features)", "#2a78d6"),
    ("cnnlstm", "CNN-LSTM (raw signals)", "#eb6834"),
    ("hybrid", "Hybrid CNN-LSTM + features", "#1baf7a"),
]
TASKS = [("binary", "Stress vs. non-stress"), ("4class", "4 affective states")]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_results():
    runs = {p.parent.name: json.loads(p.read_text()) for p in sorted(OUTPUT_DIR.glob("*/results.json"))}
    return {k: r for k, r in runs.items() if "task" in r}     # skips the archived v1 format


def write_summary(results):
    rows = []
    for run, r in results.items():
        rows.append({"run": run, "model": r["model"], "task": r["task"], "eval": r["eval"],
                     "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                     "subject_f1_mean": r.get("per_subject_macro_f1_mean"),
                     "subject_f1_std": r.get("per_subject_macro_f1_std"),
                     "val_f1_mean": r.get("val_macro_f1_mean"),
                     **{f"f1_{k}": v for k, v in r["per_class_f1"].items()}})
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(OUTPUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def headline_run(results, model, task):
    """The final run for a model/task: the `final` tag if present, else the untagged LOSO run."""
    for name in (f"{model}_{task}_loso_final", f"{model}_{task}_loso"):
        if name in results:
            return results[name]
    return None


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_model_comparison(results, path):
    present = [m for m in MODELS if any(headline_run(results, m[0], t) for t, _ in TASKS)]
    fig, axes = plt.subplots(1, len(TASKS), figsize=(11, 0.6 * len(present) + 1.6), sharey=True)
    for ax, (task, title) in zip(axes, TASKS):
        for i, (model, label, color) in enumerate(present):
            r = headline_run(results, model, task)
            if r is None:
                continue
            y = len(present) - 1 - i
            ax.barh(y, r["macro_f1"], height=0.6, color=color)
            ax.text(r["macro_f1"] + 0.01, y, f"{r['macro_f1']:.3f}", va="center", color=TEXT, fontsize=10)
        ax.set_xlim(0, 1.08)
        ax.set_title(title, color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("Macro F1 (leave-one-subject-out)", color=MUTED, fontsize=9)
        style(ax)
    axes[0].set_yticks(range(len(present)), [m[1] for m in reversed(present)], color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_per_subject(results, task, path):
    present = [(m, l, c, headline_run(results, m, task)) for m, l, c in MODELS]
    present = [p for p in present if p[3] is not None]
    subjects = [f["fold"] for f in present[0][3]["folds"]]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    offsets = np.linspace(-0.22, 0.22, len(present))
    for (model, label, color, r), dx in zip(present, offsets):
        f1 = [f["macro_f1"] for f in r["folds"]]
        ax.scatter(np.arange(len(f1)) + dx, f1, s=36, color=color, edgecolor="white", linewidth=1,
                   label=f"{label}  (mean {np.mean(f1):.2f})", zorder=3)
    ax.set_xticks(range(len(subjects)), subjects, color=TEXT)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Macro F1 on held-out subject", color=MUTED, fontsize=9)
    ax.set_title(f"Per-subject generalisation, {dict(TASKS)[task].lower()}: each point is one unseen person",
                 color=TEXT, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    results = load_results()
    write_summary(results)
    FIG_DIR.mkdir(exist_ok=True)
    plot_model_comparison(results, FIG_DIR / "model_comparison.png")
    for task, _ in TASKS:
        plot_per_subject(results, task, FIG_DIR / f"per_subject_{task}.png")
    print(f"{len(results)} runs -> {OUTPUT_DIR / 'summary.csv'}, figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
