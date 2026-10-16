"""End-to-end tests on synthetic data: preprocessing -> windows -> every model trains and predicts."""
import json

import numpy as np
import pytest
import torch

from src.config import CHANNELS, FS, WINDOW_S
from src.data import build_dataset, load_dataset, make_windows
from src.features import window_features
from src.models import CNNLSTM, HybridNet
from src import train as train_mod
from tests.synthetic import write_fake_wesad


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("wesad")
    write_fake_wesad(root / "WESAD")
    return build_dataset(root / "WESAD", root / "processed.npz",
                         subjects=["S2", "S3", "S4", "S6", "S15"])


def test_windows_never_cross_label_boundaries():
    labels = np.r_[np.full(7000, 1), np.full(7000, 2)].astype(np.int8)
    sig = np.zeros((len(labels), len(CHANNELS)), np.float32)
    X, y = make_windows(sig, labels, window_s=30, step_s=10)
    assert X.shape[1:] == (30 * FS, len(CHANNELS))
    assert set(y) == {0, 1}                      # one window per 70 s block at 30 s / 10 s hop -> 5 each
    assert np.bincount(y).tolist() == [5, 5]


def test_dataset_shapes_and_normalisation(dataset):
    X, y, subjects, classes = load_dataset("4class", dataset)
    assert X.shape[1:] == (WINDOW_S * FS, len(CHANNELS))
    assert len(X) == len(y) == len(subjects)
    assert classes == ["Baseline", "Stress", "Amusement", "Meditation"]
    # per-subject z-scoring: every channel is roughly zero-mean within a subject
    s2 = X[subjects == "S2"].astype(np.float32)
    assert np.abs(s2.mean(axis=(0, 1))).max() < 0.5


@pytest.mark.parametrize("task,n", [("binary", 2), ("3class", 3)])
def test_task_label_mapping(dataset, task, n):
    _, y, _, classes = load_dataset(task, dataset)
    assert len(classes) == n and set(np.unique(y)) <= set(range(n))


def test_features_are_finite(dataset):
    X, *_ = load_dataset("4class", dataset)
    f = window_features(X[0])
    assert f.ndim == 1 and np.isfinite(f).all()


def test_models_accept_variable_length_input():
    x = torch.randn(2, 4500, len(CHANNELS))      # 45 s crop instead of 60 s
    assert CNNLSTM(len(CHANNELS), 4)(x).shape == (2, 4)
    assert HybridNet(len(CHANNELS), 4, n_features=10)(x, torch.randn(2, 10)).shape == (2, 4)


@pytest.mark.parametrize("model,extra", [
    ("rf", []),
    ("cnnlstm", ["--crop", "45", "--chan-drop", "0.1", "--label-smoothing", "0.1"]),
    ("hybrid", ["--ensemble", "2"]),
])
def test_training_end_to_end(dataset, tmp_path, monkeypatch, model, extra):
    monkeypatch.setattr(train_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(train_mod, "MODEL_DIR", tmp_path / "models")
    res = train_mod.main(["--model", model, "--task", "binary", "--eval", "split",
                          "--epochs", "2", "--data", str(dataset)] + extra)
    assert 0 <= res["accuracy"] <= 1
    saved = json.loads((tmp_path / res["run"] / "results.json").read_text())
    assert saved["confusion_matrix"] == res["confusion_matrix"]
