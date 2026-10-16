"""Hand-crafted physiological features for the classical (Random Forest) baseline."""
import numpy as np
from scipy.signal import find_peaks

from .config import CHANNELS, FS


def _stats(x):
    t = np.arange(len(x))
    slope = np.polyfit(t, x, 1)[0] * FS * 60          # change per minute
    return [x.mean(), x.std(), x.min(), x.max(),
            np.percentile(x, 10), np.percentile(x, 90), slope]


def _ecg_features(ecg):
    # R-peaks: at least 0.33 s apart (<= 180 bpm)
    peaks, _ = find_peaks(ecg, distance=int(0.33 * FS), prominence=1.0)
    if len(peaks) < 3:
        return [0.0, 0.0, 0.0]
    rr = np.diff(peaks) / FS
    return [60 / rr.mean(), rr.std(), np.sqrt(np.mean(np.diff(rr) ** 2))]   # HR, SDNN, RMSSD


def _eda_features(eda):
    # Skin-conductance responses: count and mean amplitude
    peaks, props = find_peaks(eda, prominence=0.05, distance=FS)
    amp = props["prominences"].mean() if len(peaks) else 0.0
    return [len(peaks), amp]


def _resp_features(resp):
    # Dominant breathing frequency (0.1-0.7 Hz) -> breaths per minute
    spec = np.abs(np.fft.rfft(resp - resp.mean()))
    freqs = np.fft.rfftfreq(len(resp), 1 / FS)
    band = (freqs >= 0.1) & (freqs <= 0.7)
    return [freqs[band][spec[band].argmax()] * 60]


def _emg_features(emg):
    return [np.sqrt(np.mean(emg ** 2))]


EXTRA = {"ECG": _ecg_features, "EDA": _eda_features,
         "Resp": _resp_features, "EMG": _emg_features}


def window_features(window):
    """window: (T, C) array -> 1-D feature vector."""
    feats = []
    for i, ch in enumerate(CHANNELS):
        x = window[:, i].astype(np.float64)
        feats += _stats(x)
        if ch in EXTRA:
            feats += EXTRA[ch](x)
    return np.asarray(feats, np.float32)


def extract_features(X):
    return np.stack([window_features(w) for w in X])
