"""Neural network architectures."""
import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel, pool):
        super().__init__(
            nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )


class CNNLSTM(nn.Module):
    """1-D CNN front-end + BiLSTM + attention pooling.

    The CNN learns local waveform features (heartbeats, breaths, SCRs) and
    shortens the sequence 64x (6000 -> ~93 steps), so the LSTM only has to model
    slow dynamics. The v1 model fed 4200 raw steps straight into an LSTM, which
    cannot carry gradients that far and ended up near chance.
    """

    def __init__(self, n_channels, n_classes, hidden=64, dropout=0.3, width=32):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(n_channels, width, 7, 4),
            ConvBlock(width, 2 * width, 5, 4),
            ConvBlock(2 * width, 4 * width, 5, 4),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(4 * width, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def embed(self, x):                          # x: (batch, time, channels)
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        w = torch.softmax(self.attn(h), dim=1)   # attention weights over time
        return (w * h).sum(dim=1)

    def forward(self, x):
        return self.head(self.embed(x))


class HybridNet(nn.Module):
    """CNN-LSTM signal embedding fused with hand-crafted physiological features.

    The raw-signal branch learns waveform patterns; the feature branch gets the
    domain knowledge (HR, HRV, SCR count, breathing rate) that a small dataset
    is too small to rediscover. Both are concatenated before the classifier.
    """

    def __init__(self, n_channels, n_classes, n_features, hidden=64, dropout=0.3, width=32):
        super().__init__()
        self.signal = CNNLSTM(n_channels, n_classes, hidden, dropout, width)
        self.features = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden + 64, n_classes))

    def forward(self, x, f):
        return self.head(torch.cat([self.signal.embed(x), self.features(f)], dim=1))


class StressLSTM(nn.Module):
    """The original v1 model (kept for comparison)."""

    def __init__(self, n_channels, n_classes, hidden=128, num_layers=2,
                 lstm_dropout=0.3, fc_dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, num_layers, batch_first=True,
                            dropout=lstm_dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                        nn.Dropout(fc_dropout), nn.Linear(64, n_classes))

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.classifier(self.norm(h_n[-1]))


MODELS = {"cnnlstm": CNNLSTM, "hybrid": HybridNet, "lstm": StressLSTM}
