"""
lstm.py
-------
Sequential models (LSTM / GRU) for return prediction.
Data representation: raw or normalised OHLCV time series.

Input  : (batch, window, n_features)
Output : (batch, 2)  — up/down logits
"""

import logging
import torch
import torch.nn as nn
from typing import Literal

logger = logging.getLogger(__name__)


class RNNClassifier(nn.Module):
    """
    LSTM or GRU classifier with an MLP classification head.

    Architecture:
        Input -> RNN (LSTM or GRU) -> last hidden state -> MLP -> logits
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        rnn_type: Literal["LSTM", "GRU"] = "LSTM",
        dropout: float = 0.3,
        bidirectional: bool = False,
        n_classes: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        input_size    : number of features per time step (e.g. 6: O/H/L/C/V/MA)
        hidden_size   : hidden state dimension
        num_layers    : number of stacked RNN layers
        rnn_type      : "LSTM" or "GRU"
        dropout       : dropout between RNN layers (active if num_layers > 1)
        bidirectional : bidirectional RNN
        n_classes     : 2
        """
        super().__init__()

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_in = hidden_size * self.directions
        self.classifier = nn.Sequential(
            nn.LayerNorm(fc_in),
            nn.Dropout(dropout),
            nn.Linear(fc_in, fc_in // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_in // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)
        """
        if self.rnn_type == "LSTM":
            out, (h_n, _) = self.rnn(x)
        else:
            out, h_n = self.rnn(x)

        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]

        return self.classifier(h_last)


class AttentionLSTM(nn.Module):
    """
    LSTM with a temporal attention mechanism.
    Allows the model to weight the importance of each time step,
    improving interpretability (complementary axis to CNN Grad-CAM).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_classes: int = 2,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        logits       : (batch, n_classes)
        attn_weights : (batch, seq_len)
        """
        out, _ = self.lstm(x)

        scores = self.attention(out)
        weights = torch.softmax(scores, dim=1)
        context = (weights * out).sum(dim=1)

        return self.classifier(context), weights.squeeze(-1)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def build_lstm(
    n_features: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    bidirectional: bool = False,
) -> RNNClassifier:
    return RNNClassifier(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type="LSTM",
        dropout=dropout,
        bidirectional=bidirectional,
    )


def build_gru(
    n_features: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> RNNClassifier:
    return RNNClassifier(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type="GRU",
        dropout=dropout,
    )


def build_attention_lstm(
    n_features: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> AttentionLSTM:
    return AttentionLSTM(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    batch, seq_len, n_feat = 32, 20, 6

    logger.info("=== LSTM ===")
    lstm = build_lstm(n_features=n_feat)
    x = torch.randn(batch, seq_len, n_feat)
    out = lstm(x)
    logger.info("Output : %s", out.shape)
    logger.info("Params : %d", sum(p.numel() for p in lstm.parameters() if p.requires_grad))

    logger.info("=== GRU ===")
    gru = build_gru(n_features=n_feat)
    out = gru(x)
    logger.info("Output : %s", out.shape)
    logger.info("Params : %d", sum(p.numel() for p in gru.parameters() if p.requires_grad))

    logger.info("=== Attention LSTM ===")
    attn_lstm = build_attention_lstm(n_features=n_feat)
    logits, weights = attn_lstm(x)
    logger.info("Logits  : %s", logits.shape)
    logger.info("Weights : %s", weights.shape)
    logger.info("Params  : %d", sum(p.numel() for p in attn_lstm.parameters() if p.requires_grad))