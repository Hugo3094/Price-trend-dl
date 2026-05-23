"""
mlp.py
------
Multi-Layer Perceptron on tabular features (flat representation).
Serves as a simple baseline in the architecture comparison.

Input  : (batch, window * n_features)  — flattened sequence
Output : (batch, 2)                     — up/down logits
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """
    MLP with BatchNorm and Dropout between each layer.

    Architecture:
        Input -> [Linear -> BN -> ReLU -> Dropout] x n_layers -> Linear(2)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 128, 64],
        dropout: float = 0.3,
        n_classes: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        input_dim   : window * n_features (e.g. 20 * 6 = 120)
        hidden_dims : list of hidden layer dimensions
        dropout     : dropout rate
        n_classes   : 2 (binary up/down classification)
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, window, n_features) or (batch, window * n_features)
        """
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.net(x)


def build_mlp(
    window: int,
    n_features: int,
    hidden_dims: list[int] = [256, 128, 64],
    dropout: float = 0.3,
) -> MLP:
    """Factory function."""
    return MLP(input_dim=window * n_features, hidden_dims=hidden_dims, dropout=dropout)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    model = build_mlp(window=20, n_features=6)
    logger.info("%s", model)
    x = torch.randn(32, 20, 6)
    out = model(x)
    logger.info("Output shape : %s", out.shape)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters   : %d", n_params)