"""
cnn.py
------
2D CNN for return prediction from OHLC images.
Architecture faithful to Jiang, Kelly & Xiu (2023).

Image dimensions:
  - 5  days : 32x15
  - 20 days : 64x60
  - 60 days : 96x180

Architecture by window:
  - 5d  : 2 CNN blocks (64, 128  filters)
  - 20d : 3 CNN blocks (64, 128, 256 filters)
  - 60d : 4 CNN blocks (64, 128, 256, 512 filters)

Each block = Conv(5x3) -> BatchNorm -> LeakyReLU -> MaxPool(2x1)

Reference: Jiang, Kelly & Xiu (2023), Section II & Appendix
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

logger = logging.getLogger(__name__)


# Mirror of ohlc_chart.IMAGE_SPECS — dimensions defined in the paper (Section I)
_IMAGE_SPECS = {
    5:  {"height": 15,  "width": 32},
    20: {"height": 60,  "width": 64},
    60: {"height": 180, "width": 96},
}


# ---------------------------------------------------------------------------
# Basic CNN block
# ---------------------------------------------------------------------------
class CNNBlock(nn.Module):
    """
    Basic block: Conv2D -> BatchNorm -> LeakyReLU -> MaxPool.
    Filter 5x3 (height x width), MaxPool 2x1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (5, 3),
        pool_size: tuple[int, int] = (2, 1),
        dilation: tuple[int, int] = (1, 1),
        leaky_slope: float = 0.01,
    ) -> None:
        super().__init__()

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


# ---------------------------------------------------------------------------
# Main CNN
# ---------------------------------------------------------------------------
class PriceCNN(nn.Module):
    """
    2D CNN for OHLC images.

    The architecture is automatically selected based on the window size
    (2, 3, or 4 blocks), as specified in the paper.
    """

    CONFIGS = {
        5:  {"n_blocks": 2, "filter_base": 64, "dilation": (2, 1)},
        20: {"n_blocks": 3, "filter_base": 64, "dilation": (2, 1)},
        60: {"n_blocks": 4, "filter_base": 64, "dilation": (3, 1)},
    }

    def __init__(
        self,
        window: int = 20,
        image_height: int = 60,
        image_width: int = 64,
        in_channels: int = 1,
        dropout: float = 0.5,
        n_classes: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        window       : 5, 20, or 60
        image_height : image height in pixels
        image_width  : image width in pixels
        in_channels  : 1 (grayscale)
        dropout      : dropout on the FC layer
        n_classes    : 2 (up / down)
        """
        super().__init__()

        assert window in self.CONFIGS, f"window must be one of {list(self.CONFIGS.keys())}"
        cfg = self.CONFIGS[window]

        blocks = []
        in_ch = in_channels
        out_ch = cfg["filter_base"]
        dilation_h = cfg["dilation"][0]

        for i in range(cfg["n_blocks"]):
            # Dilation on first block only (sparse features on raw images)
            dil = (dilation_h, 1) if i == 0 else (1, 1)
            blocks.append(CNNBlock(in_ch, out_ch, dilation=dil))
            in_ch = out_ch
            out_ch = out_ch * 2

        self.cnn_blocks = nn.Sequential(*blocks)

        fc_in = self._get_fc_input_dim(image_height, image_width, in_channels, cfg["n_blocks"])

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_in, n_classes)

    def _get_fc_input_dim(self, h: int, w: int, in_channels: int, n_blocks: int) -> int:
        """Computes the FC input dimension by passing a dummy tensor."""
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            out = self.cnn_blocks(dummy)
            return int(out.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 1, H, W)

        Returns
        -------
        logits : (batch, 2)
        """
        feat = self.cnn_blocks(x)
        feat = self.flatten(feat)
        feat = self.dropout(feat)
        return self.fc(feat)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts feature maps before the FC layer (useful for Grad-CAM)."""
        return self.cnn_blocks(x)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    """
    Grad-CAM implementation for visualising the active regions
    in OHLC images that drive CNN predictions.

    Reference: Selvaraju et al. (2017) — Grad-CAM
    """

    def __init__(self, model: PriceCNN, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.target_layer = target_layer or list(model.cnn_blocks.children())[-1].conv
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        """
        Generates the Grad-CAM heatmap.

        Parameters
        ----------
        x         : (1, 1, H, W)
        class_idx : target class index (None = predicted class)

        Returns
        -------
        cam : (H, W)  — normalised heatmap [0, 1]
        """
        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_cnn(window: int = 20, dropout: float = 0.5) -> PriceCNN:
    """Builds the CNN with the correct image dimensions for the given window."""
    specs = _IMAGE_SPECS[window]
    return PriceCNN(
        window=window,
        image_height=specs["height"],
        image_width=specs["width"],
        in_channels=1,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for w in [5, 20, 60]:
        specs = _IMAGE_SPECS[w]
        H, W = specs["height"], specs["width"]

        model = build_cnn(window=w)
        x = torch.randn(4, 1, H, W)
        out = model(x)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Window=%2dd | Image %dx%d | Output %s | Params %10d", w, H, W, out.shape, n_params)

    logger.info("=== Grad-CAM test ===")
    model = build_cnn(window=20)
    gradcam = GradCAM(model)
    x = torch.randn(1, 1, 60, 64)
    cam = gradcam.generate(x)
    logger.info("CAM shape: %s, min=%.3f, max=%.3f", cam.shape, cam.min(), cam.max())