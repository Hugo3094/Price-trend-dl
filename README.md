# Price Trend Prediction — MLP vs LSTM vs CNN

> **Research question:** To what extent does the choice of data representation (tabular, sequential, visual) and its associated architecture (MLP, RNN/LSTM/GRU, CNN) influence the ability to predict stock returns?

An implementation and extension of the paper **"(Re-)Imag(in)ing Price Trends"** (Jiang, Kelly & Xiu, *Journal of Finance*, 2023).

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Data Representations](#data-representations)
- [Analysis Dimensions](#analysis-dimensions)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dependencies](#dependencies)
- [Authors](#authors)
- [Reference](#reference)

---

## Overview

This project compares three deep learning architectures applied to stock return prediction, each operating on a different representation of the same OHLCV (Open, High, Low, Close, Volume) price data:

- **MLP** on tabular (flattened) data
- **LSTM / GRU** on sequential time-series data
- **CNN** on OHLC candlestick images

The central hypothesis, drawn from Jiang et al. (2023), is that encoding price histories as images and processing them with convolutional networks captures spatial patterns in price, volatility, and volume simultaneously — outperforming classical time-series approaches.

---

## Key Results

Results obtained on a subset of S&P 500 tickers (2000–2022), 20-day window, 5-day prediction horizon.

| Model    | Representation        | Accuracy | F1    | AUC       | Brier Score ↓ |
|----------|-----------------------|----------|-------|-----------|---------------|
| **GRU**  | Sequential            | **0.653**| 0.662 | **0.701** | **0.220**     |
| **LSTM** | Sequential            | **0.655**| 0.659 | **0.701** | **0.220**     |
| MLP      | Tabular (image scale) | 0.513    | **0.672** | 0.688 | 0.248         |
| CNN      | OHLC Images           | 0.528    | 0.670 | 0.493     | 0.325         |

> ↓ Lower Brier score is better. Bold values indicate best performance per metric.

The sequential models (GRU and LSTM) dominate across accuracy, AUC, and Brier score. The MLP achieves the highest F1. Notably, the CNN — despite its visual representation — underperforms on this dataset, with an AUC close to random (0.493), suggesting the image-based approach may require larger data or longer training to generalize.

---

## Project Structure

```
price-trend-dl/
├── configs/             # Configuration files (hyperparameters, model settings)
├── scripts/             # Utility and runner scripts
├── src/                 # Core source code
│   ├── data/
│   │   └── fetch_data.py        # Yahoo Finance download, normalization, label generation
│   ├── imaging/
│   │   └── ohlc_chart.py        # OHLC image generation (faithful to the paper's spec)
│   ├── models/
│   │   ├── mlp.py               # MLP with BatchNorm + Dropout
│   │   ├── lstm.py              # LSTM / GRU / Attention-LSTM
│   │   └── cnn.py               # 2D CNN + Grad-CAM
│   ├── training/
│   │   └── train.py             # Unified training pipeline (early stopping, scheduler)
│   └── evaluation/
│       └── metrics.py           # Sharpe ratio, decile analysis, model comparison
├── tests/               # Unit and integration tests
├── main.py              # Entry point
├── pyproject.toml       # Project metadata and dependencies
└── uv.lock              # Dependency lockfile
```

---

## Data Representations

### 1. Tabular — MLP

OHLCV data is flattened into a fixed-size vector. **Image-scale normalization** (max High = 1, min Low = 0 over the window) is applied — this consistently outperforms cumulative-return normalization as shown in the original paper.

### 2. Sequential — LSTM / GRU

Data is treated as a time series of shape `(window, n_features)`. The model learns temporal dependencies through hidden states. Three variants are available: standard LSTM, GRU, and Attention-LSTM.

### 3. Visual — OHLC Images (CNN)

Each price window is encoded as a grayscale image following the exact specification from Jiang et al. (2023):

- **Black background**, white objects
- **3 pixels per trading day**: high-low bar | open tick | close tick
- **Volume bars** occupy the bottom 1/5 of the image
- **Moving average** drawn pixel-by-pixel using Bresenham's line algorithm
- **Normalization**: max High maps to the image top, min Low maps to the bottom

| Window   | Image Dimensions |
|----------|-----------------|
| 5 days   | 32 × 15 px      |
| 20 days  | 64 × 60 px      |
| 60 days  | 96 × 180 px     |

---

## Discussion: Gap with the Original Paper

Our empirical results diverge from Jiang et al. (2023) in a notable way: **the CNN does not outperform sequential models** on our dataset, contrary to the paper's central finding.

| Metric   | Paper's claim (CNN) | Our CNN | Our best model  |
|----------|---------------------|---------|-----------------|
| Accuracy | ~56%                | 52.8%   | LSTM (65.5%)    |
| AUC      | ~0.59               | 0.493   | GRU/LSTM (0.701)|

Several factors likely explain this gap:

**Data scale.** The original paper trains on the full US stock universe over several decades. Our experiments use a subset of S&P 500 tickers (2000–2022), which may be insufficient for the CNN to learn robust visual patterns from images as small as 64×60 px.

**Training budget.** CNNs require significantly more data and epochs to converge compared to LSTMs on this type of task. Our unified training pipeline applies the same budget across all architectures, which may disadvantage the CNN.

**Image resolution.** At 64×60 px for a 20-day window, the visual signal is very compressed. The CNN's near-random AUC (0.493) suggests it may be underfitting rather than learning meaningful chart patterns.

**Takeaway.** On a constrained dataset, sequential architectures (GRU, LSTM) are more data-efficient than CNNs for price trend prediction. Reproducing the paper's CNN advantage likely requires the full-scale data setup described in the original work.

---

## Analysis Dimensions

### Representation Impact

Direct comparison of MLP, LSTM, and CNN on **identical data** with **identical normalization**. Key finding: image-scale normalization is the dominant factor; the CNN adds a layer of spatial non-linearity that jointly captures relationships between price range, volatility, and volume.

### Interpretability — Grad-CAM

Gradient-weighted class activation maps highlight which regions of the OHLC image drive CNN predictions:

- Recent days (t-1, t-2) carry the most weight
- The Close position relative to the High-Low range is a strong directional signal
- High-volume bars amplify directional signals

### Robustness Analysis

- Time windows: 5-day, 20-day, 60-day
- Volume inclusion vs. exclusion
- Image noise robustness (MaxPooling)

---

## Installation

**Using pip:**

```bash
git clone https://github.com/Hugo3094/Price-trend-dl.git
cd Price-trend-dl
pip install -e .
```

**Using uv (recommended):**

```bash
git clone https://github.com/Hugo3094/Price-trend-dl.git
cd Price-trend-dl
uv sync
```

Requires **Python >= 3.10**.

---

## Quick Start

```python
from src.data.fetch_data import download_ohlcv, make_multi_stock_dataset
from src.imaging.ohlc_chart import make_image_dataset
from src.models.cnn import build_cnn
from src.training.train import Trainer, set_seed

set_seed(42)

# Download data
raw = download_ohlcv(start="2010-01-01", end="2022-12-31")

# Build image dataset
img_ds = make_image_dataset(raw, window=20, horizon=5)

# Build model
cnn = build_cnn(window=20)

# Train
trainer = Trainer(cnn, model_type="cnn", save_dir="checkpoints/cnn")
history = trainer.fit(
    img_ds["X_train"], img_ds["y_train"],
    img_ds["X_val"],   img_ds["y_val"],
    epochs=50,
    batch_size=64,
)
```

---

## Dependencies

| Package        | Version   | Purpose                          |
|----------------|-----------|----------------------------------|
| `torch`        | ≥ 2.1.0   | Deep learning framework          |
| `torchvision`  | ≥ 0.16.0  | Image transforms                 |
| `yfinance`     | ≥ 0.2.36  | Market data download             |
| `numpy`        | ≥ 1.24    | Numerical computing              |
| `pandas`       | ≥ 2.0     | Data manipulation                |
| `scikit-learn` | ≥ 1.3     | Metrics and preprocessing        |
| `pillow`       | ≥ 10.0    | Image generation                 |
| `matplotlib`   | ≥ 3.7     | Visualization                    |
| `cvxpy`        | ≥ 1.4     | Portfolio optimization           |
| `pytest`       | ≥ 9.0.3   | Testing                          |

---

## Authors

**Mathieu Lang · Mateo Molinaro · Hugo Lecointre**

---

## Reference

```bibtex
@article{jiang2023reimagining,
  title   = {(Re-)Imag(in)ing Price Trends},
  author  = {Jiang, Jingwen and Kelly, Bryan and Xiu, Dacheng},
  journal = {The Journal of Finance},
  volume  = {78},
  number  = {6},
  pages   = {3193--3249},
  year    = {2023},
  doi     = {10.1111/jofi.13268}
}
```
