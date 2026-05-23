"""
train.py
--------
Unified training pipeline for MLP, LSTM/GRU, and CNN.
Handles: early stopping, scheduler, logging, best-model checkpointing.
"""

import logging
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from reimagining_trends.utils.cache import (
    load_json, save_json, load_pickle, fingerprints_match,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = True,
    model_type: str = "mlp",
) -> DataLoader:
    """Creates a DataLoader from numpy arrays."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    # CNN: (N, H, W, 1) -> (N, 1, H, W)
    if model_type == "cnn" and X_t.dim() == 4:
        X_t = X_t.permute(0, 3, 1, 2)

    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """
    Trains any PyTorch model with:
      - Cross-entropy loss
      - Adam optimizer
      - CosineAnnealingLR scheduler
      - Early stopping on validation loss
      - Best-model checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        save_dir: str = "checkpoints",
        device: Optional[torch.device] = None,
        model_name: str = "",
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.model_name = model_name or model_type.upper()
        self.save_dir = save_dir
        self.device = device or get_device()
        self.model.to(self.device)
        os.makedirs(save_dir, exist_ok=True)
        self.fingerprint: Optional[dict] = None  # set by fit() after training or cache hit

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 5,
        verbose: bool = True,
        fingerprint: Optional[dict] = None,
    ) -> dict:
        """
        Runs training.

        Returns
        -------
        history : dict with train_loss, val_loss, train_acc, val_acc per epoch
        """
        fp_path   = os.path.join(self.save_dir, "fingerprint.json")
        ckpt_path = os.path.join(self.save_dir, "best_model.pt")
        hist_path = os.path.join(self.save_dir, f"{self.model_type}_history.json")

        if fingerprint is not None:
            cached_fp = load_json(fp_path)
            if fingerprints_match(fingerprint, cached_fp) and os.path.exists(ckpt_path):
                logger.info("[%s] Cache hit — loading checkpoint, skipping training.", self.model_name)
                self.load_best()
                self.fingerprint = fingerprint
                if os.path.exists(hist_path):
                    with open(hist_path, "r") as f:
                        return json.load(f)
                return {}

        train_loader = make_dataloader(X_train, y_train, batch_size, shuffle=True, model_type=self.model_type)
        val_loader = make_dataloader(X_val, y_val, batch_size, shuffle=False, model_type=self.model_type)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)
        logger.info(
            "Training %s | %d train / %d val samples | %d epochs | batch=%d | device=%s",
            self.model_name, n_train, n_val, epochs, batch_size, self.device,
        )

        epoch_bar = tqdm(
            range(1, epochs + 1),
            desc=self.model_name,
            unit="epoch",
            dynamic_ncols=True,
            leave=True,
        )
        for epoch in epoch_bar:
            t0 = time.time()

            train_loss, train_acc = self._run_epoch(train_loader, criterion, optimizer, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, criterion, None, train=False)

            scheduler.step()
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            epoch_bar.set_postfix(
                tr_loss=f"{train_loss:.4f}",
                vl_loss=f"{val_loss:.4f}",
                vl_acc=f"{val_acc:.3f}",
                s=f"{elapsed:.1f}s",
            )
            if verbose:
                tqdm.write(
                    f"[{self.model_name}] Epoch {epoch:3d}/{epochs} | "
                    f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
                    f"val loss={val_loss:.4f} acc={val_acc:.3f} | {elapsed:.1f}s"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch
                self._save_checkpoint("best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(
                        f"[{self.model_name}] Early stopping at epoch {epoch} "
                        f"(best epoch: {best_epoch}, val_loss: {best_val_loss:.4f})"
                    )
                    break

        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        if fingerprint is not None:
            save_json(fingerprint, fp_path)
        self.fingerprint = fingerprint or {}

        return history

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        train: bool,
    ) -> tuple[float, float]:
        """Single pass over the loader (train or eval)."""
        self.model.train(train)
        total_loss, total_correct, total_samples = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self._forward(X_batch)
                loss = criterion(logits, y_batch)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                preds = logits.argmax(dim=1)
                total_loss += loss.item() * len(y_batch)
                total_correct += (preds == y_batch).sum().item()
                total_samples += len(y_batch)

        return total_loss / total_samples, total_correct / total_samples

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Handles AttentionLSTM which returns (logits, weights)."""
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out

    def _save_checkpoint(self, fname: str) -> None:
        path = os.path.join(self.save_dir, fname)
        torch.save(self.model.state_dict(), path)

    def load_best(self) -> None:
        path = os.path.join(self.save_dir, "best_model.pt")
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Returns probabilities [P(down), P(up)] for each sample."""
        loader = make_dataloader(X, np.zeros(len(X), dtype=np.int64), batch_size,
                                 shuffle=False, model_type=self.model_type)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                logits = self._forward(X_batch)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)


# ---------------------------------------------------------------------------
# Training history visualisation
# ---------------------------------------------------------------------------
def plot_history(
    history: dict,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(history["val_loss"], label="Val", color="tomato")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", color="steelblue")
    axes[1].plot(history["val_acc"], label="Val", color="tomato")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Entry point — quick training demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from data.fetch_data import download_ohlcv, make_multi_stock_dataset
    from models.mlp import build_mlp
    from models.lstm import build_lstm
    from models.cnn import build_cnn
    from imaging.ohlc_chart import make_image_dataset

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_seed(42)
    device = get_device()
    logger.info("Device: %s", device)

    logger.info("=== Downloading data ===")
    raw = download_ohlcv(["AAPL", "MSFT", "GOOGL"], start="2010-01-01", end="2022-12-31")

    WINDOW = 20
    HORIZON = 5

    logger.info("=== Training MLP ===")
    tab_ds = make_multi_stock_dataset(raw, WINDOW, HORIZON, scaling="image")
    mlp = build_mlp(window=WINDOW, n_features=6)
    trainer = Trainer(mlp, "mlp", save_dir=f"checkpoints/mlp_w{WINDOW}")
    history = trainer.fit(
        tab_ds["X_train"], tab_ds["y_train"],
        tab_ds["X_val"], tab_ds["y_val"],
        epochs=10, batch_size=64, patience=3,
    )
    plot_history(history, "MLP")

    logger.info("=== Training LSTM ===")
    lstm = build_lstm(n_features=6)
    trainer = Trainer(lstm, "lstm", save_dir=f"checkpoints/lstm_w{WINDOW}")
    history = trainer.fit(
        tab_ds["X_train"], tab_ds["y_train"],
        tab_ds["X_val"], tab_ds["y_val"],
        epochs=10, batch_size=64, patience=3,
    )
    plot_history(history, "LSTM")

    logger.info("=== Training CNN ===")
    img_ds = make_image_dataset(raw, window=WINDOW, horizon=HORIZON)
    cnn = build_cnn(window=WINDOW)
    trainer = Trainer(cnn, "cnn", save_dir=f"checkpoints/cnn_w{WINDOW}")
    history = trainer.fit(
        img_ds["X_train"], img_ds["y_train"],
        img_ds["X_val"], img_ds["y_val"],
        epochs=10, batch_size=32, patience=3,
    )
    plot_history(history, "CNN")