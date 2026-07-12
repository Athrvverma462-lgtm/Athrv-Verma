"""
Transformer-based text classifier for the 20 Newsgroups dataset.

Each document is vectorized with TF-IDF, and each of the resulting TF-IDF
scores is treated as a single "token" fed into a Transformer encoder
(similar in spirit to how tabular-transformer models treat each feature as
a token). The encoder's pooled output is passed through a small
classification head to predict which of the 20 newsgroup categories a
document belongs to.

Usage:
    python train.py                       # train from scratch, 30 epochs
    python train.py --epochs 50           # train from scratch, 50 epochs
    python train.py --resume              # resume from the last checkpoint
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    max_features: int = 100     # TF-IDF vocabulary size (= transformer sequence length)
    embed_dims: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs_per_run: int = 30
    log_every: int = 10
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_name: str = "text_transformer.pth"
    seed: int = 42

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint_name


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(cfg: Config, device: str):
    """Load 20 Newsgroups, vectorize with TF-IDF, and return tensors + label names."""
    train_raw = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    test_raw = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))

    # Fit the vectorizer on train data only, then reuse it for test data,
    # so no test-set vocabulary or document-frequency statistics leak into training.
    vectorizer = TfidfVectorizer(max_features=cfg.max_features)
    x_train_sparse = vectorizer.fit_transform(train_raw.data)
    x_test_sparse = vectorizer.transform(test_raw.data)

    x_train = torch.tensor(x_train_sparse.toarray(), dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test_sparse.toarray(), dtype=torch.float32, device=device)
    y_train = torch.tensor(train_raw.target, dtype=torch.long, device=device)
    y_test = torch.tensor(test_raw.target, dtype=torch.long, device=device)

    return x_train, y_train, x_test, y_test, train_raw.target_names


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TextTransformerClassifier(nn.Module):
    """
    Treats each TF-IDF feature as a scalar "token": projects it to
    `embed_dims`, adds a learned positional embedding, runs it through a
    standard Transformer encoder, then mean-pools and classifies.
    """

    def __init__(self, num_features: int, num_classes: int, cfg: Config):
        super().__init__()
        self.input_proj = nn.Linear(1, cfg.embed_dims)
        self.pos_embed = nn.Parameter(torch.randn(1, num_features, cfg.embed_dims) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dims,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.embed_dims * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dims),
            nn.Linear(cfg.embed_dims, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)               # (batch, num_features) -> (batch, num_features, 1)
        x = self.input_proj(x)            # -> (batch, num_features, embed_dims)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)                 # mean-pool over the sequence dimension
        return self.head(x)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(cfg: Config, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float) -> None:
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        cfg.checkpoint_path,
    )
    print(f"  >> checkpoint saved (epoch {epoch}, loss {loss:.5f})")


def load_checkpoint(cfg: Config, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Load a checkpoint in-place if one exists. Returns the epoch to resume from (0 if none)."""
    if not cfg.checkpoint_path.exists():
        print("No checkpoint found — starting from scratch")
        return 0

    checkpoint = torch.load(cfg.checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  >> resuming from epoch {checkpoint['epoch']} (loss was {checkpoint['loss']:.5f})")
    return checkpoint["epoch"]


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    start_epoch: int,
    end_epoch: int,
    cfg: Config,
) -> float:
    """Train for epochs in [start_epoch, end_epoch). Returns the final epoch's average loss."""
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5)
    torch.manual_seed(cfg.seed)

    num_samples = x_train.shape[0]
    avg_loss = 0.0

    for epoch in range(start_epoch, end_epoch):
        model.train()
        perm = torch.randperm(num_samples)
        running_loss, num_batches = 0.0, 0

        for start in range(0, num_samples, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            xb, yb = x_train[idx], y_train[idx]

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches
        scheduler.step(avg_loss)

        if epoch % cfg.log_every == 0:
            print(f"epoch {epoch:5d} | train loss {avg_loss:.5f}")

    return avg_loss


@torch.no_grad()
def evaluate(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor, cfg: Config) -> float:
    """Return classification accuracy on the given (test) set."""
    model.eval()
    correct, total = 0, 0

    for start in range(0, x_test.shape[0], cfg.batch_size):
        xb = x_test[start : start + cfg.batch_size]
        yb = y_test[start : start + cfg.batch_size]
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.shape[0]

    return correct / total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer classifier on 20 Newsgroups TF-IDF features.")
    parser.add_argument("--epochs", type=int, default=30, help="epochs to train this run")
    parser.add_argument("--resume", action="store_true", help="resume from the last checkpoint instead of training from scratch")
    parser.add_argument("--interactive", action="store_true", help="after this run, prompt to keep training in further rounds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(epochs_per_run=args.epochs)
    device = get_device()
    print(f"Using device: {device}")

    x_train, y_train, x_test, y_test, target_names = load_data(cfg, device)
    print(f"Loaded {len(x_train)} train / {len(x_test)} test docs across {len(target_names)} classes")

    model = TextTransformerClassifier(cfg.max_features, len(target_names), cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    start_epoch = load_checkpoint(cfg, model, optimizer) if args.resume else 0
    run = 1

    while True:
        target_epoch = start_epoch + cfg.epochs_per_run
        print(f"=== run {run}: epochs {start_epoch} -> {target_epoch} ===")

        final_loss = train(model, optimizer, x_train, y_train, start_epoch, target_epoch, cfg)
        accuracy = evaluate(model, x_test, y_test, cfg)
        print(f"run {run} done | test accuracy: {accuracy:.4f}")

        save_checkpoint(cfg, model, optimizer, target_epoch, final_loss)
        start_epoch = target_epoch

        if not args.interactive:
            break
        if input("Train further? (y/n): ").strip().lower() == "n":
            break
        run += 1


if __name__ == "__main__":
    main()