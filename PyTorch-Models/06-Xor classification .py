"""
Multiclass XOR Classification with PyTorch
===========================================
Trains a deep MLP on a 3x3 grid of XOR-patterned clusters (9 classes).
Supports checkpoint saving and resuming training across multiple runs.

Usage
-----
    python xor_classification.py

On each run the script:
  1. Generates (or reuses) the dataset and shows a scatter plot.
  2. Resumes from the last checkpoint if one exists, otherwise trains from scratch.
  3. Trains for EPOCHS_PER_RUN epochs, then plots decision boundaries.
  4. Asks whether to continue training for another EPOCHS_PER_RUN epochs.

Dependencies
------------
    pip install torch scikit-learn matplotlib numpy
    # helper_function.py must expose plot_decision_boundary(model, X, y)
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
from sklearn.model_selection import train_test_split
from helper_function import plot_decision_boundary


# ── Config ────────────────────────────────────────────────────────────────────

DEVICE          = "cpu"
RANDOM_SEED     = 42
LEARNING_RATE   = 1e-3
EPOCHS_PER_RUN  = 1000
LOG_INTERVAL    = 100       # print metrics every N epochs

# Dataset
POINTS_PER_BLOCK = 200
GRID_SIZE        = 3        # produces GRID_SIZE² = 9 classes
SCALE            = 1.0
GAP              = 2.4
NOISE            = 0.05

# Checkpoint
CHECKPOINT_DIR  = Path("xor_model")
CHECKPOINT_PATH = CHECKPOINT_DIR / "xor_model.pth"


# ── Dataset ───────────────────────────────────────────────────────────────────

def make_multiclass_xor(
    points_per_block: int = POINTS_PER_BLOCK,
    grid_size: int = GRID_SIZE,
    scale: float = SCALE,
    gap: float = GAP,
    noise: float = NOISE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a 2-D multiclass dataset arranged as a grid of XOR patterns.

    Each cell in the grid contributes two interleaved classes whose boundary
    follows a checkerboard (XOR) rule within that cell.

    Parameters
    ----------
    points_per_block : int
        Number of points sampled per grid cell.
    grid_size : int
        Number of cells along each axis; total classes = grid_size².
    scale : float
        Half-width of the uniform distribution inside each cell.
    gap : float
        Distance between cell centres along each axis.
    noise : float
        Standard deviation of Gaussian jitter added to each point.

    Returns
    -------
    X : torch.Tensor, shape (N, 2)
    y : torch.Tensor, shape (N,), dtype=torch.long
    """
    offsets  = np.linspace(-gap, gap, grid_size)
    x_list, y_list = [], []
    class_id = 0

    for row in range(grid_size):
        for col in range(grid_size):
            cx, cy = offsets[col], offsets[row]
            for _ in range(points_per_block):
                px = (np.random.rand() - 0.5) * scale * 2
                py = (np.random.rand() - 0.5) * scale * 2
                is_xor = (px > 0) != (py > 0)
                label  = class_id if is_xor else (class_id + 1) % (grid_size ** 2)
                x_list.append([
                    cx + px + np.random.randn() * noise,
                    cy + py + np.random.randn() * noise,
                ])
                y_list.append(label)
        class_id = (class_id + 2) % (grid_size ** 2)

    X = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y


# ── Model ─────────────────────────────────────────────────────────────────────

class XORModel(nn.Module):
    """
    Deep MLP for multiclass XOR classification.

    Architecture
    ------------
    2 → 64 → 256 → 256 → 256 → 256 → 64 → 64 → 9

    Dropout (p=0.2) is applied after every hidden layer except the last two
    to regularise the wide intermediate layers.
    """

    def __init__(self, num_classes: int = GRID_SIZE ** 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),

            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ── Helpers ───────────────────────────────────────────────────────────────────

def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Return percentage of correct predictions."""
    return torch.eq(predictions, targets).sum().item() / len(targets) * 100


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: torch.Tensor,
) -> None:
    """Persist model weights, optimiser state, epoch, and loss to disk."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":             epoch,
            "model_state_dict":  model.state_dict(),
            "optimizer":         optimizer.state_dict(),
            "loss":              loss,
        },
        CHECKPOINT_PATH,
    )
    print(f"  ✓ Checkpoint saved at epoch {epoch}.")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[nn.Module, torch.optim.Optimizer, int]:
    """
    Load weights and optimiser state from the latest checkpoint.

    Returns the model and optimiser with restored state, plus the epoch at
    which training was last saved. Returns epoch=0 if no checkpoint exists.
    """
    if not CHECKPOINT_PATH.exists():
        print("No checkpoint found — starting from scratch.")
        return model, optimizer, 0

    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    saved_loss  = checkpoint["loss"]
    print(f"  ✓ Checkpoint loaded — resuming from epoch {start_epoch} "
          f"(loss: {saved_loss:.5f}).")
    return model, optimizer, start_epoch


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    start_epoch: int,
    total_epochs: int,
) -> torch.Tensor:
    """
    Run the training loop from `start_epoch` to `total_epochs` (inclusive).

    Uses CrossEntropyLoss and a ReduceLROnPlateau scheduler that halves the
    learning rate after 100 epochs without improvement on the test loss.

    Parameters
    ----------
    model, optimizer    : the model and its optimiser.
    X_train, y_train    : training split.
    X_test,  y_test     : evaluation split.
    start_epoch         : first epoch index (0 for a fresh run, or the value
                          stored in the checkpoint when resuming).
    total_epochs        : last epoch index to train up to.

    Returns
    -------
    loss : the final training loss tensor.
    """
    loss_fn   = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, factor=0.5
    )
    torch.manual_seed(RANDOM_SEED)
    loss = None

    for epoch in range(start_epoch, total_epochs + 1):
        # ── Train step ────────────────────────────────────────────────────────
        model.train()
        logits      = model(X_train)
        predictions = torch.softmax(logits, dim=1).argmax(dim=1)
        loss        = loss_fn(logits, y_train)
        train_acc   = accuracy(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Eval step ─────────────────────────────────────────────────────────
        model.eval()
        with torch.inference_mode():
            test_logits      = model(X_test)
            test_predictions = torch.softmax(test_logits, dim=1).argmax(dim=1)
            test_loss        = loss_fn(test_logits, y_test)
            test_acc         = accuracy(test_predictions, y_test)

        scheduler.step(test_loss.item())

        if epoch % LOG_INTERVAL == 0:
            print(
                f"  Epoch {epoch:5d} | "
                f"Loss: {loss:.5f}  Acc: {train_acc:.2f}% | "
                f"Test loss: {test_loss:.5f}  Test acc: {test_acc:.2f}%"
            )

    return loss


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> None:
    """Display decision boundary plots for ground-truth labels and predictions."""
    model.eval()
    with torch.inference_mode():
        test_pred = torch.softmax(model(X_test), dim=1).argmax(dim=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    plt.sca(ax1)
    ax1.set_title("Ground truth")
    plot_decision_boundary(model, X_test, y_test)

    plt.sca(ax2)
    ax2.set_title("Predictions")
    plot_decision_boundary(model, X_test, test_pred)

    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # ── Data ──────────────────────────────────────────────────────────────────
    print("Generating dataset...")
    X, y = make_multiclass_xor()

    plt.figure(figsize=(6, 6))
    plt.title("Dataset")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, s=10)
    plt.tight_layout()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    while True:
        model     = XORModel().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        model, optimizer, start_epoch = load_checkpoint(model, optimizer)
        target_epoch = start_epoch + EPOCHS_PER_RUN

        print(f"\nTraining epochs {start_epoch + 1} → {target_epoch} …")
        loss = train(
            model       = model,
            optimizer   = optimizer,
            X_train     = X_train,
            y_train     = y_train,
            X_test      = X_test,
            y_test      = y_test,
            start_epoch = start_epoch,
            total_epochs= target_epoch,
        )

        plot_results(model, X_test, y_test)
        save_checkpoint(model, optimizer, epoch=target_epoch, loss=loss)

        answer = input("\nTrain for another 1 000 epochs? (y/n): ").strip().lower()
        if answer != "y":
            print("Done.")
            break


if __name__ == "__main__":
    main()