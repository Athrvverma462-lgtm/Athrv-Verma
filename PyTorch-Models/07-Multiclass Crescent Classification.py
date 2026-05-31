"""
Multiclass Crescent Classification with PyTorch
================================================
Trains a deep MLP on 20 crescent-shaped clusters (20 classes).
Supports checkpoint saving and resuming training across multiple runs.

Usage
-----
    python dfg.py

On each run the script:
  1. Generates the dataset.
  2. Resumes from the last checkpoint if one exists, otherwise trains from scratch.
  3. Trains for 1000 epochs, then plots decision boundaries.
  4. Asks whether to continue training for another 1000 epochs.

Dependencies
------------
    pip install torch scikit-learn matplotlib numpy
    # helper_function.py must expose plot_decision_boundary(model, X, y)
    # helper_function.py must expose accuracy_fn(y_true, y_pred)
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from helper_function import plot_decision_boundary, accuracy_fn


# ── Config ────────────────────────────────────────────────────────────────────

DEVICE       = "cpu"
RANDOM_SEED  = 42
LOG_INTERVAL = 100
EPOCHS_PER_RUN = 1000

# Dataset
N_CRESCENTS      = 20
SAMPLES_PER_MOON = 2500
NOISE            = 0.08
RADIUS_SCALE     = 1.0
RANDOM_STATE     = 42

# Checkpoint
CHECKPOINT_DIR  = Path("Crescents_model")
CHECKPOINT_PATH = CHECKPOINT_DIR / "Crescents_model.pth"


# ── Dataset ───────────────────────────────────────────────────────────────────

def make_multiclass_crescents(
    n_crescents: int = N_CRESCENTS,
    samples_per_moon: int = SAMPLES_PER_MOON,
    noise: float = NOISE,
    radius_scale: float = RADIUS_SCALE,
    random_state: int = RANDOM_STATE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a 2-D multiclass dataset of crescent (half-moon) shapes.

    Each class is one crescent arc from sklearn's make_moons.
    Crescents are arranged in a horizontal row, offset along the x-axis
    so they don't overlap, and each pair shares the same local geometry
    as the classic two-moon pattern.

    Parameters
    ----------
    n_crescents : int
        Number of crescent classes. Each make_moons call yields 2 classes,
        so the actual number of make_moons calls is ceil(n_crescents / 2).
        If n_crescents is odd, the last upper crescent is dropped.
    samples_per_moon : int
        Points sampled per crescent arc (passed as n_samples to make_moons,
        which splits them equally between the two arcs).
    noise : float
        Standard deviation of the Gaussian noise added by make_moons.
        0.0 → perfectly clean arcs; >0.15 → arcs begin to overlap.
    radius_scale : float
        Horizontal gap between successive crescent-pair centres.
        Increase if you want more whitespace between groups.
    random_state : int
        Master seed. Each crescent pair gets seed = random_state + pair_idx
        for full reproducibility without identical geometry across pairs.

    Returns
    -------
    X : torch.Tensor, shape (N, 2)
        2-D point coordinates, where N = n_pairs * samples_per_moon.
    y : torch.Tensor, shape (N,), dtype=torch.long
        Class labels in [0, n_crescents).
    """
    rng     = np.random.default_rng(random_state)
    n_pairs = int(np.ceil(n_crescents / 2))
    x_parts, y_parts = [], []

    for pair_idx in range(n_pairs):
        X_pair, y_pair = make_moons(
            n_samples=samples_per_moon,
            noise=noise,
            random_state=random_state + pair_idx,
        )

        x_offset       = pair_idx * radius_scale * 2.5
        X_pair[:, 0]  += x_offset

        base_class = pair_idx * 2
        y_global   = y_pair + base_class

        if base_class + 1 >= n_crescents:
            mask     = y_global < n_crescents
            X_pair   = X_pair[mask]
            y_global = y_global[mask]

        x_parts.append(X_pair)
        y_parts.append(y_global)

    X_np  = np.vstack(x_parts)
    y_np  = np.concatenate(y_parts)
    X_np -= X_np.mean(axis=0)

    perm = rng.permutation(len(X_np))
    X_np = X_np[perm]
    y_np = y_np[perm]

    return (
        torch.tensor(X_np, dtype=torch.float32),
        torch.tensor(y_np, dtype=torch.long),
    )


# ── Model ─────────────────────────────────────────────────────────────────────

class CrescentModel(nn.Module):
    """
    Deep MLP for multiclass crescent classification.

    Architecture
    ------------
    2 → 64 → 256 → 256 → 256 → 64 → 20

    Dropout (p=0.2) is applied after every wide hidden layer to regularise
    the intermediate representations.
    """

    def __init__(self, num_classes: int = N_CRESCENTS) -> None:
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

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ── Helpers ───────────────────────────────────────────────────────────────────

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
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "optimizer":        optimizer.state_dict(),
            "loss":             loss,
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

    checkpoint  = torch.load(CHECKPOINT_PATH, weights_only=False)
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
) -> tuple[torch.Tensor, float]:
    """
    Run the training loop from `start_epoch` to `total_epochs` (inclusive).

    Uses CrossEntropyLoss and a ReduceLROnPlateau scheduler that halves the
    learning rate after 100 epochs without improvement on the test loss.
    Tracks wall-clock time and prints elapsed time and ETA at each log step.

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
    loss    : the final training loss tensor.
    elapsed : total wall-clock seconds spent in the loop.
    """
    loss_fn   = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, factor=0.5
    )
    torch.manual_seed(RANDOM_SEED)
    n_epochs = total_epochs - start_epoch
    loss     = None

    loop_start = time.perf_counter()

    for epoch in range(start_epoch, total_epochs + 1):
        # ── Train step ────────────────────────────────────────────────────────
        model.train()
        logits      = model(X_train)
        predictions = torch.softmax(logits, dim=1).argmax(dim=1)
        loss        = loss_fn(logits, y_train)
        train_acc   = accuracy_fn(y_true=y_train, y_pred=predictions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Eval step ─────────────────────────────────────────────────────────
        model.eval()
        with torch.inference_mode():
            test_logits      = model(X_test)
            test_predictions = torch.softmax(test_logits, dim=1).argmax(dim=1)
            test_loss        = loss_fn(test_logits, y_test)
            test_acc         = accuracy_fn(y_true=y_test, y_pred=test_predictions)

        scheduler.step(test_loss.item())

        if epoch % LOG_INTERVAL == 0:
            elapsed   = time.perf_counter() - loop_start
            done      = epoch - start_epoch
            remaining = (elapsed / done * (n_epochs - done)) if done > 0 else 0.0
            print(
                f"  Epoch {epoch:5d} | "
                f"Loss: {loss:.5f}  Acc: {train_acc:.2f}% | "
                f"Test loss: {test_loss:.5f}  Test acc: {test_acc:.2f}% | "
                f"Elapsed: {elapsed:6.1f}s  ETA: {remaining:6.1f}s"
            )

    total_time = time.perf_counter() - loop_start
    return loss, total_time


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
    X, y = make_multiclass_crescents()
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  classes : {y.unique().tolist()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    iteration = 0
    while True:
        iteration += 1
        model     = CrescentModel(N_CRESCENTS).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model, optimizer, start_epoch = load_checkpoint(model, optimizer)
        target_epoch = start_epoch + EPOCHS_PER_RUN

        print(f"\n── Iteration {iteration}  |  epochs {start_epoch + 1} → {target_epoch} ──")
        iter_start = time.perf_counter()

        loss, train_time = train(
            model        = model,
            optimizer    = optimizer,
            X_train      = X_train,
            y_train      = y_train,
            X_test       = X_test,
            y_test       = y_test,
            start_epoch  = start_epoch,
            total_epochs = target_epoch,
        )

        iter_time = time.perf_counter() - iter_start
        print(
            f"\n  ✓ Iteration {iteration} done | "
            f"train: {train_time:.2f}s | "
            f"per epoch: {train_time / EPOCHS_PER_RUN:.4f}s | "
            f"total: {iter_time:.2f}s"
        )

        plot_results(model, X_test, y_test)
        save_checkpoint(model, optimizer, epoch=target_epoch, loss=loss)

        answer = input("\nTrain for another 1 000 epochs? (y/n): ").strip().lower()
        if answer != "y":
            print("Done.")
            break


if __name__ == "__main__":
    main()