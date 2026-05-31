"""
FashionMNIST Image Classification with PyTorch
===============================================
Trains a deep MLP on the FashionMNIST dataset (10 classes, 28×28 greyscale images).
Uses batched DataLoaders, CrossEntropyLoss, and a ReduceLROnPlateau scheduler.
Supports checkpoint saving and resuming training across multiple runs.

Usage
-----
    python fashion_mnist.py

On each run the script:
  1. Downloads (or reuses) the FashionMNIST dataset.
  2. Resumes from the last checkpoint if one exists, otherwise trains from scratch.
  3. Trains for EPOCHS_PER_RUN epochs, prints metrics, then saves a checkpoint.
  4. Asks whether to continue training for another EPOCHS_PER_RUN epochs.

Dependencies
------------
    pip install torch torchvision matplotlib
    # helper_function.py must expose accuracy_fn(y_true, y_pred)
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import time
import torch
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from helper_function import accuracy_fn
from timeit import default_timer as timer
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

DEVICE         = "cpu"
RANDOM_SEED    = 42
BATCH_SIZE     = 32
EPOCHS_PER_RUN = 3
LOG_INTERVAL   = 400       # print batch progress every N batches

# Checkpoint
CHECKPOINT_DIR  = Path("FashionMNIST_model")
CHECKPOINT_PATH = CHECKPOINT_DIR / "FashionMNIST_model.pth"


# ── Model ─────────────────────────────────────────────────────────────────────

class FashionMNISTModelV0(nn.Module):
    """
    Deep MLP for FashionMNIST image classification.

    Architecture
    ------------
    784 → 512 → 512 → 512 → 512 → 10

    Flatten converts each (1, 28, 28) image to a 784-dim vector.
    Dropout (p=0.2) is applied after the first three hidden layers.
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),

            nn.Linear(hidden_units, output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_train_time(start: float, end: float, device: torch.device = None) -> None:
    """Print the elapsed wall-clock time for a training run."""
    print(f"  Train time on {device}: {end - start:.3f}s")


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
        print("  No checkpoint found — starting from scratch.")
        return model, optimizer, 0

    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    saved_loss  = checkpoint["loss"]
    print(f"  ✓ Checkpoint loaded — resuming from epoch {start_epoch} "
          f"(loss: {saved_loss:.5f}).")
    return model, optimizer, start_epoch


def eval_model(
    model       : nn.Module,
    data_loader : torch.utils.data.DataLoader,
    accuracy_fn,
    loss_fn     : nn.Module = nn.CrossEntropyLoss(),
) -> dict:
    """
    Evaluate model on a DataLoader and return a results dictionary.

    Returns
    -------
    dict with keys: model_name, model_loss, model_acc
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)
            loss  += loss_fn(y_pred, y)
            acc   += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    loss /= len(data_loader)
    acc  /= len(data_loader)

    return {
        "model_name" : model.__class__.__name__,
        "model_loss" : loss.item(),
        "model_acc"  : acc,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model            : nn.Module,
    optimizer        : torch.optim.Optimizer,
    train_dataloader : DataLoader,
    test_dataloader  : DataLoader,
    start_epoch      : int,
    total_epochs     : int,
) -> tuple[torch.Tensor, float]:
    """
    Run the training loop from `start_epoch` to `total_epochs` (inclusive).

    Uses CrossEntropyLoss and a ReduceLROnPlateau scheduler that halves the
    learning rate after 100 epochs without improvement on the test loss.
    Tracks wall-clock time and prints elapsed time at each log step.

    Parameters
    ----------
    model               : the model to train.
    optimizer           : the model's optimiser (passed in, not recreated here).
    train_dataloader    : batched training DataLoader.
    test_dataloader     : batched evaluation DataLoader.
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
    loss       = None
    n_epochs   = total_epochs - start_epoch
    loop_start = time.perf_counter()           # ← start timer

    for epoch in range(start_epoch, total_epochs):   # fix 1: was total_epochs+1 (off-by-one)
        print(f"\n  Epoch: {epoch + 1}\n  -----")

        # ── Train step ────────────────────────────────────────────────────────
        train_loss, train_acc = 0, 0
        model.train()
        for batch, (x, y) in enumerate(train_dataloader):
            y_pred      = model(x)
            loss        = loss_fn(y_pred, y)
            train_loss += loss
            train_acc  += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % LOG_INTERVAL == 0:
                print(f"    looked at {batch * len(x)}/{len(train_dataloader.dataset)} samples")

        train_loss /= len(train_dataloader)
        train_acc  /= len(train_dataloader)

        # ── Eval step ─────────────────────────────────────────────────────────
        test_loss, test_acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for x_test, y_test in test_dataloader:
                test_preds  = model(x_test)
                test_loss  += loss_fn(test_preds, y_test)
                test_acc   += accuracy_fn(y_true=y_test, y_pred=test_preds.argmax(dim=1))

            test_loss /= len(test_dataloader)
            test_acc  /= len(test_dataloader)

        scheduler.step(test_loss)

        elapsed = time.perf_counter() - loop_start
        print(
            f"  Epoch {epoch + 1:5d} | "
            f"Loss: {train_loss:.5f}  Acc: {train_acc:.2f}% | "
            f"Test loss: {test_loss:.5f}  Test acc: {test_acc:.2f}% | "
            f"Elapsed: {elapsed:.1f}s"
        )

    total_time = time.perf_counter() - loop_start   # ← total elapsed
    print_train_time(
        start=0, end=total_time,
        device=str(next(model.parameters()).device)
    )
    return loss, total_time                          # fix 2: was missing return


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Generating dataset...")
    train_data = datasets.FashionMNIST(
        root             = "data",
        train            = True,
        download         = True,
        transform        = transforms.ToTensor(),
        target_transform = None,
    )

    test_data = datasets.FashionMNIST(
        root             = "data",
        train            = False,
        download         = True,
        transform        = transforms.ToTensor(),
        target_transform = None,
    )

    # DataLoaders batch the dataset so we're not passing all 60k images at once.
    # Batching also gives the model more gradient update steps per epoch.
    train_dataloader = DataLoader(
        dataset    = train_data,
        batch_size = BATCH_SIZE,
        shuffle    = True,    # shuffle so the model doesn't learn sample order
    )

    test_dataloader = DataLoader(
        dataset    = test_data,
        batch_size = BATCH_SIZE,
        shuffle    = False,   # no need to shuffle data the model never trains on
    )

    class_names = train_data.classes
    print(f"  Classes : {class_names}")
    print(f"  Train   : {len(train_data)} samples")
    print(f"  Test    : {len(test_data)} samples")

    # ── Training loop ─────────────────────────────────────────────────────────
    iteration = 0
    while True:
        iteration += 1

        # fix 3: model needs its three required args
        model     = FashionMNISTModelV0(
            input_shape  = 28 * 28,
            hidden_units = 512,
            output_shape = len(class_names),
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model, optimizer, start_epoch = load_checkpoint(model, optimizer)
        target_epoch = start_epoch + EPOCHS_PER_RUN

        print(f"\n── Iteration {iteration}  |  epochs {start_epoch + 1} → {target_epoch} ──")
        iter_start = time.perf_counter()

        # fix 4: train() now returns (loss, train_time) — was returning None
        loss, train_time = train(
            model            = model,
            optimizer        = optimizer,   # fix 5: optimizer no longer recreated inside train()
            train_dataloader = train_dataloader,
            test_dataloader  = test_dataloader,
            start_epoch      = start_epoch,
            total_epochs     = target_epoch,
        )

        iter_time = time.perf_counter() - iter_start
        print(
            f"\n  ✓ Iteration {iteration} done | "
            f"train: {train_time:.2f}s | "
            f"per epoch: {train_time / EPOCHS_PER_RUN:.4f}s | "
            f"total: {iter_time:.2f}s"
        )

        # fix 6: loss is now a valid tensor (not None) so save_checkpoint won't crash
        save_checkpoint(model, optimizer, epoch=target_epoch, loss=loss)

        # ── Final evaluation ──────────────────────────────────────────────────
        torch.manual_seed(RANDOM_SEED)
        model_results = eval_model(
            model       = model,
            data_loader = test_dataloader,
            accuracy_fn = accuracy_fn,
        )
        print(
            f"\n  Eval — "
            f"model: {model_results['model_name']} | "
            f"loss: {model_results['model_loss']:.5f} | "
            f"acc: {model_results['model_acc']:.2f}%"
        )

        answer = input(f"\nTrain for another {EPOCHS_PER_RUN} epochs? (y/n): ").strip().lower()
        if answer != "y":
            print("Done.")
            break


if __name__ == "__main__":
    main()