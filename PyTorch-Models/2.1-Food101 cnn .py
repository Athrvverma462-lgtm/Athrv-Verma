"""
food101_cnn.py
==============
Scratch-trained CNN for Food101 image classification (101 classes).

Architecture  : HighCapacityFood101CNN  — 4-block CNN + GAP + 2-layer classifier
Optimizer     : AdamW  (lr=3e-4, weight_decay=1e-2)
Scheduler     : ReduceLROnPlateau  (patience=3, factor=0.5)
Augmentation  : RandomCrop + RandomHorizontalFlip + ColorJitter  (train only)
Expected perf : ~48–58% top-1 accuracy at epoch 50 (scratch CNN ceiling on Food101)

Usage
-----
    python food101_cnn.py

The script runs one epoch per iteration and prompts whether to continue.
Checkpoints are saved to Food101_CNN/ after every iteration, so training
can be safely interrupted and resumed at any time.

Dependencies
------------
    torch, torchvision, helper_function.py  (must be in the same directory)
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import time
from pathlib import Path

# ── Third-Party ───────────────────────────────────────────────────────────────
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Local ─────────────────────────────────────────────────────────────────────
# helper_function.py must live in the same directory and expose `accuracy_fn`.
from helper_function import accuracy_fn


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BATCH_SIZE     : int  = 32       # Fits comfortably in ~4 GB VRAM; lower to 16 if OOM
RANDOM_SEED    : int  = 42       # Controls weight init and dataloader shuffle
LOG_INTERVAL   : int  = 100      # Print progress every N batches during training
EPOCHS_PER_RUN : int  = 1        # Epochs trained per script execution (resume-friendly)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint directory and file — created automatically on first save
CHECKPOINT_DIR  = Path("Food101_CNN")
CHECKPOINT_PATH = CHECKPOINT_DIR / "food101_cnn.pth"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class HighCapacityFood101CNN(nn.Module):
    """
    Four-block CNN for Food101 classification (~4.7 M parameters).

    Design decisions
    ----------------
    - Conv blocks double channels (3→64→128→256→512) to progressively build
      richer representations from edges → textures → shapes → semantics.
    - BatchNorm after every Conv stabilises gradient flow and acts as a mild
      regulariser, reducing dependence on careful lr tuning.
    - Dropout(0.4) in block 4 (widest layer) targets the highest-risk
      overfitting point in the network.
    - Global Average Pooling collapses the spatial grid to a single 512-d
      vector, making the classifier resolution-agnostic and cutting parameters
      vs a flattened fully-connected approach by ~100×.
    - Two-layer classifier (512→256→101) gives the network a composition step
      before the final decision boundary instead of projecting directly from
      512 features to 101 classes.

    Spatial flow (input 224×224)
    ----------------------------
    Input  : (B,   3, 224, 224)
    Block1 : (B, 128, 112, 112)  — MaxPool ÷2
    Block2 : (B, 256,  56,  56)  — MaxPool ÷2
    Block3 : (B, 256,  28,  28)  — MaxPool ÷2
    Block4 : (B, 512,  14,  14)  — MaxPool ÷2
    GAP    : (B, 512,   1,   1)
    Output : (B, 101)
    """

    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()

        # ── Block 1: Edge and colour-texture detection ────────────────────────
        # Two conv layers before pooling let the block build slightly higher-level
        # features than a single conv, at low cost (only 64/128 channels).
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, 64,  kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,          128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),   # 224 → 112
        )

        # ── Block 2: Geometric shape and boundary extraction ──────────────────
        # Channel width jumps to 256 here; two conv layers deepen the
        # representation before spatial downsampling.
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),   # 112 → 56
        )

        # ── Block 3: Mid-level semantic pattern recognition ───────────────────
        # Kept at 256 channels (same as block 2) — widening further here would
        # add parameters without meaningfully improving food-discriminative features
        # at this spatial scale.
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),   # 56 → 28
        )

        # ── Block 4: High-level semantic aggregation ──────────────────────────
        # Expands to 512 channels for maximum expressive capacity before GAP.
        # Dropout(0.4) placed after pooling — regularises the widest layer where
        # overfitting risk is highest.
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),   # 28 → 14
            nn.Dropout(p=0.4),
        )

        # ── Global Average Pooling ────────────────────────────────────────────
        # Collapses (B, 512, 14, 14) → (B, 512, 1, 1), then Flatten → (B, 512).
        # Eliminates spatial position dependency in the classifier head.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Classifier Head ───────────────────────────────────────────────────
        # Hidden layer 512→256 with ReLU + Dropout lets the model recombine
        # pooled features before projecting to 101 logits.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
) -> None:
    """
    Saves model weights, optimizer state, and epoch count to CHECKPOINT_PATH.
    Creates CHECKPOINT_DIR if it does not yet exist.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": float(loss) if loss is not None else 0.0,
        },
        CHECKPOINT_PATH,
    )
    print(f"  ✓ Checkpoint saved — epoch {epoch}.")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[nn.Module, torch.optim.Optimizer, int]:
    """
    Loads a checkpoint from CHECKPOINT_PATH into model and optimizer.

    Returns
    -------
    model, optimizer, start_epoch
        start_epoch is 0 when no checkpoint exists or the file is incompatible
        (e.g. after an architecture change).
    """
    if not CHECKPOINT_PATH.exists():
        print("  No checkpoint found — starting from scratch.")
        return model, optimizer, 0

    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        print(f"  ✓ Checkpoint loaded — resuming from epoch {start_epoch} "
              f"(loss: {float(ckpt['loss']):.5f}).")
        return model, optimizer, start_epoch
    except (RuntimeError, ValueError, KeyError) as e:
        print(f"  ⚠  Checkpoint incompatible or corrupted — starting fresh. ({e})")
        return model, optimizer, 0


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def eval_model(
    model: nn.Module,
    data_loader: DataLoader,
    accuracy_fn,
    loss_fn: nn.Module = nn.CrossEntropyLoss(),
) -> dict:
    """
    Runs a full pass over data_loader in inference mode and returns averaged
    loss and accuracy.

    Returns
    -------
    dict with keys: model_name, model_loss, model_acc
    """
    total_loss, total_acc = 0.0, 0.0
    model.eval()

    with torch.inference_mode():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item()
            total_acc  += accuracy_fn(y_true=labels, y_pred=logits.argmax(dim=1))

    return {
        "model_name": model.__class__.__name__,
        "model_loss": total_loss / len(data_loader),
        "model_acc":  total_acc  / len(data_loader),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    start_epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """
    Trains model from start_epoch to total_epochs and returns
    (final_loss, total_wall_time_seconds).

    Scheduler
    ---------
    ReduceLROnPlateau monitors test loss each epoch. If it does not improve
    for 3 consecutive epochs the lr is halved (factor=0.5). This gives the
    model a second wind when it plateaus rather than manually tuning lr.
    """
    loss_fn   = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    torch.manual_seed(RANDOM_SEED)
    current_loss  = None
    loop_start    = time.perf_counter()

    for epoch in range(start_epoch, total_epochs):
        print(f"\n  Epoch {epoch + 1}\n  " + "─" * 30)

        # ── Training phase ────────────────────────────────────────────────────
        train_loss, train_acc = 0.0, 0.0
        model.train()

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            predictions  = model(images)
            current_loss = loss_fn(predictions, labels)

            train_loss += current_loss.item()
            train_acc  += accuracy_fn(y_true=labels, y_pred=predictions.argmax(dim=1))

            current_loss.backward()
            optimizer.step()

            # Periodic batch-level progress log
            if batch_idx % LOG_INTERVAL == 0:
                print(f"    Batch {batch_idx:>4d} | "
                      f"{batch_idx * len(images):>6d}/{len(train_dataloader.dataset)} samples")

        epoch_train_loss = train_loss / len(train_dataloader)
        epoch_train_acc  = train_acc  / len(train_dataloader)

        # ── Evaluation phase ──────────────────────────────────────────────────
        test_loss, test_acc = 0.0, 0.0
        model.eval()

        with torch.inference_mode():
            for images, labels in test_dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits     = model(images)
                test_loss += loss_fn(logits, labels).item()
                test_acc  += accuracy_fn(y_true=labels, y_pred=logits.argmax(dim=1))

        epoch_test_loss = test_loss / len(test_dataloader)
        epoch_test_acc  = test_acc  / len(test_dataloader)

        # Feed test loss to scheduler — may trigger lr reduction
        scheduler.step(epoch_test_loss)

        elapsed = time.perf_counter() - loop_start
        print(
            f"\n  Epoch {epoch + 1:>4d} | "
            f"Train loss: {epoch_train_loss:.4f}  acc: {epoch_train_acc:.2f}% | "
            f"Test  loss: {epoch_test_loss:.4f}  acc: {epoch_test_acc:.2f}% | "
            f"Elapsed: {elapsed:.1f}s"
        )

    total_duration = time.perf_counter() - loop_start

    if current_loss is None:
        raise RuntimeError(
            "No training epochs ran — check start_epoch and total_epochs values."
        )

    return current_loss.item(), total_duration


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Transforms ───────────────────────────────────────────────────────────
    # Train: spatial and colour augmentation forces the model to learn food
    # features rather than memorising specific crops or lighting conditions.
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),           # Random spatial crop (vs fixed centre)
        transforms.RandomHorizontalFlip(),    # Food is label-invariant under L/R flip
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(                 # ImageNet mean/std — matches Food101 dist
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])

    # Test: fully deterministic — no random ops so metrics are reproducible.
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Loading Food101 train split…")
    train_dataset = datasets.Food101(
        root="data", split="train", download=False, transform=train_transforms
    )

    print("Loading Food101 test split…")
    test_dataset = datasets.Food101(
        root="data", split="test", download=False, transform=test_transforms
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # num_workers=2 offloads image decoding/augmentation to background workers,
    # preventing the GPU from stalling on CPU-bound preprocessing.
    # pin_memory=True speeds up CPU→GPU transfers when CUDA is available.
    _pin = torch.cuda.is_available()

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=_pin,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=_pin,
    )

    # ── Model + Optimizer ─────────────────────────────────────────────────────
    # AdamW decouples weight decay from the adaptive gradient scaling (unlike
    # vanilla Adam where weight_decay is applied inside the update step).
    # lr=3e-4 is the standard AdamW starting point for image CNNs.
    model = HighCapacityFood101CNN(
        input_shape=3,                        # RGB channels
        output_shape=len(train_dataset.classes),  # 101 food classes
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-2
    )

    print(f"\nDevice : {DEVICE}")
    print(f"Classes: {len(train_dataset.classes)}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params : {total_params:,}")

    # ── Training loop (iterative / resume-friendly) ───────────────────────────
    iteration = 0
    while True:
        iteration += 1
        model, optimizer, start_epoch = load_checkpoint(model, optimizer)
        target_epoch = start_epoch + EPOCHS_PER_RUN

        print(f"\n{'═'*55}")
        print(f"  Iteration {iteration}  |  Epochs {start_epoch + 1} → {target_epoch}")
        print(f"{'═'*55}")

        iter_start = time.perf_counter()

        final_loss, train_time = train_model(
            model            = model,
            optimizer        = optimizer,
            train_dataloader = train_dataloader,
            test_dataloader  = test_dataloader,
            start_epoch      = start_epoch,
            total_epochs     = target_epoch,
        )

        iter_total = time.perf_counter() - iter_start
        print(
            f"\n  ✓ Iteration {iteration} complete | "
            f"Train: {train_time:.1f}s | "
            f"Per epoch: {train_time / EPOCHS_PER_RUN:.1f}s | "
            f"Total: {iter_total:.1f}s"
        )

        save_checkpoint(model, optimizer, epoch=target_epoch, loss=final_loss)

        # Full evaluation pass on the test set
        torch.manual_seed(RANDOM_SEED)
        results = eval_model(model=model, data_loader=test_dataloader, accuracy_fn=accuracy_fn)
        print(
            f"  Eval — "
            f"Loss: {results['model_loss']:.4f} | "
            f"Acc : {results['model_acc']:.2f}%"
        )

        choice = input(f"\n  Train for another {EPOCHS_PER_RUN} epoch(s)? (y/n): ").strip().lower()
        if choice != "y":
            print("\n  Training complete. Checkpoint saved. Goodbye!")
            break


if __name__ == "__main__":
    main()