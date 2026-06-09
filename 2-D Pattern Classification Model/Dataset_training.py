"""
2D Classification Pattern Datasets
====================================
All generators return  (X, y)  as torch.Tensor pairs.
  X : shape (N, 2),  dtype=torch.float32
  y : shape (N,),    dtype=torch.long

Dataset structure — 25 datasets in 5 groups by class count:
  Group 1 — 2  classes (5 datasets): TwoSpirals, Circles, TwoMoons, XOR, SineBoundary
  Group 2 — 6  classes (5 datasets): ConcentricRings_6,  AngularWedges_6,  GaussianBlobs_6,  MultiArmSpiral_6,  GridGaussians_6
  Group 3 — 10 classes (5 datasets): ConcentricRings_10, AngularWedges_10, GaussianBlobs_10, MultiArmSpiral_10, GridGaussians_10
  Group 4 — 14 classes (5 datasets): ConcentricRings_14, AngularWedges_14, GaussianBlobs_14, MultiArmSpiral_14, GridGaussians_14
  Group 5 — 20 classes (5 datasets): ConcentricRings_20, AngularWedges_20, GaussianBlobs_20, MultiArmSpiral_20, GridGaussians_20

Each multi-class generator is a single parameterised function called with
different n_classes values — no code duplication, no magic class-count
inference from heuristics.

Datasets are lazily loaded — nothing is computed on import.
Data is generated and split the first time a dataset is accessed,
then cached for subsequent accesses.
"""

import numpy as np
import torch
import time
import random
from pathlib import Path
from torch import nn
from sklearn.model_selection import train_test_split as _split
from sklearn.datasets import (
    make_circles          as _sk_circles,
    make_moons            as _sk_moons,
    make_blobs            as _sk_blobs,
)
from helper_function import accuracy_fn

# ── Global defaults ────────────────────────────────────────────────────────────
N_SAMPLES     : int   = 10000   # base sample count (multi-class generators scale up)
NOISE         : float = 0.10
RANDOM_STATE  : int   = 42
TEST_SIZE     : float = 0.20
EPOCHS_PER_RUN: int   = 1000
RANDOM_SEED   : int   = 42
DEVICE        : str   = "cpu"
LOG_INTERVAL  : int   = 100

# ── Per-run seed — varied each launch so train/test splits and generator noise
#    differ across runs, preventing the model from memorising a fixed split.
RUN_SEED: int = random.randint(0, 99999)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _to_torch(X: np.ndarray, y: np.ndarray):
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long))

def _to_float_tensor(arr): return torch.tensor(arr, dtype=torch.float32)
def _to_label_tensor(arr): return torch.tensor(arr, dtype=torch.long)

def _counts_per_class(total: int, n_classes: int) -> np.ndarray:
    """Distribute *total* samples as evenly as possible across *n_classes*."""
    if n_classes <= 0:
        raise ValueError("n_classes must be > 0")
    if total < n_classes:
        raise ValueError("n_samples must be >= n_classes")
    base   = total // n_classes
    rem    = total % n_classes
    counts = np.full(n_classes, base, dtype=int)
    if rem:
        counts[:rem] += 1
    return counts

def _n_samples_for(n_classes: int, base: int = N_SAMPLES) -> int:
    """Scale sample count with class count so every class has enough points."""
    return max(base, n_classes * 300)


# ══════════════════════════════════════════════════════════════════════════════
# LAZY DATASET WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class _LazyDataset:
    def __init__(self, fn, **kwargs):
        self._fn     = fn
        self._kwargs = kwargs
        self._cache  = None

    def _load(self):
        if self._cache is None:
            X, y       = self._fn(**self._kwargs)
            X_np, y_np = X.numpy(), y.numpy()
            X_tr, X_te, y_tr, y_te = _split(
                X_np, y_np,
                test_size    = TEST_SIZE,
                random_state = RUN_SEED,   # varies each launch — no fixed split memorisation
                stratify     = y_np,
            )
            self._cache = (
                _to_float_tensor(X_tr), _to_label_tensor(y_tr),
                _to_float_tensor(X_te), _to_label_tensor(y_te),
            )
        return self._cache

    def fresh_eval_data(self, n_samples: int | None = None):
        """Generate a brand-new batch with a different seed for deployment validation.

        This data is completely unseen — different noise realisation from the
        training/test sets — so it catches models that memorised the fixed seed.

        n_samples is only forwarded to generators that accept it (2-class group).
        Multi-class generators compute their own sample count from n_classes
        internally via _n_samples_for(), so injecting n_samples raises TypeError.
        """
        import inspect
        fresh_seed = RUN_SEED + 1000   # guaranteed different from training seed
        kwargs     = dict(self._kwargs)
        if n_samples is not None and "n_samples" in inspect.signature(self._fn).parameters:
            kwargs["n_samples"] = n_samples
        kwargs["seed"] = fresh_seed
        return self._fn(**kwargs)

    def __iter__(self):   return iter(self._load())
    def __getitem__(self, idx): return self._load()[idx]


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 1 — 2 CLASSES
# Geometrically diverse, all naturally binary.
# ══════════════════════════════════════════════════════════════════════════════

def make_two_spirals(n_samples=N_SAMPLES, noise=0.35, n_turns=1.50, scale=5.00, seed=RANDOM_STATE):
    """Two interleaved Archimedean spirals."""
    np.random.seed(seed)
    n     = n_samples // 2
    theta = np.linspace(0, n_turns * 2 * np.pi, n)
    r     = np.linspace(0.5, scale, n)
    X0 = np.column_stack([r * np.cos(theta),         r * np.sin(theta)])
    X1 = np.column_stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])
    X0 += np.random.randn(*X0.shape) * noise
    X1 += np.random.randn(*X1.shape) * noise
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(int)
    return _to_torch(X, y)


def make_circles(n_samples=N_SAMPLES, noise=0.08, factor=0.50, seed=RANDOM_STATE):
    """Sklearn concentric circles — inner vs outer ring."""
    X, y = _sk_circles(n_samples=n_samples, noise=noise,
                       factor=factor, random_state=seed)
    return _to_torch(X, y)


def make_two_moons(n_samples=N_SAMPLES, noise=0.15, seed=RANDOM_STATE):
    """Sklearn two-moons — two crescent shapes."""
    X, y = _sk_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return _to_torch(X, y)


def make_xor(n_samples=N_SAMPLES, scale=3.00, noise=0.10, seed=RANDOM_STATE):
    """XOR quadrant pattern — label = sign(x) XOR sign(y)."""
    np.random.seed(seed)
    px    = (np.random.rand(n_samples) - 0.5) * scale * 2
    py    = (np.random.rand(n_samples) - 0.5) * scale * 2
    label = ((px > 0) != (py > 0)).astype(int)
    X = np.column_stack([
        px + np.random.randn(n_samples) * noise,
        py + np.random.randn(n_samples) * noise,
    ])
    return _to_torch(X, label)


def make_sine_boundary(n_samples=N_SAMPLES, scale=5.00, noise=0.15,
                       frequency=1.00, amplitude=1.50, seed=RANDOM_STATE):
    """Points above vs below a sinusoidal boundary curve."""
    np.random.seed(seed)
    px       = (np.random.rand(n_samples) - 0.5) * scale * 2
    py       = (np.random.rand(n_samples) - 0.5) * scale * 2
    boundary = amplitude * np.sin(frequency * px)
    label    = (py > boundary).astype(int)
    X = np.column_stack([
        px + np.random.randn(n_samples) * noise,
        py + np.random.randn(n_samples) * noise,
    ])
    return _to_torch(X, label)


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 2-5 — MULTI-CLASS GENERATORS (6 / 10 / 14 / 20 classes)
# Each generator takes n_classes as an explicit argument.
# Called multiple times from the registry with different values.
# ══════════════════════════════════════════════════════════════════════════════

# ── Pattern A: Concentric Rings ────────────────────────────────────────────────

def make_concentric_rings(n_classes: int, ring_width=0.80, gap=0.25, noise=0.07, seed=RANDOM_STATE):
    """
    n_classes tightly-spaced concentric annuli.
    ring_width and gap are kept fixed so inner/outer rings stay proportional.
    Sample count scales with n_classes so each ring is well-populated.
    """
    np.random.seed(seed)
    n_samples = _n_samples_for(n_classes, base=2000)
    counts    = _counts_per_class(n_samples, n_classes)
    x_list, y_list = [], []
    for k, n_k in enumerate(counts):
        r_inner = k * (ring_width + gap)
        r_outer = r_inner + ring_width
        r       = np.random.uniform(r_inner, r_outer, size=n_k)
        theta   = np.random.uniform(0, 2 * np.pi, size=n_k)
        Xk      = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        Xk     += np.random.randn(*Xk.shape) * noise
        x_list.append(Xk)
        y_list.append(np.full(n_k, k, dtype=int))
    return _to_torch(np.vstack(x_list), np.hstack(y_list))


# ── Pattern B: Angular Wedges ──────────────────────────────────────────────────

def make_angular_wedges(n_classes: int, r_min=0.30, r_max=5.00, noise=0.08, seed=RANDOM_STATE):
    """
    n_classes pie-slice wedges of equal angular width.
    Boundary sharpness (noise) stays constant so difficulty scales purely
    with the number of wedges (narrower slices = harder).
    """
    np.random.seed(seed)
    n_samples = _n_samples_for(n_classes, base=2000)
    theta  = np.random.uniform(0, 2 * np.pi, size=n_samples)
    r      = np.random.uniform(r_min, r_max, size=n_samples)
    labels = (theta / (2 * np.pi) * n_classes).astype(int) % n_classes
    X      = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    X     += np.random.randn(*X.shape) * noise
    return _to_torch(X, labels)


# ── Pattern C: Gaussian Blobs ──────────────────────────────────────────────────

def make_gaussian_blobs(n_classes: int, cluster_std=0.55, seed=RANDOM_STATE):
    """
    n_classes isotropic Gaussian clusters placed by sklearn.
    cluster_std is fixed so within-class spread doesn't change with n_classes.
    """
    np.random.seed(seed)
    n_samples = _n_samples_for(n_classes, base=2000)
    X, y = _sk_blobs(n_samples=n_samples, centers=n_classes,
                     cluster_std=cluster_std, random_state=seed)
    return _to_torch(X, y.astype(int))


# ── Pattern D: Multi-Arm Spiral ────────────────────────────────────────────────

def make_multiarm_spiral(n_classes: int, noise=0.06, n_turns=1.50, scale=8.00, seed=RANDOM_STATE):
    """
    n_classes evenly-spaced spiral arms rotating out from the origin.
    scale=8 and low noise keep the arms resolvable even at 20 classes
    (angular gap ≈ 0.31 rad at inner radius vs noise band ≈ 0.06 units).
    """
    np.random.seed(seed)
    n_samples = _n_samples_for(n_classes, base=3000)
    counts    = _counts_per_class(n_samples, n_classes)
    x_list, y_list = [], []
    for k, n_k in enumerate(counts):
        theta  = np.linspace(0.3, n_turns * 2 * np.pi, n_k)
        r      = np.linspace(0.5, scale, n_k)
        offset = 2 * np.pi * k / n_classes
        Xk     = np.column_stack([r * np.cos(theta + offset),
                                  r * np.sin(theta + offset)])
        Xk    += np.random.randn(*Xk.shape) * noise
        x_list.append(Xk)
        y_list.append(np.full(n_k, k, dtype=int))
    return _to_torch(np.vstack(x_list), np.hstack(y_list))


# ── Pattern E: Grid Gaussians ──────────────────────────────────────────────────

def make_grid_gaussians(n_classes: int, gap=2.20, spread=0.28, seed=RANDOM_STATE):
    """
    n_classes Gaussian clusters arranged on a regular 2D grid.
    Grid side = ceil(sqrt(n_classes)); unused grid cells are skipped.
    gap and spread are fixed so cluster separation stays constant.
    """
    np.random.seed(seed)
    n_samples = _n_samples_for(n_classes, base=2000)
    grid_dim  = int(np.ceil(np.sqrt(n_classes)))
    counts    = _counts_per_class(n_samples, n_classes)
    x_list, y_list = [], []
    label = 0
    for row in range(grid_dim):
        for col in range(grid_dim):
            if label >= n_classes:
                break
            center = np.array([col * gap, row * gap])
            n_k    = int(counts[label])
            Xk     = center + np.random.randn(n_k, 2) * spread
            x_list.append(Xk)
            y_list.append(np.full(n_k, label, dtype=int))
            label += 1
    return _to_torch(np.vstack(x_list), np.hstack(y_list))


# ══════════════════════════════════════════════════════════════════════════════
# DATASET REGISTRY
# 25 datasets in strict class-count order:
#   5 × 2-class → 5 × 6-class → 5 × 10-class → 5 × 14-class → 5 × 20-class
# ══════════════════════════════════════════════════════════════════════════════

DATASETS: dict[str, _LazyDataset] = {

    # ── Group 1: 2 classes ─────────────────────────────────────────────────────
    "TwoSpirals":            _LazyDataset(make_two_spirals),
    "Circles":               _LazyDataset(make_circles),
    "TwoMoons":              _LazyDataset(make_two_moons),
    "XOR":                   _LazyDataset(make_xor),
    "SineBoundary":          _LazyDataset(make_sine_boundary),

    # ── Group 2: 6 classes ─────────────────────────────────────────────────────
    "ConcentricRings_6":     _LazyDataset(make_concentric_rings,  n_classes=6),
    "AngularWedges_6":       _LazyDataset(make_angular_wedges,    n_classes=6),
    "GaussianBlobs_6":       _LazyDataset(make_gaussian_blobs,    n_classes=6),
    "MultiArmSpiral_6":      _LazyDataset(make_multiarm_spiral,   n_classes=6),
    "GridGaussians_6":       _LazyDataset(make_grid_gaussians,    n_classes=6),

    # ── Group 3: 10 classes ────────────────────────────────────────────────────
    "ConcentricRings_10":    _LazyDataset(make_concentric_rings,  n_classes=10),
    "AngularWedges_10":      _LazyDataset(make_angular_wedges,    n_classes=10),
    "GaussianBlobs_10":      _LazyDataset(make_gaussian_blobs,    n_classes=10),
    "MultiArmSpiral_10":     _LazyDataset(make_multiarm_spiral,   n_classes=10),
    "GridGaussians_10":      _LazyDataset(make_grid_gaussians,    n_classes=10),

    # ── Group 4: 14 classes ────────────────────────────────────────────────────
    "ConcentricRings_14":    _LazyDataset(make_concentric_rings,  n_classes=14),
    "AngularWedges_14":      _LazyDataset(make_angular_wedges,    n_classes=14),
    "GaussianBlobs_14":      _LazyDataset(make_gaussian_blobs,    n_classes=14),
    "MultiArmSpiral_14":     _LazyDataset(make_multiarm_spiral,   n_classes=14),
    "GridGaussians_14":      _LazyDataset(make_grid_gaussians,    n_classes=14),

    # ── Group 5: 20 classes ────────────────────────────────────────────────────
    "ConcentricRings_20":    _LazyDataset(make_concentric_rings,  n_classes=20),
    "AngularWedges_20":      _LazyDataset(make_angular_wedges,    n_classes=20),
    "GaussianBlobs_20":      _LazyDataset(make_gaussian_blobs,    n_classes=20),
    "MultiArmSpiral_20":     _LazyDataset(make_multiarm_spiral,   n_classes=20),
    "GridGaussians_20":      _LazyDataset(make_grid_gaussians,    n_classes=20),
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
#
# Design principles (anti-overfitting):
#
# 1. FourierFeatureMapping — B matrix is fixed (non-trainable) and small.
#    scale is kept modest so the model learns smooth low-to-mid frequency
#    functions rather than memorising point-level density fluctuations.
#    The sin/cos output is L2-normalised per sample so that the downstream
#    linear layer sees unit-variance inputs regardless of scale, preventing
#    the first linear from compensating for scale by growing large weights.
#
# 2. ResidualBlock — bottleneck design (hidden → hidden//2 → hidden) with
#    dropout on the bottleneck only.  The narrow bottleneck forces the block
#    to compress information, acting as an implicit regulariser.  LayerNorm
#    is applied *before* the residual addition (Pre-LN) which gives more
#    stable gradients than Post-LN and allows lower learning rates.
#
# 3. FourierMLP — n_blocks is capped at 2.  More blocks on a 2D task add
#    capacity without generalisation.  The output head is a single linear
#    layer directly on the hidden representation — no hidden//2 bottleneck
#    at the end, which was previously the largest source of extra parameters.
#
# 4. Parameter count targets (approximate):
#    LOW/2-class:   ~15k–30k params   (was ~80k+)
#    MEDIUM/6-cl:   ~40k–80k params   (was ~200k+)
#    HIGH/20-cl:    ~100k–200k params  (was ~800k+)
# ══════════════════════════════════════════════════════════════════════════════

class FourierFeatureMapping(nn.Module):
    """
    Random Fourier features.  B is fixed — never trained.
    Output is L2-normalised so downstream layers see stable variance.
    """
    def __init__(self, input_dim: int, mapping_size: int, scale: float):
        super().__init__()
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B                                        # (N, M)
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2M)
        # L2-normalise each sample: keeps magnitude information but bounds scale
        return feat 


class ResidualBlock(nn.Module):
    """
    Bottleneck residual block: hidden → hidden//2 → hidden.
    Dropout only on the narrow bottleneck — regularises without killing signal.
    Pre-LN: LayerNorm before the transform, not after addition.
    """
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        neck = max(dim // 2, 16)
        self.norm    = nn.LayerNorm(dim)
        self.down    = nn.Linear(dim, neck)
        self.act     = nn.GELU()
        self.drop    = nn.Dropout(dropout)
        self.up      = nn.Linear(neck, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)          # pre-norm
        h = self.down(h)
        h = self.act(h)
        h = self.drop(h)          # dropout only at bottleneck
        h = self.up(h)
        return x + h              # residual addition


class FourierMLP(nn.Module):
    """
    Fourier feature MLP with optional residual blocks.

    Network layout:
        fourier(x)  →  Linear(2M, hidden)  →  GELU  →  Dropout
        →  [ResidualBlock] × n_blocks
        →  Linear(hidden, output_dim)       ← single output head, no extra bottleneck

    n_blocks is capped at 2.  The output head maps directly from hidden_dim
    to output_dim — removing the hidden//2 intermediate layer cuts ~30% of
    parameters and forces the residual blocks to carry the representational load.
    """
    def __init__(self, output_dim: int, hidden_dim: int, dropout: float,
                 scale: float, mapping_size: int = 48,
                 n_blocks: int = 1, input_dim: int = 2):
        super().__init__()
        self.fourier = FourierFeatureMapping(input_dim, mapping_size, scale)
        fourier_out  = mapping_size * 2

        self.input_layer = nn.Sequential(
            nn.Linear(fourier_out, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(self.fourier(x))
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

BATCH_SIZE    = 2048
EVAL_INTERVAL = 25


def run_training_loop(model, num_classes, optimizer, dataset_name,
                      X_train, X_test, y_train, y_test,
                      start_epoch, total_epochs, label_smoothing=0.05):
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - start_epoch, eta_min=1e-5,
    )
    device    = next(model.parameters()).device

    X_mean    = X_train.mean(0)
    X_std     = X_train.std(0).clamp(min=1e-6)
    X_train_n = ((X_train - X_mean) / X_std).to(device)
    X_test_n  = ((X_test  - X_mean) / X_std).to(device)
    y_train   = y_train.to(device)
    y_test    = y_test.to(device)
    n_train   = X_train_n.size(0)

    rng = torch.Generator(device=device)
    rng.manual_seed(RUN_SEED + start_epoch)

    log_interval  = max(EVAL_INTERVAL, (total_epochs - start_epoch) // 4)
    start_time    = time.perf_counter()
    train_loss    = 0.0
    test_loss_val = 0.0
    test_acc_val  = 0.0

    # Initialised before the loop so they are always defined even when
    # start_epoch == total_epochs (zero-epoch run).
    do_log     = False
    epoch_loss = torch.zeros(1, device=device)

    for epoch in range(start_epoch, total_epochs):
        model.train()
        perm       = torch.randperm(n_train, generator=rng, device=device)
        epoch_loss = torch.zeros(1, device=device)

        for i in range(0, n_train, BATCH_SIZE):
            xb = X_train_n[perm[i : i + BATCH_SIZE]]
            yb = y_train  [perm[i : i + BATCH_SIZE]]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.detach()

        scheduler.step()

        do_log = (
            (epoch + 1 - start_epoch) % log_interval == 0
            or epoch == start_epoch
            or epoch == total_epochs - 1
        )
        if do_log:
            n_batches  = -(-n_train // BATCH_SIZE)
            train_loss = (epoch_loss / n_batches).item()
            model.eval()
            with torch.inference_mode():
                test_logits   = model(X_test_n)
                test_loss_val = loss_fn(test_logits, y_test).item()
                test_acc_val  = (test_logits.argmax(1) == y_test).float().mean().item() * 100
            model.train()
            print(f"    [{epoch+1:04d}/{total_epochs}] "
                  f"loss={train_loss:.4f} | test_loss={test_loss_val:.4f} | "
                  f"acc={test_acc_val:.2f}%")

    # Final eval if last epoch wasn't a log epoch, and only if any epochs ran.
    if not do_log and total_epochs > start_epoch:
        n_batches  = -(-n_train // BATCH_SIZE)
        train_loss = (epoch_loss / n_batches).item()
        model.eval()
        with torch.inference_mode():
            test_logits   = model(X_test_n)
            test_loss_val = loss_fn(test_logits, y_test).item()
            test_acc_val  = (test_logits.argmax(1) == y_test).float().mean().item() * 100

    return train_loss, time.perf_counter() - start_time, X_mean, X_std


def eval_model(model, X_test, y_test, accuracy_fn, X_mean=None, X_std=None):
    model.eval()
    device = next(model.parameters()).device
    if X_mean is not None and X_std is not None:
        X_test = (X_test - X_mean) / X_std
    X_test = X_test.to(device)
    with torch.inference_mode():
        logits = model(X_test)
        preds  = logits.argmax(dim=1).cpu()
    return {"accuracy": accuracy_fn(y_true=y_test, y_pred=preds)}


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
#
# One .pth file per dataset: checkpoints/<dataset_name>.pth
#
# Splitting the monolithic registry into individual files solves two problems:
#
#   1. Architecture mismatch — each save fully overwrites the single dataset
#      file with the current architecture's weights.  There is no stale key
#      from a previous architecture sitting in the same blob, so the
#      load_state_dict mismatch warning disappears.
#
#   2. File size — each file holds exactly one model (~15 KB-200 KB depending
#      on capacity bucket) instead of all 25 models in one blob (~5-10 MB and
#      growing).  The server reads individual files by dataset name, which is
#      unchanged from its current interface.
#
#   fresh_acc and all other metadata are written into each per-dataset file
#   alongside the model weights so nothing is lost.
# ──────────────────────────────────────────────────────────────────────────────

def _ckpt_path(ckpt_dir: Path, dataset_name: str) -> Path:
    """Return the per-dataset checkpoint path."""
    return ckpt_dir / f"{dataset_name}.pth"


def save_unified_registry(registry: dict, path: Path) -> None:
    """Save each dataset entry as its own .pth file inside the checkpoint dir.

    *path* is the master registry path passed in from main(); the directory
    that contains it is used as the checkpoint folder so the call-site in
    main() does not need to change.

    Each file is written atomically (write to .tmp, then rename) so the server
    never reads a half-written checkpoint.
    """
    ckpt_dir = path.parent
    for dataset_name, entry in registry.items():
        out = _ckpt_path(ckpt_dir, dataset_name)
        tmp = out.with_suffix(".tmp")
        torch.save(entry, tmp)
        tmp.replace(out)


def load_unified_registry(path: Path) -> dict:
    """Load all per-dataset .pth files from the checkpoint directory.

    Returns a dict keyed by dataset name — identical structure to the old
    monolithic registry so the rest of main() needs no changes.

    Files that are absent, empty, or corrupt are silently skipped so a fresh
    run on a new dataset is handled gracefully.  Architecture-mismatched files
    are also skipped here; the mismatch is caught later in main() when
    load_state_dict is called, and that dataset simply restarts from scratch
    with the current architecture written back on the next save.
    """
    ckpt_dir = path.parent
    registry = {}
    for dataset_name in DATASETS:
        file = _ckpt_path(ckpt_dir, dataset_name)
        if not file.exists():
            continue
        try:
            entry = torch.load(file, map_location=DEVICE, weights_only=False)
            registry[dataset_name] = entry
        except Exception as exc:
            print(f"  ⚠️  Could not load {file.name} ({exc}) — will restart fresh.")
    return registry


# ══════════════════════════════════════════════════════════════════════════════
# COMPLEXITY SCORING
# ══════════════════════════════════════════════════════════════════════════════
#
# Each dataset is scored on four independent axes (each 0–10).
# The axes capture different kinds of difficulty that affect different
# hyperparameters, so they are kept separate rather than collapsed to one
# number immediately.
#
# Axis 1 — BOUNDARY FREQUENCY (bf)
#   How many direction changes per unit length does the decision boundary
#   have?  Low = one straight or gently curved line.  High = many tight
#   oscillations or interleaved spirals.
#   Drives: Fourier mapping_size and scale.  Higher frequency needs higher
#   scale (more high-frequency random projections) and more mapping neurons
#   to represent it without aliasing.
#
# Axis 2 — BOUNDARY SHARPNESS (bs)
#   Is the boundary a hard geometric edge (XOR, Checkerboard) or a soft
#   overlapping region (OverlappingGaussians, GaussianBlobs)?
#   Hard edges require the model to commit to a precise boundary — they
#   benefit from higher scale but also stronger regularisation to avoid
#   overconfident memorisation.
#   Drives: weight_decay (harder edge → stronger wd), label_smoothing.
#
# Axis 3 — TOPOLOGICAL COMPLEXITY (tc)
#   Is the boundary simply connected (one line) or multiply connected
#   (rings, spirals that wind around)?  Multiply connected boundaries
#   require the hidden layers to maintain multiple simultaneous "loops"
#   of representation.
#   Drives: hidden_dim and n_blocks.  More topology = deeper/wider network.
#
# Axis 4 — CLASS DENSITY PRESSURE (cd)
#   With n_classes > 2, how much do classes crowd each other?  A 6-class
#   wedge pattern is coarser (large angular gap) than a 20-class wedge
#   (tiny gap = high density pressure).  This is partly captured by
#   n_classes but the *geometry* of crowding matters too — concentric rings
#   pack radially, wedges pack angularly, blobs are placed by sklearn with
#   variable margins.
#   Drives: all params but especially lr (lower when crowded to avoid
#   overshooting decision margins) and hidden_dim.
#
# Final composite score = weighted sum → bucketed → (LOW | MEDIUM | HIGH)
#   LOW    [0.0, 3.5) — simple, smooth, well-separated
#   MEDIUM [3.5, 6.5) — moderate curvature or class count
#   HIGH   [6.5, 10]  — complex topology, high frequency, or many classes
#
# Scores are hand-assigned per dataset using geometric reasoning, then
# verified against expected kNN-5 accuracy (proxy for learnability):
#   LOW    → kNN-5 ≥ 90% typically achievable
#   MEDIUM → kNN-5 ≈ 75–90%
#   HIGH   → kNN-5 < 75% without a strong model
#
# ── Score table ───────────────────────────────────────────────────────────────
#
#  Dataset               bf    bs    tc    cd    composite   bucket
#  ─────────────────────────────────────────────────────────────────
#  Circles                2     3     4     0      2.6        LOW
#  TwoMoons               3     3     3     0      2.8        LOW
#  DiagonalLinear         1     4     1     0      1.8        LOW
#  OverlappingGaussians   1     2     1     0      1.3        LOW
#  GaussianBlobs_6        2     3     2     2      2.4        LOW
#  GridGaussians_6        2     3     2     2      2.4        LOW
#  GaussianBlobs_10       2     3     2     4      2.9        LOW
#  GridGaussians_10       2     3     2     4      2.9        LOW
#  GaussianBlobs_14       2     3     2     6      3.3        LOW
#  ─────────────────────────────────────────────────────────────────
#  XOR                    4     7     2     0      3.8        MEDIUM
#  SineBoundary           5     5     2     0      3.7        MEDIUM
#  ConcentricRings_6      4     6     5     2      4.3        MEDIUM
#  AngularWedges_6        3     6     3     2      3.7        MEDIUM
#  MultiArmSpiral_6       6     5     6     2      4.9        MEDIUM
#  ConcentricRings_10     5     6     6     4      5.3        MEDIUM
#  AngularWedges_10       4     6     4     4      4.6        MEDIUM
#  MultiArmSpiral_10      7     5     7     4      5.9        MEDIUM
#  GridGaussians_14       3     3     3     6      3.9        MEDIUM
#  GaussianBlobs_20       2     3     2     8      3.8        MEDIUM
#  GridGaussians_20       3     3     3     8      4.3        MEDIUM
#  AngularWedges_14       5     6     5     6      5.5        MEDIUM
#  ConcentricRings_14     6     6     7     6      6.2        MEDIUM
#  ─────────────────────────────────────────────────────────────────
#  TwoSpirals             8     6     8     0      6.7        HIGH
#  AngularWedges_20       6     7     6     8      6.8        HIGH
#  ConcentricRings_20     7     6     8     8      7.2        HIGH
#  MultiArmSpiral_14      8     5     8     6      6.9        HIGH
#  MultiArmSpiral_20      9     5     9     8      7.9        HIGH
#
# composite = 0.30*bf + 0.25*bs + 0.30*tc + 0.15*cd
# ─────────────────────────────────────────────────────────────────────────────

# Weights for composite score
# ── Composite score weights ────────────────────────────────────────────────────
_W = dict(bf=0.30, bs=0.25, tc=0.30, cd=0.15)

def _composite(bf, bs, tc, cd) -> float:
    return _W["bf"]*bf + _W["bs"]*bs + _W["tc"]*tc + _W["cd"]*cd

# Per-dataset complexity profiles
# fmt: (boundary_freq, boundary_sharpness, topological_complexity,
#        class_density, separability)
#
# separability (sep) — how cleanly separated the class regions are (1=very
#   clean/well-separated, 10=heavily overlapping).
#   GaussianBlobs and GridGaussians: sklearn places centers with large margins
#   → sep=1 (trivially memorisable — needs smallest model).
#   Circles, TwoMoons: clean but non-linear → sep=3.
#   Spirals, ConcentricRings with many arms: regions interleave → sep=6–8.
#   Drives a capacity scale-down factor: small sep → less hidden_dim and
#   mapping_size, more dropout and weight_decay.
#
#  Dataset               bf  bs  tc  cd  sep
_COMPLEXITY_PROFILES: dict[str, tuple[int,int,int,int,int]] = {
    # 2-class
    "Circles":              (2, 3, 4, 0, 3),
    "TwoMoons":             (3, 3, 3, 0, 3),
    "XOR":                  (4, 7, 2, 0, 5),
    "SineBoundary":         (5, 5, 2, 0, 4),
    "TwoSpirals":           (8, 6, 8, 0, 7),
    # 6-class
    "ConcentricRings_6":    (4, 6, 5, 2, 5),
    "AngularWedges_6":      (3, 6, 3, 2, 4),
    "GaussianBlobs_6":      (2, 3, 2, 2, 1),
    "MultiArmSpiral_6":     (6, 5, 6, 2, 6),
    "GridGaussians_6":      (2, 3, 2, 2, 1),
    # 10-class
    "ConcentricRings_10":   (5, 6, 6, 4, 6),
    "AngularWedges_10":     (4, 6, 4, 4, 5),
    "GaussianBlobs_10":     (2, 3, 2, 4, 1),
    "MultiArmSpiral_10":    (7, 5, 7, 4, 7),
    "GridGaussians_10":     (2, 3, 2, 4, 1),
    # 14-class
    "ConcentricRings_14":   (6, 6, 7, 6, 7),
    "AngularWedges_14":     (5, 6, 5, 6, 6),
    "GaussianBlobs_14":     (2, 3, 2, 6, 1),
    "MultiArmSpiral_14":    (8, 5, 8, 6, 8),
    "GridGaussians_14":     (3, 3, 3, 6, 1),
    # 20-class
    "ConcentricRings_20":   (7, 6, 8, 8, 8),
    "AngularWedges_20":     (6, 7, 6, 8, 7),
    "GaussianBlobs_20":     (2, 3, 2, 8, 1),
    "MultiArmSpiral_20":    (9, 5, 9, 8, 9),
    "GridGaussians_20":     (3, 3, 3, 8, 1),
}

_LOW_MAX    = 3.5
_MEDIUM_MAX = 6.5

# ── Per-dataset render extents ────────────────────────────────────────────────
# Half-width of the square canvas in world coordinates.
# Computed from each generator's actual data range (scale + noise*3) × 1.10.
# Keeps the canvas tight around where training data exists so the model is
# not asked to classify regions far outside its training distribution.
RENDER_EXTENTS: dict[str, float] = {
    "TwoSpirals":              6.66,
    "Circles":                 1.49,
    "TwoMoons":                2.34,
    "XOR":                     3.63,
    "SineBoundary":            6.00,
    "ConcentricRings_6":       6.89,
    "AngularWedges_6":         5.76,
    "GaussianBlobs_6":         5.50,
    "MultiArmSpiral_6":        9.00,
    "GridGaussians_6":         3.82,
    "ConcentricRings_10":     11.51,
    "AngularWedges_10":        5.76,
    "GaussianBlobs_10":        7.50,
    "MultiArmSpiral_10":       9.00,
    "GridGaussians_10":        5.08,
    "ConcentricRings_14":     16.13,
    "AngularWedges_14":        5.76,
    "GaussianBlobs_14":       10.00,
    "MultiArmSpiral_14":       9.00,
    "GridGaussians_14":        5.08,
    "ConcentricRings_20":     23.06,
    "AngularWedges_20":        5.76,
    "GaussianBlobs_20":       13.00,
    "MultiArmSpiral_20":       9.00,
    "GridGaussians_20":        6.35,
}


def get_complexity(dataset_name: str) -> dict:
    """
    Return complexity axes, composite score, and bucket for *dataset_name*.
    """
    bf, bs, tc, cd, sep = _COMPLEXITY_PROFILES[dataset_name]
    score = _composite(bf, bs, tc, cd)
    if score < _LOW_MAX:
        bucket = "LOW"
    elif score < _MEDIUM_MAX:
        bucket = "MEDIUM"
    else:
        bucket = "HIGH"
    return dict(bf=bf, bs=bs, tc=tc, cd=cd, sep=sep,
                score=round(score, 2), bucket=bucket)


# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER SELECTION
# ══════════════════════════════════════════════════════════════════════════════
#
# Two-stage sizing:
#
# Stage 1 — base capacity from bucket × boundary axes
#   mapping_size: min 64, grows with bf (more oscillations = more features)
#   hidden_dim:   min 128, grows with tc and cd
#   n_blocks:     0/1/2 by bucket and tc
#
# Stage 2 — separability correction (anti-memorisation)
#   sep=1 (blobs, grids): trivially separable → scale capacity DOWN sharply
#     so the model cannot memorise the clean training boundary.
#     capacity_factor = 0.50 → half the neurons, more wd, more dropout.
#   sep=3–5: moderate → small reduction (factor 0.75–0.85)
#   sep=6–9: genuinely hard → full capacity (factor 1.0)
#
#   This is the key lever that prevents 99–100% accuracy on easy datasets
#   while still giving hard datasets enough capacity to converge.
#
# weight_decay grows as sep decreases (easy datasets → stronger L2 to prevent
#   the model memorising the clean boundary).
# dropout grows as sep decreases for the same reason.
# ─────────────────────────────────────────────────────────────────────────────

def get_hparams(dataset_name: str, num_classes: int) -> dict:
    cx = get_complexity(dataset_name)
    bf, bs, tc, cd, sep = cx["bf"], cx["bs"], cx["tc"], cx["cd"], cx["sep"]
    bucket = cx["bucket"]

    # ── Stage 1: base capacity ─────────────────────────────────────────────────

    _ms_base = {
        "LOW":    64  + bf * 4,
        "MEDIUM": 96  + bf * 8,
        "HIGH":   128 + bf * 10,
    }[bucket]
    mapping_size_base = _ms_base + ((num_classes - 2) // 5) * 8
    mapping_size_base = max(64, (mapping_size_base // 32) * 32)

    _scale_from_bf = 1.5 + bf * 0.40
    _bs_bonus      = (bs - 5) * 0.12 if bs > 5 else 0.0
    scale = round(min(_scale_from_bf + _bs_bonus, 5.0), 2)
    scale = float(max(1.5, scale))

    _hd_base = {
        "LOW":    128 + tc * 8  + cd * 4,
        "MEDIUM": 192 + tc * 12 + cd * 6,
        "HIGH":   256 + tc * 16 + cd * 8,
    }[bucket]
    hidden_dim_base = max(128, max(num_classes * 8,
                                   ((_hd_base + 32) // 64) * 64))
    hidden_dim_base = min(hidden_dim_base, 512)

    if bucket == "LOW":
        n_blocks = 0 if tc <= 3 else 1
    elif bucket == "MEDIUM":
        n_blocks = 1
    else:
        n_blocks = 2

    # ── Stage 2: separability correction ──────────────────────────────────────
    # sep=1 → very easy → capacity_factor=0.50, strong regularisation
    # sep=5 → moderate  → capacity_factor=0.80
    # sep=9 → hard      → capacity_factor=1.00
    # Linear interpolation between these anchor points.
    if sep <= 1:
        cap_factor = 0.50
    elif sep >= 9:
        cap_factor = 1.00
    else:
        cap_factor = 0.50 + (sep - 1) * (0.50 / 8)   # 0.50 → 1.00 over sep 1–9

    # Apply factor to width (snap to multiples of 32 / 64)
    mapping_size = max(64, int(mapping_size_base * cap_factor // 32) * 32)
    hidden_dim   = max(128, int(hidden_dim_base  * cap_factor // 64) * 64)

    # n_blocks: drop one block for very easy datasets to reduce depth capacity
    if sep <= 2 and n_blocks > 0:
        n_blocks = max(0, n_blocks - 1)

    # ── dropout ────────────────────────────────────────────────────────────────
    # Base dropout, then increase for easy/separable datasets
    _drop_base = {"LOW": 0.10, "MEDIUM": 0.15, "HIGH": 0.20}[bucket]
    _drop_sep_bonus = max(0.0, (5 - sep) * 0.03)   # +0.03 per sep step below 5
    dropout = round(min(_drop_base + _drop_sep_bonus, 0.35), 3)

    # ── learning rate ──────────────────────────────────────────────────────────
    _lr_base = {"LOW": 3e-3, "MEDIUM": 2e-3, "HIGH": 1.5e-3}[bucket]
    _lr_cd_penalty = 5e-4 if cd >= 6 else 0.0
    lr = float(_lr_base - _lr_cd_penalty)

    # ── weight decay ───────────────────────────────────────────────────────────
    # Base wd, then increase for easy datasets (prevent boundary memorisation)
    _wd_base = {"LOW": 1e-3, "MEDIUM": 3e-3, "HIGH": 5e-3}[bucket]
    _wd_bs_bonus  = 2e-3 if bs >= 7 else 0.0
    _wd_sep_bonus = max(0.0, (5 - sep) * 2e-3)   # +2e-3 per sep step below 5
    wd = float(_wd_base + _wd_bs_bonus + _wd_sep_bonus)

    # ── label smoothing ────────────────────────────────────────────────────────
    label_smoothing = 0.10 if num_classes >= 10 else 0.05

    # ── max epochs ────────────────────────────────────────────────────────────
    max_epochs = {"LOW": 2000, "MEDIUM": 3000, "HIGH": 4000}[bucket]

    render_extent = RENDER_EXTENTS.get(dataset_name, 6.0)

    return dict(
        hidden_dim            = hidden_dim,
        mapping_size          = mapping_size,
        scale                 = scale,
        n_blocks              = n_blocks,
        dropout               = dropout,
        lr                    = lr,
        wd                    = wd,
        label_smoothing       = label_smoothing,
        max_epochs            = max_epochs,
        complexity_score      = cx["score"],
        complexity_bucket     = bucket,
        separability          = sep,
        capacity_factor       = round(cap_factor, 2),
        render_extent         = render_extent,
    )


def fresh_eval(model, dataset: _LazyDataset, X_mean, X_std,
               n_samples: int = 2000) -> float:
    """Evaluate *model* on freshly generated data (different seed from training).

    Returns accuracy (0–100).  A model that memorised the fixed training seed
    will score noticeably lower here than on the held-out test set.
    """
    X_fresh, y_fresh = dataset.fresh_eval_data(n_samples=n_samples)
    device = next(model.parameters()).device
    X_fresh_n = ((X_fresh - X_mean) / X_std).to(device)
    y_fresh   = y_fresh.to(device)
    model.eval()
    with torch.inference_mode():
        preds = model(X_fresh_n).argmax(1)
    acc = (preds == y_fresh).float().mean().item() * 100
    model.train()
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def main():

    ckpt_dir         = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    master_ckpt_path = ckpt_dir / "unified_checkpoint.pth"
    _EPOCHS_PER_RUN  = EPOCHS_PER_RUN

    print(f"\n  RUN_SEED = {RUN_SEED}  (train/test split and generator noise vary each launch)")

    master_registry = load_unified_registry(master_ckpt_path)
    if master_registry:
        print(f"Loaded master registry from {master_ckpt_path} "
              f"({len(master_registry)} dataset(s) tracked).")
    else:
        print("No existing master registry found — starting fresh.")

    # Print complexity summary on first run so it's easy to audit
    print("\n  Dataset complexity + hparam summary:")
    print(f"  {'Dataset':<24} {'cls':>4} {'sep':>4} {'cap':>5} {'Bucket':>7}  "
          f"{'hd':>5} {'ms':>4} {'sc':>5} {'blk':>4} {'do':>5} {'wd':>7}")
    print("  " + "─" * 100)
    for name in DATASETS:
        ds   = DATASETS[name]
        X_tr, y_tr, X_te, y_te = ds
        nc   = int(torch.unique(torch.cat([y_tr, y_te])).numel())
        hp   = get_hparams(name, nc)
        print(f"  {name:<24} {nc:>4} {hp['separability']:>4} {hp['capacity_factor']:>5.2f}"
              f" {hp['complexity_bucket']:>7}  "
              f"{hp['hidden_dim']:>5} {hp['mapping_size']:>4} {hp['scale']:>5.2f} "
              f"{hp['n_blocks']:>4} {hp['dropout']:>5.2f} {hp['wd']:>7.1e}")
    print()

    for run in range(1, 16):   # 15 outer passes — all datasets train every run

        
        print(f" ==================================================\n"
          f"========================={run}=====================\n"
          f"===================================================\n")
    

        for dataset_name, dataset in DATASETS.items():
            entry      = master_registry.get(dataset_name, {})
            prior_best = entry.get("best_acc",   0.0)
            prior_fresh= entry.get("fresh_acc",  0.0)
            prior_ep   = entry.get("epoch",      0)

            X_tr, y_tr, X_te, y_te = dataset
            num_classes = int(torch.unique(torch.cat([y_tr, y_te])).numel())
            hp          = get_hparams(dataset_name, num_classes)

            max_epochs  = hp["max_epochs"]


            model = FourierMLP(
                output_dim   = num_classes,
                hidden_dim   = hp["hidden_dim"],
                dropout      = hp["dropout"],
                scale        = hp["scale"],
                mapping_size = hp["mapping_size"],
                n_blocks     = hp["n_blocks"],
            ).to(DEVICE)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=hp["lr"], weight_decay=hp["wd"],
            )

            start_epoch = 0
            best_acc    = 0.0
            status      = "fresh start"
            if entry:
                try:
                    model.load_state_dict(entry["model_state_dict"])
                    optimizer.load_state_dict(entry["optimizer_state_dict"])
                    start_epoch = entry["epoch"]
                    best_acc    = entry.get("best_acc", 0.0)
                    status      = f"resuming ep {start_epoch}"
                except RuntimeError:
                    print(f"  ⚠️  Architecture mismatch for {dataset_name} — restarting fresh.")

            target_epoch  = min(start_epoch + _EPOCHS_PER_RUN, max_epochs)
            iteration     = (start_epoch // _EPOCHS_PER_RUN) + 1

            print(f"\n── {dataset_name}  [{hp['complexity_bucket']} | score={hp['complexity_score']}]"
                  f"  {num_classes} classes | run {run} iter {iteration}"
                  f"  ep {start_epoch}→{target_epoch} | {status}")
            print(f"   hparams: hd={hp['hidden_dim']} ms={hp['mapping_size']} "
                  f"scale={hp['scale']} blk={hp['n_blocks']} "
                  f"lr={hp['lr']:.1e} wd={hp['wd']:.1e} "
                  f"ls={hp['label_smoothing']}")

            loss, train_time, X_mean, X_std = run_training_loop(
                model            = model,
                num_classes      = num_classes,
                optimizer        = optimizer,
                dataset_name     = dataset_name,
                X_train          = X_tr,
                X_test           = X_te,
                y_train          = y_tr,
                y_test           = y_te,
                start_epoch      = start_epoch,
                total_epochs     = target_epoch,
                label_smoothing  = hp["label_smoothing"],
            )

            torch.manual_seed(RANDOM_SEED)
            results     = eval_model(model=model, X_test=X_te, y_test=y_te,
                                     accuracy_fn=accuracy_fn,
                                     X_mean=X_mean, X_std=X_std)
            current_acc = results["accuracy"]
            best_acc    = max(current_acc, best_acc)

            # ── Deployment validation: fresh data with a different seed ─────────
            fresh_acc = fresh_eval(model, dataset, X_mean, X_std, n_samples=2000)
            print(f"   ✓ {train_time:.1f}s | acc={current_acc:.2f}% | best={best_acc:.2f}%")
            print(f"   🔍 fresh_eval={fresh_acc:.2f}%  (unseen noise realisation)")

            master_registry[dataset_name] = {
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch":                target_epoch,
                "loss":                 loss,
                "best_acc":             best_acc,
                "fresh_acc":            fresh_acc,   # deployment proxy — tracked across runs
                "X_mean":               X_mean.cpu(),
                "X_std":                X_std.cpu(),
                "model_config": {
                    "output_dim":   num_classes,
                    "hidden_dim":   hp["hidden_dim"],
                    "dropout":      hp["dropout"],
                    "scale":        hp["scale"],
                    "mapping_size": hp["mapping_size"],
                    "n_blocks":     hp["n_blocks"],
                },
                "complexity": {
                    "score":  hp["complexity_score"],
                    "bucket": hp["complexity_bucket"],
                },
                "render_extent": hp["render_extent"],
            }
            save_unified_registry(master_registry, master_ckpt_path)



if __name__ == "__main__":
    main()