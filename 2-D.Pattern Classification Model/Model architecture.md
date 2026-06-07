# Model Architecture — FourierMLP

## Overview

`FourierMLP` is a compact neural network designed to learn non-linear 2D decision boundaries without memorising the training data. It is built around three ideas: fixed random Fourier features to lift coordinates into a high-frequency space, a bottleneck residual structure to regularise the mid-network representation, and a complexity-aware hyperparameter system that scales capacity to the actual difficulty of each dataset.

---

## Layer-by-Layer Walkthrough

```
Input: (N, 2)  — normalised (x, y) coordinates
    │
    ▼
FourierFeatureMapping            → (N, 2M)
    │
    ▼
Linear(2M → hidden_dim)          → (N, hidden_dim)
GELU
Dropout(p)
    │
    ▼
ResidualBlock × n_blocks         → (N, hidden_dim)   [0, 1, or 2 blocks]
    │
    ▼
Linear(hidden_dim → output_dim)  → (N, num_classes)  ← logits, no softmax
```

---

## FourierFeatureMapping

```python
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, mapping_size, scale):
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer("B", B)   # fixed — never trained

    def forward(self, x):
        proj = x @ self.B                                     # (N, M)
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2M)
        return feat
```

**What it does:** Projects the 2D input into a 2M-dimensional space of sinusoidal features. The matrix `B` is randomly initialised once and frozen — it never receives gradients.

**Why it works:** Plain MLPs are spectrally biased toward learning low-frequency functions first (the "spectral bias" of neural networks). For datasets like TwoSpirals or ConcentricRings_20, the decision boundary has high spatial frequency that a plain MLP struggles to learn without overfitting. The Fourier mapping lifts the input into a space where those high-frequency boundaries become linearly separable, allowing the downstream layers to solve a much simpler problem.

**The `scale` hyperparameter** controls the frequency range of the random projections. A higher scale means `B` contains larger values, which means the dot products `x @ B` span a wider range, mapping to higher spatial frequencies. Each dataset gets a `scale` tuned to its boundary complexity — TwoSpirals (MEDIUM) uses `scale=4.82`, while GridGaussians_6 (LOW) uses `scale=2.3`.

**Critical:** The forward pass contains no `2π` multiplier. The formula is exactly `proj = x @ B`, then `[sin(proj), cos(proj)]`. The JavaScript forward pass must match this exactly — adding `2π` would completely destroy the correspondence between training and inference.

---

## ResidualBlock

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        neck = max(dim // 2, 16)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, neck)    # compress
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)     # only at bottleneck
        self.up   = nn.Linear(neck, dim)    # expand

    def forward(self, x):
        h = self.norm(x)    # Pre-LN: normalise before transform
        h = self.down(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.up(h)
        return x + h        # residual addition
```

**Bottleneck design:** Each block compresses `hidden_dim → hidden_dim//2 → hidden_dim`. The narrow middle forces the block to distil information rather than pass it through unmodified, acting as a built-in regulariser. Dropout is applied only at the narrowest point — the most information-dense location — rather than everywhere.

**Pre-LN (Pre-LayerNorm):** LayerNorm is applied before the transform (`h = self.norm(x)`), not after the residual addition. This gives more stable gradients during training, especially with lower learning rates, and avoids the gradient vanishing problem that Post-LN suffers on deeper stacks.

**Residual connection:** `return x + h` lets the block learn only the _correction_ to the current representation rather than the full transform from scratch. This makes training more stable and allows gradients to flow back directly through the skip connection.

---

## FourierMLP (Full Model)

```python
class FourierMLP(nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout, scale,
                 mapping_size=48, n_blocks=1, input_dim=2):
        self.fourier     = FourierFeatureMapping(input_dim, mapping_size, scale)
        self.input_layer = nn.Sequential(
            nn.Linear(mapping_size * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.input_layer(self.fourier(x))
        for block in self.blocks:
            h = block(h)
        return self.head(h)   # raw logits
```

The output head maps directly from `hidden_dim` to `output_dim` with a single linear layer. There is no intermediate bottleneck before the head — removing it cuts ~30% of parameters and forces the residual blocks to carry the representational load rather than an extra MLP stage.

---

## Complexity-Aware Hyperparameter System

Every dataset is characterised by five axes before training:

| Axis                          | Range | Meaning                                    |
| ----------------------------- | ----- | ------------------------------------------ |
| `boundary_freq` (bf)          | 1–9   | Spatial frequency of the decision boundary |
| `boundary_sharpness` (bs)     | 1–9   | How crisp the class transitions are        |
| `topological_complexity` (tc) | 1–9   | Number of disconnected class regions       |
| `class_density` (cd)          | 1–9   | How many classes compete locally           |
| `separability` (sep)          | 1–9   | How linearly separable the classes are     |

The **complexity score** is a weighted sum: `0.30·bf + 0.20·bs + 0.20·tc + 0.15·cd + 0.15·(10-sep)`. Score ≤3.5 → LOW, ≤6.0 → MEDIUM, else HIGH.

### Hyperparameter scaling from complexity

```
Bucket    hidden_dim    mapping_size    n_blocks    dropout    lr
LOW       128           64              0–1         0.10+      3e-3
MEDIUM    192–256       96–128          1           0.15+      2e-3
HIGH      256–320       128–192         2           0.20+      1.5e-3
```

The `sep` axis applies additional corrections: easy datasets (sep=1) get `capacity_factor=0.50` which halves hidden_dim and mapping_size, and gains extra weight decay (`+6e-3`) to prevent memorisation of trivially separable clusters.

### Parameter counts by bucket

| Bucket           | Approx. params |
| ---------------- | -------------- |
| LOW / 2-class    | ~15k–30k       |
| MEDIUM / 6-class | ~40k–80k       |
| HIGH / 20-class  | ~100k–200k     |

This is intentionally small — the goal is generalisation, not maximum accuracy.

---

## Training Loop

```
Optimiser:   AdamW (lr per bucket, weight_decay per dataset)
Scheduler:   CosineAnnealingLR (T_max = epochs in this run, eta_min = 1e-5)
Loss:        CrossEntropyLoss with label_smoothing (0.05 for <10 classes, 0.10 for ≥10)
Batch size:  2048
Grad clip:   max_norm = 1.0
```

Input normalisation is computed from the training split only (`X_mean`, `X_std` per feature dimension) and saved into the checkpoint. The same statistics are applied at inference time by both `server.py` and `app.js`.

### Label smoothing

Prevents the model from becoming over-confident on the training points. With `smoothing=0.05`, the target distribution for the correct class is `1 - 0.05 = 0.95` rather than `1.0`, and the remaining `0.05` is spread uniformly across wrong classes. This is especially important for geometrically clean datasets (GaussianBlobs) where the model would otherwise saturate its outputs.

---

## Anti-Memorisation Design

Three mechanisms work together:

**1. Varied splits across runs.** `RUN_SEED` is randomised every launch. The same 10,000 points are never split the same way twice, so the model cannot memorise which specific points ended up in the test set.

**2. Varied generator noise.** The generator seed also varies, so the actual point positions differ run-to-run. The model must learn the underlying geometric structure (the shape of the spiral, the radius of the ring) rather than the coordinates of specific points.

**3. Fresh eval gate.** After each training block, `fresh_eval()` generates 2000 points with `seed = RUN_SEED + 1000` and scores the model. A gap larger than ~3–4% between test accuracy and fresh eval indicates residual memorisation and is tracked across runs.

---

## Checkpoint Structure

```python
{
    "model_state_dict":     ...,
    "optimizer_state_dict": ...,   # preserves momentum for resuming
    "epoch":                int,
    "loss":                 float,
    "best_acc":             float,
    "fresh_acc":            float,
    "X_mean":               Tensor([2]),
    "X_std":                Tensor([2]),
    "model_config": {
        "output_dim":   int,
        "hidden_dim":   int,
        "dropout":      float,
        "scale":        float,
        "mapping_size": int,
        "n_blocks":     int,
    },
    "complexity": {
        "score":  float,
        "bucket": str,    # "LOW" | "MEDIUM" | "HIGH"
    },
}
```
