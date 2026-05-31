# PyTorch Learning Projects

A series of neural network projects built while learning PyTorch from the ground up — starting from simple linear regression and progressing through binary and multiclass classification.

---

## Projects Overview

| #   | Project                   | Type           | Key Concept                                          |
| --- | ------------------------- | -------------- | ---------------------------------------------------- |
| 01  | Linear Regression         | Regression     | `nn.Linear`, SGD, MAE loss                           |
| 02  | Binary Classification     | Classification | `BCEWithLogitsLoss`, sigmoid, circles data           |
| 03  | Multiclass Classification | Classification | `CrossEntropyLoss`, softmax, blob data               |
| 04  | Spiral Classification     | Classification | Dropout, checkpointing, resume training              |
| 05  | XOR Grid Classification   | Classification | Custom dataset, deep MLP, LR scheduling, checkpoints |
| 06  | Crescent Classification   | Classification | `make_moons`, 20-class custom generator, timer       |

---

## Project Structure

```
pytorch-learning/
├── README.md
├── requirements.txt
├── 01_linear_regression/
│   └── linear_model.py
├── 02_binary_classification/
│   └── non_linear_binary.py
├── 03_multiclass_classification/
│   └── multiclass_classification_model.py
├── 04_spiral_classification/
│   └── first_model.py
├── 05_xor_grid_classification/
│   ├── xor_classification.py
│   └── helper_function.py
└── 06_crescent_classification/
    ├── crescent_classification.py
    └── helper_function.py
```

---

## Requirements

```
torch
matplotlib
scikit-learn
torchmetrics
numpy
```

Install all at once:

```bash
pip install torch matplotlib scikit-learn torchmetrics numpy
```

---

## 01 — Linear Regression

**File:** `01_linear_regression/linear_model.py`

The simplest possible neural network — a single `nn.Linear` layer that learns to fit a straight line. Synthetic data is generated using a known equation (`y = 0.95x + 0.45`) so we can directly check if the model learned the correct weight and bias.

**Model:** 1 linear layer — 1 input, 1 output  
**Loss:** L1Loss (Mean Absolute Error)  
**Optimizer:** SGD  
**Epochs:** 300

**What it covers:**

- Building a model with `nn.Module`
- The full training loop (forward → loss → backward → step)
- Saving and loading model weights with `state_dict()`
- Plotting predictions and loss curve

**How to run:**

```bash
python 01_linear_regression/linear_model.py
```

---

## 02 — Non-Linear Binary Classification

**File:** `02_binary_classification/non_linear_binary.py`

Trains a model to separate two concentric circles — a problem a straight line can never solve, so non-linear activation functions (ReLU) are required. Uses `make_circles` from scikit-learn.

**Model:** 3 linear layers with ReLU activations  
**Loss:** BCEWithLogitsLoss (includes sigmoid internally)  
**Optimizer:** Adam  
**Epochs:** 10,000

**What it covers:**

- Why non-linearity is needed for real-world data
- Binary classification with `BCEWithLogitsLoss`
- Converting logits → probabilities → labels via sigmoid
- Gradient clipping with `clip_grad_norm_`
- Plotting decision boundaries

**How to run:**

```bash
python 02_binary_classification/non_linear_binary.py
```

---

## 03 — Multiclass Classification

**File:** `03_multiclass_classification/multiclass_classification_model.py`

Classifies data points into 10 distinct groups using blob data from scikit-learn. Extends binary classification to multiple classes using softmax and CrossEntropyLoss.

**Model:** 4 linear layers with ReLU activations  
**Loss:** CrossEntropyLoss  
**Optimizer:** Adam  
**Epochs:** 5,000  
**Classes:** 10 | **Features:** 5

**What it covers:**

- Multiclass classification with `CrossEntropyLoss`
- Converting logits → probabilities via `softmax`
- Using `torchmetrics` Accuracy for evaluation
- Visualizing true vs predicted labels

**How to run:**

```bash
python 03_multiclass_classification/multiclass_classification_model.py
```

---

## 04 — Spiral Multiclass Classification

**File:** `04_spiral_classification/first_model.py`

The most complex project — classifies 5-class spiral data, which is much harder to separate than blobs. Introduces dropout regularization, learning rate scheduling, and a full checkpoint save/resume system so training can be paused and continued across sessions.

**Model:** 5 linear layers with ReLU + Dropout(0.2)  
**Loss:** CrossEntropyLoss  
**Optimizer:** Adam with `ReduceLROnPlateau` scheduler  
**Epochs:** 1,000 per session (resumes indefinitely)  
**Classes:** 5

**What it covers:**

- Dropout for regularization
- Learning rate scheduling with `ReduceLROnPlateau`
- Saving and loading full training checkpoints (model + optimizer state)
- Resuming training from a saved checkpoint
- Generating spiral data from scratch with NumPy

**How to run:**

```bash
python 04_spiral_classification/first_model.py
```

After the first 1,000 epochs it will ask `Train ahead (y/n)` — enter `y` to continue training or `n` to stop.

---

## 05 — XOR Grid Multiclass Classification

**File:** `05_xor_grid_classification/xor_classification.py`

The most challenging dataset in the series — a fully custom-built 3 × 3 grid of XOR-patterned clusters producing 9 classes. Within each grid cell, the class boundary follows a checkerboard rule (positive in quadrants 1 & 3, negative in 2 & 4), with Gaussian noise added to prevent the model from exploiting clean axis-aligned boundaries. No library generates this data; it is written from scratch with NumPy.

Builds on everything from project 04 — dropout, LR scheduling, and checkpoint resuming — and adds a deeper network to handle the more complex decision boundaries.

**Model:** 8 linear layers with ReLU + Dropout(0.2)  
**Loss:** CrossEntropyLoss  
**Optimizer:** Adam with `ReduceLROnPlateau` scheduler  
**Epochs:** 1,000 per session (resumes indefinitely)  
**Classes:** 9 | **Features:** 2

```
Input (2) → Linear(2 → 64)   → ReLU
          → Linear(64 → 256)  → ReLU → Dropout(0.2)
          → Linear(256 → 256) → ReLU → Dropout(0.2)  ×3
          → Linear(256 → 64)  → ReLU → Dropout(0.2)
          → Linear(64 → 64)   → ReLU
          → Linear(64 → 9)    → Logits
```

**What it covers:**

- Building a fully custom dataset generator with NumPy
- Classifying data with interleaved, non-convex XOR boundaries
- Deeper MLP design for harder decision surfaces
- `ReduceLROnPlateau` learning rate scheduling
- Full checkpoint system — saves model weights, optimiser state, epoch, and loss
- Resuming training seamlessly across multiple sessions
- Side-by-side decision boundary plots (ground truth vs predictions)

**How to run:**

```bash
python 05_xor_grid_classification/xor_classification.py
```

On first run the model trains for 1,000 epochs and saves a checkpoint. On every subsequent run it resumes automatically from where it left off. After each session it will ask `Train for another 1 000 epochs? (y/n)` — enter `y` to continue or `n` to stop.

**Checkpoint location:** `05_xor_grid_classification/xor_model/xor_model.pth`

---

## 06 — Crescent Multiclass Classification

**File:** `06_crescent_classification/crescent_classification.py`

Extends the half-moon idea from scikit-learn's `make_moons` into a 20-class multiclass problem. A custom generator stacks 10 crescent pairs along the x-axis, remaps their local {0, 1} labels to global class ids, centres the dataset at the origin, and shuffles before returning. Each pair gets its own random seed so no two crescents share identical geometry.

Builds on everything from project 05 — dropout, LR scheduling, and checkpoint resuming — and introduces a wall-clock training timer that prints live elapsed time and ETA at every log step, plus a per-iteration summary on completion.

**Model:** 6 linear layers with ReLU + Dropout(0.2)  
**Loss:** CrossEntropyLoss  
**Optimizer:** Adam with `ReduceLROnPlateau` scheduler  
**Epochs:** 1,000 per session (resumes indefinitely)  
**Classes:** 20 | **Features:** 2

```
Input (2) → Linear(2 → 64)   → ReLU
          → Linear(64 → 256)  → ReLU → Dropout(0.2)
          → Linear(256 → 256) → ReLU → Dropout(0.2)
          → Linear(256 → 256) → ReLU → Dropout(0.2)
          → Linear(256 → 64)  → ReLU
          → Linear(64 → 20)   → Logits
```

**What it covers:**

- Extending `make_moons` into an arbitrary-class crescent generator
- Pair-wise x-axis layout with global label remapping
- Odd-class guard — cleanly handles any `n_crescents` value including odd numbers
- Wall-clock timer using `time.perf_counter` — live elapsed + ETA per log step
- Per-iteration summary: total train time, time per epoch, full iteration time
- Full checkpoint system — saves model weights, optimiser state, epoch, and loss
- Resuming training seamlessly across multiple sessions
- Side-by-side decision boundary plots (ground truth vs predictions)

**How to run:**

```bash
python 06_crescent_classification/crescent_classification.py
```

On first run the model trains for 1,000 epochs and saves a checkpoint. On every subsequent run it resumes automatically from where it left off. After each session it will ask `Train for another 1 000 epochs? (y/n)` — enter `y` to continue or `n` to stop.

**Checkpoint location:** `06_crescent_classification/Crescents_model/Crescents_model.pth`

---

## What I learned across this series

- How neural networks learn through gradient descent and backpropagation
- The difference between regression and classification problems
- When and why to use ReLU, sigmoid, and softmax activations
- How loss functions differ: L1, BCE, CrossEntropy — and when to use each
- How to prevent overfitting using Dropout and gradient clipping
- How to save, load, and resume model training with checkpoints
- How to design and generate custom datasets for non-standard problems
- How deeper networks handle more complex, non-convex decision boundaries
- How to extend a two-class sklearn generator into an arbitrary multiclass dataset
- How to measure and log training time with live ETA estimates
