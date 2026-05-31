# FashionMNIST Image Classification with PyTorch

Trains a deep MLP on the FashionMNIST dataset — 10 classes of clothing images at 28×28 pixels. Supports checkpoint saving and resuming training across multiple runs.

---

## Usage

```bash
python fashion_mnist.py
```

On each run the script:

1. Downloads (or reuses) the FashionMNIST dataset into `data/`.
2. Resumes from the last checkpoint if one exists, otherwise trains from scratch.
3. Trains for `EPOCHS_PER_RUN` epochs and prints metrics per epoch.
4. Saves a checkpoint, runs a final evaluation, then asks to continue.

---

## Project Structure

```
project/
├── fashion_mnist.py
├── helper_function.py        # must expose accuracy_fn(y_true, y_pred)
├── data/                     # auto-created on first run (FashionMNIST download)
└── FashionMNIST_model/
    └── FashionMNIST_model.pth
```

---

## Requirements

```
torch
torchvision
matplotlib
```

```bash
pip install torch torchvision matplotlib
```

---

## Classes

FashionMNIST has 10 clothing categories:

| Label | Class       |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

---

## Model

**File:** `fashion_mnist.py` → `FashionMNISTModelV0`

Each 28×28 greyscale image is flattened to a 784-dim vector, then passed through a deep MLP.

```
Input (784) → Linear(784 → 512) → ReLU → Dropout(0.2)
            → Linear(512 → 512) → ReLU → Dropout(0.2)
            → Linear(512 → 512) → ReLU → Dropout(0.2)
            → Linear(512 → 512) → ReLU
            → Linear(512 → 10)  → Logits
```

**Loss:** CrossEntropyLoss  
**Optimizer:** Adam (lr=0.001)  
**Scheduler:** ReduceLROnPlateau — halves lr after 100 epochs without test loss improvement  
**Epochs:** `EPOCHS_PER_RUN` per session (resumes indefinitely)  
**Classes:** 10 | **Input features:** 784 (28×28 flattened)

---

## Config

All key constants are at the top of `fashion_mnist.py`:

| Constant         | Default              | Description                      |
| ---------------- | -------------------- | -------------------------------- |
| `DEVICE`         | `"cpu"`              | Training device                  |
| `RANDOM_SEED`    | `42`                 | Seed for reproducibility         |
| `BATCH_SIZE`     | `32`                 | Images per batch                 |
| `EPOCHS_PER_RUN` | `3`                  | Epochs trained per session       |
| `LOG_INTERVAL`   | `400`                | Batch progress print frequency   |
| `CHECKPOINT_DIR` | `FashionMNIST_model` | Folder where checkpoint is saved |

---

## Checkpoint system

On every run the script saves a `.pth` file containing:

- `model_state_dict` — trained weights
- `optimizer` — optimiser state (momentum, adaptive lr)
- `epoch` — last completed epoch
- `loss` — last training loss

On the next run it loads this automatically and resumes from where it left off. If no checkpoint exists it starts from scratch.

**Checkpoint location:** `FashionMNIST_model/FashionMNIST_model.pth`

---

## Output example

```
── Iteration 1  |  epochs 1 → 3 ──

  Epoch:  1
  -----
    looked at 0/60000 samples
    looked at 12800/60000 samples
    ...
  Epoch     1 | Loss: 0.61234  Acc: 78.45% | Test loss: 0.52100  Test acc: 81.20% | Elapsed: 42.1s

  Train time on cpu: 128.443s

  ✓ Iteration 1 done | train: 128.44s | per epoch: 42.8133s | total: 131.20s
  ✓ Checkpoint saved at epoch 3.

  Eval — model: FashionMNISTModelV0 | loss: 0.52100 | acc: 81.20%

Train for another 3 epochs? (y/n):
```

---

## Bugs fixed from original

| #   | Bug                                                                                   | Fix                                                        |
| --- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 1   | `train()` recreated optimizer internally, discarding the loaded checkpoint's state    | Removed internal optimizer — uses the one passed in        |
| 2   | `train()` had no `return` statement — `main()` crashed unpacking `(loss, train_time)` | Added `return loss, total_time`                            |
| 3   | No timer in `train()` — `train_time` was never computed                               | Added `time.perf_counter()` start/end with elapsed logging |
| 4   | `FashionMNISTModelV0()` called with no arguments in `main()`                          | Added `input_shape`, `hidden_units`, `output_shape` args   |
| 5   | `save_checkpoint(loss=loss)` called when `loss` was `None`                            | Fixed by fixing bug 2 — loss now has a valid value         |
| 6   | `range(start_epoch, total_epochs+1)` trained one extra epoch                          | Changed to `range(start_epoch, total_epochs)`              |

---

## What it covers

- Loading and batching image datasets with `torchvision` and `DataLoader`
- Flattening image tensors with `nn.Flatten` for MLP input
- Multiclass image classification with `CrossEntropyLoss`
- Dropout regularization during training, disabled during evaluation
- `torch.inference_mode()` for faster, memory-efficient evaluation
- Wall-clock training timer with elapsed time per epoch
- Full checkpoint system — save and resume across sessions
