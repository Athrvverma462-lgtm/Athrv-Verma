# 2-D Pattern Classification Model

A full-stack machine learning system that trains a neural network to classify 2D geometric patterns and visualises the learned decision boundaries live in a browser. The model runs entirely client-side at inference time — no Python needed after training.

---

## What It Does

You train a `FourierMLP` model on 25 procedurally generated 2D datasets. Once trained, a Python HTTP server serves the weights to a browser-based visualiser that re-runs the full forward pass in vanilla JavaScript. You see the decision boundary as a colour-filled canvas, and hovering over any pixel shows the predicted class and confidence in real time.

---

## Project Structure

```
2-D.Pattern Classification Model/
├── Dataset_training.py   # dataset generators, model, training loop, checkpoints
├── server.py             # HTTP inference server (weights + dataset overlay API)
├── app.js                # browser forward pass, rendering, UI logic
├── index.html            # visualiser UI
├── helper_function.py    # accuracy_fn utility
└── checkpoints/          # one .pth file per trained dataset (auto-created)
    ├── TwoSpirals.pth
    ├── Circles.pth
    └── ...               # 25 files total when fully trained
```

---

## Datasets — 25 Patterns in 5 Groups

All datasets are 2D: every point is `(x, y)` with a class label. The five groups are organised by class count.

| Group | Classes | Datasets                                                                                    |
| ----- | ------- | ------------------------------------------------------------------------------------------- |
| 1     | 2       | TwoSpirals, Circles, TwoMoons, XOR, SineBoundary                                            |
| 2     | 6       | ConcentricRings_6, AngularWedges_6, GaussianBlobs_6, MultiArmSpiral_6, GridGaussians_6      |
| 3     | 10      | ConcentricRings_10, AngularWedges_10, GaussianBlobs_10, MultiArmSpiral_10, GridGaussians_10 |
| 4     | 14      | ConcentricRings_14, AngularWedges_14, GaussianBlobs_14, MultiArmSpiral_14, GridGaussians_14 |
| 5     | 20      | ConcentricRings_20, AngularWedges_20, GaussianBlobs_20, MultiArmSpiral_20, GridGaussians_20 |

Each multi-class pattern (ConcentricRings, AngularWedges, etc.) is a single parameterised generator called with different `n_classes` values — no code duplication. Datasets are lazily loaded and cached on first access.

---

## The Anti-Memorisation Training Strategy

Every training launch picks a fresh `RUN_SEED = random.randint(0, 99999)` which controls three things simultaneously:

- **Train/test split** — different 80/20 partition of the generated points
- **Generator noise** — different noise realization (different point cloud)
- **Mini-batch shuffle order** — different sequence of gradient updates

After each training block, `fresh_eval` generates a completely new batch with `seed = RUN_SEED + 1000` — an unseen noise realization — and scores the model on it. A model that memorised a fixed seed will show a large gap between `acc` (test split accuracy) and `fresh_eval` (deployment proxy accuracy).

The script is designed to be run 15 times. Early runs may memorise, but by run 10–12 the model has seen enough varied noise realizations that it generalises the geometric structure rather than specific point positions.

---

## Complexity-Aware Hyperparameters

Every dataset is scored on five axes and assigned a complexity bucket (LOW / MEDIUM / HIGH). The bucket drives all hyperparameters automatically — no manual tuning per dataset.

| Axis                     | What it measures                                          |
| ------------------------ | --------------------------------------------------------- |
| `boundary_freq`          | How many oscillations the decision boundary makes         |
| `boundary_sharpness`     | How crisp vs soft the class transitions are               |
| `topological_complexity` | How many disconnected regions per class                   |
| `class_density`          | How many classes compete in a region                      |
| `separability`           | How cleanly separated the classes are (1=trivial, 9=hard) |

LOW datasets get smaller models with stronger regularisation. HIGH datasets get wider networks with more residual blocks and higher weight decay.

---

## Requirements

```
torch
numpy
scikit-learn
```

Install with:

```bash
pip install torch numpy scikit-learn
```

---

## Training

Open a terminal in the project folder and run:

```bash
python Dataset_training.py
```

Run the script 15 times total for full generalisation. Each run trains every dataset for up to 1000 epochs and saves a checkpoint. Datasets that have already hit their `max_epochs` ceiling are processed in ~0 seconds and skipped automatically.

You will see output like:

```
RUN_SEED = 47291  (train/test split and generator noise vary each launch)

── TwoSpirals  [MEDIUM | score=6.3]  2 classes | run 1 iter 1  ep 0→1000 | fresh start
   hparams: hd=256 ms=128 scale=4.82 blk=1 lr=2.0e-03 wd=3.0e-03 ls=0.05
    [0100/1000] loss=0.4821 | test_loss=0.3912 | acc=84.20%
    [0500/1000] loss=0.1834 | test_loss=0.1201 | acc=95.40%
    [1000/1000] loss=0.0743 | test_loss=0.0891 | acc=98.30%
   ✓ 12.4s | acc=98.30% | best=98.30%
   🔍 fresh_eval=95.10%  (unseen noise realisation)
```

A small `fresh_eval` gap (under 3–4%) means the model has genuinely generalised.

---

## Serving the Visualiser

Once at least one dataset is trained:

```bash
python server.py
```

Then open `http://localhost:8000` in a browser.

The startup log shows which checkpoints are loaded:

```
FourierMLP inference server
Checkpoint dir : .../checkpoints
✓ 25/25 pattern(s) ready
   TwoSpirals                 acc=98.3%  fresh=95.1%  ep=3000
   Circles                    acc=100.0% fresh=99.9%  ep=2000
   ...
```

---

## Using the Visualiser

1. Click any pattern button on the left panel
2. The decision boundary renders as a colour-filled canvas — each colour = one class
3. Hover over the canvas to see the predicted class and confidence at that point
4. Toggle **overlay dataset points** to see the training data on top of the boundary
5. Adjust **resolution** (grid density), **point opacity**, and **sample count** with the sliders
6. The model info panel shows `best acc`, `fresh acc`, `complexity bucket`, `epoch`, and architecture details

---

## Checkpoint Format

Each dataset saves as `checkpoints/<DatasetName>.pth` — one file per dataset. This means:

- Files are small (~15 KB–200 KB each depending on model size)
- An architecture change only invalidates the affected dataset's file
- The server reads each file independently at startup

Each `.pth` file contains:

```python
{
    "model_state_dict":     ...,   # nn.Module weights
    "optimizer_state_dict": ...,   # AdamW state for resuming
    "epoch":                int,   # epochs trained so far
    "loss":                 float,
    "best_acc":             float, # best test split accuracy
    "fresh_acc":            float, # most recent deployment proxy score
    "X_mean":               Tensor([2]),
    "X_std":                Tensor([2]),
    "model_config":         { output_dim, hidden_dim, dropout, scale, mapping_size, n_blocks },
    "complexity":           { score, bucket },
}
```

---

## File Descriptions

| File                  | Role                                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `Dataset_training.py` | Dataset generators, `FourierMLP` model, training loop, checkpoint save/load, `main()` orchestrator                             |
| `server.py`           | Pure Python HTTP server — serves `index.html`, `app.js`, and three API endpoints: `/api/patterns`, `/api/weights`, `/api/data` |
| `app.js`              | Full FourierMLP forward pass in vanilla JS, canvas rendering, hover inference, UI controls                                     |
| `index.html`          | Single-page UI — pattern selector grid, canvas, model info panel, sliders                                                      |
| `helper_function.py`  | `accuracy_fn(y_pred, y_true)` used by the eval loop                                                                            |
