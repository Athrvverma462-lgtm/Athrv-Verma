# Hey, I'm Athrv 👋

I'm a self-taught ML engineer building things I actually care about — not just running tutorials, but designing systems end-to-end and using AI as a genuine engineering tool, not a shortcut.

My focus is on what I call **agentic engineering**: understanding _why_ a model works, not just _that_ it works. Every project here is something I built to learn something real.

---

## 🧠 What I Believe

Most people use AI to generate code they don't understand. I'm trying to do the opposite — use AI to go _deeper_ into concepts I'm actively building, so I understand the system well enough to debug, extend, and own it completely.

> _"Vibe coding gets you 80%. Agentic engineering gets you the other 20% — and the understanding to do it again."_

---

## 🚀 Projects

### 📍 2D Pattern Classification Model _(active)_

> Train neural networks on 25 handcrafted 2D datasets and visualise their decision boundaries live in the browser.

A full ML pipeline built from scratch — custom dataset generators, a FourierMLP architecture with Fourier feature mapping and residual transformer blocks, a complexity-aware hyperparameter system, and a browser-based inference visualiser that runs the trained model in JavaScript without any backend at inference time.

**What makes it interesting:**

- 25 datasets organised in 5 complexity groups (2 → 6 → 10 → 14 → 20 classes)
- Every dataset gets its own complexity score across 4 axes: boundary frequency, boundary sharpness, topological complexity, and class density pressure — hyperparameters are derived dynamically from these scores, not hardcoded
- Trained weights are loaded directly from checkpoints — no retraining needed to run the visualiser
- Full batched JS forward pass (FourierMLP in vanilla JS) for fast real-time decision boundary rendering

**Stack:** Python · PyTorch · NumPy · scikit-learn · JavaScript · HTML/CSS

---

### 📈 Trading Bot _(coming soon)_

> Algorithmic trading system — signal generation, backtesting, and live execution.

Planned focus: feature engineering on market data, time-series modelling, and designing a system that knows when _not_ to trade. The hard part isn't the model; it's the decision boundary between noise and signal.

---

### 🍳 Recipe Vision Bot _(coming soon)_

> Upload a photo of ingredients + write any custom constraints → get a structured recipe.

Multimodal pipeline: vision model for ingredient recognition, LLM for recipe generation, constrained by user instructions (dietary restrictions, cuisine style, cooking time). The interesting problem is making the constraints actually stick.

---

## 🛠 Skills

| Area             | Tools                                                                   |
| ---------------- | ----------------------------------------------------------------------- |
| Machine Learning | PyTorch · scikit-learn · NumPy · Fourier Features · MLPs · Transformers |
| Data             | SQL · Pandas · dataset design · train/test pipeline architecture        |
| Frontend         | JavaScript · HTML · CSS · browser-side ML inference                     |
| Engineering      | Python · checkpoint systems · modular pipeline design                   |
| Mindset          | Complexity-aware design · agentic AI usage · debugging over guessing    |

---

## 📂 Repo Structure

Each project lives in its own folder with its own README explaining the design decisions, what I learned, and what I'd do differently.

```
Athrvverma/
├── 2-D.Pattern Classification Model/
│   ├── Dataset_training.py     # 25 dataset generators + complexity scoring
│   ├── server.py               # inference server
│   ├── app.js                  # batched JS forward pass + visualiser
│   ├── index.html              # browser UI
│   └── checkpoints/            # saved model weights
├── trading-bot/                # (coming soon)
└── recipe-vision-bot/          # (coming soon)
```

---

## 📌 Current Focus

- Finishing the 2D classifier: validating complexity-aware hparams across all 25 datasets, improving convergence on HIGH-complexity patterns (TwoSpirals, MultiArmSpiral_20)
- Reading: _Deep Learning_ (Goodfellow et al.) · papers on Fourier feature networks · RL for trading

---

## 🤖 How I Actually Use AI

There's a lot of noise around AI in coding right now. Here's my honest approach:

**1. Exploring new concepts — then going deeper myself**
When I hit something unfamiliar (a paper, an architecture, an unfamiliar loss function), I'll use AI to get a fast first-principles explanation. But that's just the starting point — I then go and read the actual papers, StackOverflow threads, documentation, and other people's implementations to build a real mental model. AI gives me the door; I walk through it myself.

**2. Writing long scripts I already understand**
If I've already designed something in my head — the data flow, the logic, the edge cases — I'll use AI to write the boilerplate fast. I'm not asking it to design the system; I'm asking it to type faster than I can. The understanding comes first, the code second.

**3. Cleaning and refactoring code**
Once something works, I use AI to make it readable — consistent naming, removing redundancy, better structure. This is where it genuinely saves time without any risk, because I can read every line of the output and know immediately if something is wrong.

What I don't do: ask AI to solve problems I haven't thought through, copy output I don't understand, or use it as a substitute for actually learning something. That's vibe coding — and it shows up the moment something breaks.

---

_Built with curiosity. Mistakes included — that's how it works._
