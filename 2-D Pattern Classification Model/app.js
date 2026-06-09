/**
 * app.js — FourierMLP 2D Pattern Classifier
 *
 * Forward pass runs entirely in JS using batched matrix ops.
 * All grid cells are processed in one shot — no per-pixel function calls.
 *
 * Layer list from server:
 *   [0]          fourier         { B: (2, M) — transposed to (M,2) on load }
 *   [1]          linear          { W, b }   fourier_out → hidden
 *   [2]          gelu
 *   [3…3+n-1]    residual_block  { ln_w, ln_b, W_down, b_down, W_up, b_up }
 *   [3+n]        linear          { W, b }   hidden → output_dim
 */

"use strict";

const API = "http://127.0.0.1:8000";

const CLASS_COLORS = [
  "#7c6af7",
  "#3ecfbf",
  "#f0607a",
  "#f7b731",
  "#45b7d1",
  "#a8e06e",
  "#ff8b64",
  "#a29bfe",
  "#fd79a8",
  "#00cec9",
  "#e17055",
  "#74b9ff",
  "#fdcb6e",
  "#55efc4",
  "#b2bec3",
  "#6c5ce7",
  "#00b894",
  "#e84393",
  "#e2d96e",
  "#81ecec",
];
const COLOR_RGB = CLASS_COLORS.map((h) => [
  parseInt(h.slice(1, 3), 16),
  parseInt(h.slice(3, 5), 16),
  parseInt(h.slice(5, 7), 16),
]);

// ── State ──────────────────────────────────────────────────────────────────────
let model = null;
let compiled = null;
let dataset = null;
let activeBtn = null;
let showPoints = false;
let btnMap = {};

// ── DOM ────────────────────────────────────────────────────────────────────────
const canvas = document.getElementById("plotCanvas");
const ctx = canvas.getContext("2d");
const patternGrid = document.getElementById("patternGrid");
const coordDisplay = document.getElementById("coordDisplay");
const predDisplay = document.getElementById("predDisplay");
const statusEl = document.getElementById("globalStatus");
const modelInfoEl = document.getElementById("modelInfo");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingMsg = document.getElementById("loadingMsg");
const legendEl = document.getElementById("legend");
const emptyMsg = document.getElementById("emptyMsg");
const runBtn = document.getElementById("runBtn");
const dataBtn = document.getElementById("dataBtn");
const resSlider = document.getElementById("resSlider");
const resVal = document.getElementById("resVal");
const alphaSlider = document.getElementById("alphaSlider");
const alphaVal = document.getElementById("alphaVal");
const nSamplesSlider = document.getElementById("nSamplesSlider");
const nSamplesVal = document.getElementById("nSamplesVal");
const noiseSelect = document.getElementById("noiseSelect");

// ── Boot ───────────────────────────────────────────────────────────────────────
async function init() {
  setStatus("busy", "connecting…");
  try {
    const data = await fetchJSON(`${API}/api/patterns`);
    buildPatternGrid(data.patterns);
    setStatus("ok", `${data.patterns.length} patterns found`);
  } catch {
    setStatus("err", "server unreachable");
    patternGrid.innerHTML =
      '<p style="color:var(--danger);font-size:.75rem;font-family:var(--mono);grid-column:1/-1">' +
      "Cannot reach server.<br>Run: python server.py</p>";
  }
}

function buildPatternGrid(patterns) {
  patternGrid.innerHTML = "";
  if (!patterns.length) {
    patternGrid.innerHTML =
      '<p style="color:var(--muted);font-size:.75rem;font-family:var(--mono);grid-column:1/-1">' +
      "No trained checkpoints found.<br>Run Dataset_training.py first.</p>";
    return;
  }
  for (const name of patterns) {
    const btn = document.createElement("button");
    btn.className = "pat-btn";
    btn.textContent = name.replace(/([A-Z])/g, " $1").trim();
    btn.title = name;
    btn.addEventListener("click", () => selectPattern(name));
    patternGrid.appendChild(btn);
    btnMap[name] = btn;
  }
}

// ── Pattern selection ──────────────────────────────────────────────────────────
async function selectPattern(name) {
  if (activeBtn) activeBtn.classList.remove("active");
  activeBtn = btnMap[name];
  if (activeBtn) activeBtn.classList.add("active");

  emptyMsg.style.display = "none";
  showPoints = false;
  dataBtn.textContent = "overlay dataset points";

  setStatus("busy", "loading…");
  showLoading("fetching weights from server…");

  try {
    const [wData, dData] = await Promise.all([
      postJSON(`${API}/api/weights`, { pattern: name }),
      postJSON(`${API}/api/data`, { pattern: name }),
    ]);

    model = processWeights(wData);
    compiled = compileModel(model);
    dataset = dData;

    updateModelInfo();
    runBtn.disabled = false;
    dataBtn.disabled = false;
    setStatus("ok", `${name} ready`);
    toast(`${name} loaded — ${model.num_classes} classes, ep ${model.epoch}`);
    renderBoundary();
  } catch (e) {
    setStatus("err", "load failed");
    toast("Error: " + e.message);
    console.error(e);
  } finally {
    hideLoading();
  }
}

async function refreshData() {
  if (!model) return;
  showLoading("regenerating dataset…");
  try {
    dataset = await postJSON(`${API}/api/data`, { pattern: model.dataset });
    renderBoundary();
  } catch (e) {
    toast("Data refresh failed: " + e.message);
  } finally {
    hideLoading();
  }
}

// ── Fetch helpers ──────────────────────────────────────────────────────────────
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const e = await res.json();
    throw new Error(e.error || `HTTP ${res.status}`);
  }
  return res.json();
}
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const e = await res.json();
    throw new Error(e.error || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Weight processing ──────────────────────────────────────────────────────────
function processWeights(json) {
  // Server sends B as (2, M): [[row0...], [row1...]]
  // We need (M, 2) for row-wise dot product in fourierBatch
  const B_raw = json.layers[0].B;
  const M = B_raw[0].length;
  json.layers[0].B = Array.from({ length: M }, (_, i) => [
    B_raw[0][i],
    B_raw[1][i],
  ]);
  return json;
}

function compileModel(m) {
  // Convert nested JS arrays → flat Float32Arrays once on load
  function flat2d(arr) {
    const rows = arr.length,
      cols = arr[0].length;
    const out = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) out[r * cols + c] = arr[r][c];
    return { data: out, rows, cols };
  }
  function flat1d(arr) {
    return new Float32Array(arr);
  }

  // Flatten B: (M, 2) nested → flat [x0,y0, x1,y1, ...]
  const B_raw = m.layers[0].B;
  const M = B_raw.length;
  const B = new Float32Array(M * 2);
  for (let i = 0; i < M; i++) {
    B[i * 2] = B_raw[i][0];
    B[i * 2 + 1] = B_raw[i][1];
  }

  const steps = [];
  for (let i = 1; i < m.layers.length; i++) {
    const L = m.layers[i];
    if (L.type === "linear") {
      steps.push({ type: "linear", W: flat2d(L.W), b: flat1d(L.b) });
    } else if (L.type === "gelu") {
      steps.push({ type: "gelu" });
    } else if (L.type === "residual_block") {
      steps.push({
        type: "residual",
        lw: flat1d(L.ln_w),
        lb: flat1d(L.ln_b),
        W_down: flat2d(L.W_down),
        b_down: flat1d(L.b_down),
        W_up: flat2d(L.W_up),
        b_up: flat1d(L.b_up),
      });
    }
  }

  return {
    B,
    M,
    steps,
    mean0: m.X_mean[0],
    mean1: m.X_mean[1],
    std0: m.X_std[0],
    std1: m.X_std[1],
    render_extent: m.render_extent || 6.0,
    num_classes: m.num_classes,
  };
}

// ══════════════════════════════════════════════════════════════════════════════
// MATH PRIMITIVES
// All operate on flat Float32Arrays (row-major, N×D layout).
// ══════════════════════════════════════════════════════════════════════════════

/** (N×inDim) @ (outDim×inDim)^T + bias → (N×outDim) */
function matMul(X, N, inDim, W, b, outDim) {
  const out = new Float32Array(N * outDim);
  for (let n = 0; n < N; n++) {
    const xOff = n * inDim;
    const oOff = n * outDim;
    for (let o = 0; o < outDim; o++) {
      const wOff = o * inDim;
      let s = b[o];
      for (let k = 0; k < inDim; k++) s += X[xOff + k] * W.data[wOff + k];
      out[oOff + o] = s;
    }
  }
  return out;
}

/** GELU activation — in-place */
function geluInPlace(X) {
  const c = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < X.length; i++) {
    const v = X[i];
    X[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
  }
}

/** LayerNorm — per row, in-place */
function layerNormBatch(X, N, D, w, b) {
  for (let n = 0; n < N; n++) {
    const off = n * D;
    let mu = 0;
    for (let d = 0; d < D; d++) mu += X[off + d];
    mu /= D;
    let vr = 0;
    for (let d = 0; d < D; d++) {
      const v = X[off + d] - mu;
      vr += v * v;
    }
    const invStd = 1 / Math.sqrt(vr / D + 1e-5);
    for (let d = 0; d < D; d++)
      X[off + d] = (X[off + d] - mu) * invStd * w[d] + b[d];
  }
}

/** Fourier features: (N×2) coords → (N×2M) — NO normalisation, NO 2π factor.
 *  Mirrors Python exactly: proj = x @ B; feat = [sin(proj), cos(proj)]
 */
function fourierBatch(xs, ys, N, B, M) {
  const out = new Float32Array(N * M * 2);
  for (let n = 0; n < N; n++) {
    const x0 = xs[n],
      x1 = ys[n];
    const oOff = n * M * 2;
    for (let m = 0; m < M; m++) {
      const p = B[m * 2] * x0 + B[m * 2 + 1] * x1;
      out[oOff + m] = Math.sin(p);
      out[oOff + m + M] = Math.cos(p);
    }
  }
  return out;
}

/**
 * Residual block (Pre-LN bottleneck):
 *   h = LayerNorm(x) → W_down → GELU → W_up → x + h
 */
function residualBatch(X, N, D, step) {
  const neck = step.W_down.rows;
  const normed = X.slice(); // copy for pre-norm
  layerNormBatch(normed, N, D, step.lw, step.lb);
  const down = matMul(normed, N, D, step.W_down, step.b_down, neck);
  geluInPlace(down);
  const up = matMul(down, N, neck, step.W_up, step.b_up, D);
  const out = new Float32Array(N * D);
  for (let i = 0; i < out.length; i++) out[i] = X[i] + up[i];
  return out;
}

// ── Full forward pass ──────────────────────────────────────────────────────────
function runForward(xs, ys, N) {
  const c = compiled;
  let H = fourierBatch(xs, ys, N, c.B, c.M);
  let dim = c.M * 2;
  for (const step of c.steps) {
    if (step.type === "linear") {
      H = matMul(H, N, dim, step.W, step.b, step.W.rows);
      dim = step.W.rows;
    } else if (step.type === "gelu") {
      geluInPlace(H);
    } else if (step.type === "residual") {
      H = residualBatch(H, N, dim, step);
    }
  }
  return { H, dim };
}

/** Batch argmax — returns Int32Array of predicted class per row */
function inferBatch(xs, ys) {
  const { H, dim } = runForward(xs, ys, xs.length);
  const K = compiled.num_classes;
  const N = xs.length;
  const preds = new Int32Array(N);
  for (let n = 0; n < N; n++) {
    const off = n * dim;
    let best = 0,
      bestVal = H[off];
    for (let k = 1; k < K; k++)
      if (H[off + k] > bestVal) {
        bestVal = H[off + k];
        best = k;
      }
    preds[n] = best;
  }
  return preds;
}

/** Single-point inference for hover — returns { cls, conf } */
function inferOne(wx, wy) {
  if (!compiled) return null;
  const c = compiled;
  const xs = new Float32Array([(wx - c.mean0) / c.std0]);
  const ys = new Float32Array([(wy - c.mean1) / c.std1]);
  const { H } = runForward(xs, ys, 1);
  const K = c.num_classes;
  // Softmax for confidence
  let mx = H[0];
  for (let k = 1; k < K; k++) if (H[k] > mx) mx = H[k];
  let sm = 0;
  const probs = new Float32Array(K);
  for (let k = 0; k < K; k++) {
    probs[k] = Math.exp(H[k] - mx);
    sm += probs[k];
  }
  for (let k = 0; k < K; k++) probs[k] /= sm;
  let cls = 0;
  for (let k = 1; k < K; k++) if (probs[k] > probs[cls]) cls = k;
  return { cls, conf: probs[cls] };
}

// ══════════════════════════════════════════════════════════════════════════════
// RENDERING
// ══════════════════════════════════════════════════════════════════════════════

function dataRange() {
  // Use per-dataset render_extent instead of ±3.5σ heuristic.
  // This keeps the canvas inside the actual training data region so
  // the model is never asked to classify far-OOD points.
  const ext = compiled.render_extent;
  const [m0, m1] = model.X_mean;
  return {
    xMin: m0 - ext,
    xMax: m0 + ext,
    yMin: m1 - ext,
    yMax: m1 + ext,
  };
}

function renderBoundary() {
  if (!model || !compiled) return;
  showLoading("computing decision boundary…");

  requestAnimationFrame(() => {
    const res = parseInt(resSlider.value);
    const CW = canvas.width,
      CH = canvas.height;
    const { xMin, xMax, yMin, yMax } = dataRange();
    const c = compiled;

    // Build normalised grid coords for all res×res cells
    const N = res * res;
    const gxNorm = new Float32Array(N);
    const gyNorm = new Float32Array(N);
    for (let gy = 0; gy < res; gy++) {
      const worldY = yMax - ((gy + 0.5) * (yMax - yMin)) / res;
      const ny = (worldY - c.mean1) / c.std1;
      for (let gx = 0; gx < res; gx++) {
        const worldX = xMin + ((gx + 0.5) * (xMax - xMin)) / res;
        gxNorm[gy * res + gx] = (worldX - c.mean0) / c.std0;
        gyNorm[gy * res + gx] = ny;
      }
    }

    // Single batched forward pass over all grid cells
    const preds = inferBatch(gxNorm, gyNorm);

    // Paint canvas pixels from grid predictions
    const imgData = ctx.createImageData(CW, CH);
    const px = imgData.data;
    for (let row = 0; row < CH; row++) {
      const gy = Math.min(Math.floor((row / CH) * res), res - 1);
      for (let col = 0; col < CW; col++) {
        const gx = Math.min(Math.floor((col / CW) * res), res - 1);
        const cls = preds[gy * res + gx];
        const rgb = COLOR_RGB[cls % COLOR_RGB.length];
        const idx = (row * CW + col) * 4;
        px[idx] = rgb[0];
        px[idx + 1] = rgb[1];
        px[idx + 2] = rgb[2];
        px[idx + 3] = 138;
      }
    }
    ctx.clearRect(0, 0, CW, CH);
    ctx.putImageData(imgData, 0, 0);
    if (showPoints && dataset) overlayPoints();
    buildLegend();
    hideLoading();
  });
}

function overlayPoints() {
  const { xMin, xMax, yMin, yMax } = dataRange();
  const alpha = parseInt(alphaSlider.value) / 100;
  const W = canvas.width,
    H = canvas.height;
  for (let i = 0; i < dataset.X.length; i++) {
    const [wx, wy] = dataset.X[i];
    const cls = dataset.y[i];
    const cx = ((wx - xMin) / (xMax - xMin)) * W;
    const cy = ((yMax - wy) / (yMax - yMin)) * H;
    const rgb = COLOR_RGB[cls % COLOR_RGB.length];
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
    ctx.strokeStyle = "rgba(0,0,0,0.4)";
    ctx.lineWidth = 0.6;
    ctx.fill();
    ctx.stroke();
  }
}

function buildLegend() {
  legendEl.innerHTML = "";
  for (let i = 0; i < model.num_classes; i++) {
    const d = document.createElement("div");
    d.className = "leg-item";
    d.innerHTML = `<div class="leg-dot" style="background:${CLASS_COLORS[i % CLASS_COLORS.length]}"></div>class ${i}`;
    legendEl.appendChild(d);
  }
  legendEl.classList.add("show");
}

// ── Hover ──────────────────────────────────────────────────────────────────────
canvas.addEventListener("mousemove", (e) => {
  if (!model) return;
  const rect = canvas.getBoundingClientRect();
  const { xMin, xMax, yMin, yMax } = dataRange();
  const wx = xMin + ((e.clientX - rect.left) / canvas.width) * (xMax - xMin);
  const wy = yMax - ((e.clientY - rect.top) / canvas.height) * (yMax - yMin);
  coordDisplay.textContent = `x: ${wx.toFixed(3)}   y: ${wy.toFixed(3)}`;
  const r = inferOne(wx, wy);
  if (r) {
    predDisplay.textContent = `→ class ${r.cls}  (${(r.conf * 100).toFixed(1)}%)`;
    predDisplay.style.color = CLASS_COLORS[r.cls % CLASS_COLORS.length];
  }
});
canvas.addEventListener("mouseleave", () => {
  coordDisplay.textContent = "move cursor over canvas";
  predDisplay.textContent = "";
});

// ── Controls ───────────────────────────────────────────────────────────────────
runBtn.addEventListener("click", () => {
  if (model) renderBoundary();
});
dataBtn.addEventListener("click", () => {
  if (!model || !dataset) return;
  showPoints = !showPoints;
  dataBtn.textContent = showPoints
    ? "hide dataset points"
    : "overlay dataset points";
  renderBoundary();
});
resSlider.addEventListener("input", () => {
  resVal.textContent = resSlider.value;
});
alphaSlider.addEventListener("input", () => {
  alphaVal.textContent = alphaSlider.value + "%";
  if (model && showPoints) renderBoundary();
});
nSamplesSlider.addEventListener("input", () => {
  nSamplesVal.textContent = parseInt(nSamplesSlider.value).toLocaleString();
});
document.getElementById("applyParams").addEventListener("click", () => {
  if (model) refreshData();
});

// ── UI helpers ─────────────────────────────────────────────────────────────────
function updateModelInfo() {
  document.getElementById("iDataset").textContent = model.dataset;
  document.getElementById("iClasses").textContent = model.num_classes;
  document.getElementById("iHidden").textContent = model.hidden_dim;
  document.getElementById("iEpoch").textContent = model.epoch;
  document.getElementById("iLoss").textContent = model.loss.toFixed(4);
  document.getElementById("iScale").textContent = model.scale.toFixed(2);
  document.getElementById("iBestAcc").textContent =
    model.best_acc.toFixed(1) + "%";
  document.getElementById("iFreshAcc").textContent =
    model.fresh_acc.toFixed(1) + "%";
  document.getElementById("iComplexity").textContent = model.complexity_bucket;
  modelInfoEl.style.display = "block";
  document.getElementById("applyParams").disabled = false;
}
function setStatus(type, txt) {
  statusEl.className = type;
  statusEl.textContent = txt;
}
function showLoading(msg) {
  loadingMsg.textContent = msg;
  loadingOverlay.classList.add("show");
}
function hideLoading() {
  loadingOverlay.classList.remove("show");
}

let _toastTimer;
function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove("show"), 2800);
}

init();
