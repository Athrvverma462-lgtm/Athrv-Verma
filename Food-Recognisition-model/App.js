/**
 * app.js — Food101 Dish Scanner (frontend)
 *
 * All model math (conv/batchnorm/forward pass) happens in server.py.
 * This file only handles: getting an image from the user (file picker,
 * drag-drop, or clipboard paste), previewing it, sending it to
 * POST /api/predict as a base64 data URL, and rendering the JSON response.
 */

"use strict";

const API = "http://127.0.0.1:8000";

// ── DOM ────────────────────────────────────────────────────────────────────────
const statusEl = document.getElementById("globalStatus");
const dropzone = document.getElementById("dropzone");
const viewfinder = document.getElementById("viewfinder");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const scanBtn = document.getElementById("scanBtn");
const previewImg = document.getElementById("preview");
const placeholder = document.getElementById("placeholder");
const scanLine = document.getElementById("scanLine");

const emptyResult = document.getElementById("emptyResult");
const resultBody = document.getElementById("resultBody");
const predLabel = document.getElementById("predLabel");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceVal = document.getElementById("confidenceVal");
const top5List = document.getElementById("top5List");
const metaEpoch = document.getElementById("metaEpoch");
const metaParams = document.getElementById("metaParams");
const metaClasses = document.getElementById("metaClasses");

let currentDataURL = null;

// ── Boot: check server / model status ───────────────────────────────────────────
async function init() {
  setStatus("busy", "connecting…");
  try {
    const res = await fetch(`${API}/api/status`);
    const data = await res.json();
    if (data.ready) {
      setStatus(
        "ok",
        `model ready · ${data.num_classes} classes · ${data.device}`,
      );
      metaClasses.textContent = data.num_classes;
      metaParams.textContent = formatParams(data.total_params);
      metaEpoch.textContent = data.epoch;
    } else {
      setStatus("err", "no checkpoint found");
      toast(data.error || "Train the model first with 2_1-Food101_cnn_.py");
    }
  } catch {
    setStatus("err", "server unreachable");
    toast("Cannot reach server. Run: python server.py");
  }
}

function formatParams(n) {
  if (!n) return "–";
  return (n / 1e6).toFixed(1) + "M";
}

// ── Image intake ────────────────────────────────────────────────────────────────
viewfinder.addEventListener("click", () => fileInput.click());
browseBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((evt) =>
  viewfinder.addEventListener(evt, (e) => {
    e.preventDefault();
    viewfinder.classList.add("drag-over");
  }),
);
["dragleave", "drop"].forEach((evt) =>
  viewfinder.addEventListener(evt, (e) => {
    e.preventDefault();
    viewfinder.classList.remove("drag-over");
  }),
);
viewfinder.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});

// Paste from clipboard anywhere on the page
window.addEventListener("paste", (e) => {
  const items = e.clipboardData?.items || [];
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) loadFile(file);
      break;
    }
  }
});

function loadFile(file) {
  if (!file.type.startsWith("image/")) {
    toast("That doesn't look like an image file.");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    currentDataURL = reader.result;
    previewImg.src = currentDataURL;
    previewImg.style.display = "block";
    placeholder.style.display = "none";
    scanBtn.disabled = false;
    toast("Photo loaded — click \u201cidentify dish\u201d");
  };
  reader.onerror = () => toast("Could not read that file.");
  reader.readAsDataURL(file);
}

// ── Run inference ────────────────────────────────────────────────────────────────
scanBtn.addEventListener("click", runScan);

async function runScan() {
  if (!currentDataURL) return;
  scanBtn.disabled = true;
  scanLine.classList.add("active");
  setStatus("busy", "identifying…");

  try {
    const res = await fetch(`${API}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: currentDataURL }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

    renderResult(data);
    setStatus("ok", "done");
  } catch (e) {
    setStatus("err", "prediction failed");
    toast("Error: " + e.message);
  } finally {
    scanLine.classList.remove("active");
    scanBtn.disabled = false;
  }
}

function renderResult(data) {
  emptyResult.style.display = "none";
  resultBody.style.display = "block";

  predLabel.textContent = prettify(data.prediction);
  confidenceVal.textContent = data.confidence.toFixed(1) + "%";
  requestAnimationFrame(() => {
    confidenceFill.style.width = data.confidence + "%";
  });

  top5List.innerHTML = "";
  data.top5.forEach((item, i) => {
    const row = document.createElement("div");
    row.className = `top5-row rank-${i}`;
    row.innerHTML = `
      <span class="top5-rank">${i + 1}</span>
      <span class="top5-name">${prettify(item.label)}</span>
      <span class="top5-bar-track"><span class="top5-bar-fill" style="width:${item.prob}%"></span></span>
      <span class="top5-pct">${item.prob.toFixed(1)}%</span>
    `;
    top5List.appendChild(row);
  });

  metaEpoch.textContent = data.epoch;
  metaParams.textContent = formatParams(data.total_params);
}

function prettify(name) {
  return name.replace(/_/g, " ");
}

// ── UI helpers ───────────────────────────────────────────────────────────────────
function setStatus(type, txt) {
  statusEl.className = "status " + type;
  statusEl.textContent = txt;
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
