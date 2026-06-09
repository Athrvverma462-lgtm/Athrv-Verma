"""
server.py — FourierMLP inference server
========================================
Serves the visualiser UI and three API endpoints.

Checkpoint (written by Dataset_training.py):
    <this_file's_dir>/checkpoints/<DatasetName>.pth  — one file per dataset

    Each file holds:
        model_state_dict    – nn.Module state
        model_config        – {output_dim, hidden_dim, dropout, scale,
                                mapping_size, n_blocks}
        epoch, loss, best_acc, fresh_acc
        X_mean, X_std       – training-time normalisation tensors (shape [2])
        complexity          – {score, bucket}

Dataset groups (25 total):
    Group 1 —  2 classes: TwoSpirals, Circles, TwoMoons, XOR, SineBoundary
    Group 2 —  6 classes: ConcentricRings_6,  AngularWedges_6,  GaussianBlobs_6,
                           MultiArmSpiral_6,  GridGaussians_6
    Group 3 — 10 classes: ConcentricRings_10, AngularWedges_10, GaussianBlobs_10,
                           MultiArmSpiral_10, GridGaussians_10
    Group 4 — 14 classes: ConcentricRings_14, AngularWedges_14, GaussianBlobs_14,
                           MultiArmSpiral_14, GridGaussians_14
    Group 5 — 20 classes: ConcentricRings_20, AngularWedges_20, GaussianBlobs_20,
                           MultiArmSpiral_20, GridGaussians_20

Architecture (mirrors Dataset_training.py exactly):
    FourierMLP.fourier  → FourierFeatureMapping   (separate, not in .network)
    FourierMLP.network  → nn.Sequential:
        [0]              Linear(mapping_size*2, hidden_dim)
        [1]              GELU
        [2]              Dropout
        [3…3+n_blocks-1] PostLNTransformerBlock (Linear → GELU → Dropout → LayerNorm)
        [3+n_blocks]     Linear(hidden_dim, hidden_dim//2)
        [3+n_blocks+1]   GELU
        [3+n_blocks+2]   Dropout
        [3+n_blocks+3]   Linear(hidden_dim//2, output_dim)

Run:
    python server.py
Open:
    http://localhost:8000
"""

import json
import math
import numpy as np
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import torch
from torch import nn
from sklearn.datasets import make_circles as _sk_circles
from sklearn.datasets import make_moons   as _sk_moons
from sklearn.datasets import make_blobs   as _sk_blobs

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
CKPT_DIR  = BASE_DIR / "checkpoints"
PORT      = 8000

# Dataset names in group order — used by /api/patterns to return a structured list
DATASET_GROUPS = {
    "2 classes":  ["TwoSpirals", "Circles", "TwoMoons", "XOR", "SineBoundary"],
    "6 classes":  ["ConcentricRings_6",  "AngularWedges_6",  "GaussianBlobs_6",
                   "MultiArmSpiral_6",  "GridGaussians_6"],
    "10 classes": ["ConcentricRings_10", "AngularWedges_10", "GaussianBlobs_10",
                   "MultiArmSpiral_10", "GridGaussians_10"],
    "14 classes": ["ConcentricRings_14", "AngularWedges_14", "GaussianBlobs_14",
                   "MultiArmSpiral_14", "GridGaussians_14"],
    "20 classes": ["ConcentricRings_20", "AngularWedges_20", "GaussianBlobs_20",
                   "MultiArmSpiral_20", "GridGaussians_20"],
}
RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# MODEL  (mirrors Dataset_training.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim: int, mapping_size: int, scale: float):
        super().__init__()
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        proj = x @ self.B
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return feat 


class ResidualBlock(nn.Module):
    """Bottleneck residual block: hidden → hidden//2 → hidden. Pre-LN."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        neck = max(dim // 2, 16)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, neck)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up   = nn.Linear(neck, dim)

    def forward(self, x):
        h = self.norm(x)
        h = self.down(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.up(h)
        return x + h


class FourierMLP(nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout, scale,
                 mapping_size=48, n_blocks=1, input_dim=2):
        super().__init__()
        self.fourier = FourierFeatureMapping(input_dim, mapping_size, scale)
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
        return self.head(h)


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT LOADING
# ══════════════════════════════════════════════════════════════════════════════

_registry: dict | None = None


def _get_registry() -> dict:
    # Load all per-dataset .pth files from CKPT_DIR. Cached after first load.
    global _registry
    if _registry is not None:
        return _registry
    if not CKPT_DIR.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {CKPT_DIR}\n"
            "Train first with:  python Dataset_training.py"
        )
    reg = {}
    for pth in sorted(CKPT_DIR.glob("*.pth")):
        name = pth.stem
        try:
            reg[name] = torch.load(pth, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"  ⚠  Could not load {pth.name}: {exc}")
    if not reg:
        raise FileNotFoundError(
            f"No .pth files found in {CKPT_DIR}\n"
            "Train first with:  python Dataset_training.py"
        )
    _registry = reg
    return _registry


def available_patterns() -> list[str]:
    """Return all trained pattern names in group order (group1 first, etc.)."""
    try:
        reg = _get_registry()
    except FileNotFoundError:
        return []
    # Return in group order, skipping any not yet trained
    ordered = []
    for names in DATASET_GROUPS.values():
        for name in names:
            if name in reg:
                ordered.append(name)
    # Append any checkpoint entries not in our group table (shouldn't happen)
    known = {n for names in DATASET_GROUPS.values() for n in names}
    for name in reg:
        if name not in known:
            ordered.append(name)
    return ordered


_payload_cache: dict = {}


def get_weights_payload(name: str) -> dict:
    """Load checkpoint entry for *name* and return a JSON-ready payload."""
    if name in _payload_cache:
        return _payload_cache[name]

    reg = _get_registry()
    if name not in reg:
        raise KeyError(
            f"'{name}' not found in checkpoint. "
            f"Available: {sorted(reg.keys())}"
        )

    entry = reg[name]
    cfg   = entry["model_config"]

    output_dim   = cfg["output_dim"]
    hidden_dim   = cfg["hidden_dim"]
    dropout      = cfg["dropout"]
    scale        = cfg["scale"]
    mapping_size = cfg["mapping_size"]
    n_blocks     = cfg["n_blocks"]

    model = FourierMLP(
        output_dim=output_dim, hidden_dim=hidden_dim, dropout=dropout,
        scale=scale, mapping_size=mapping_size, n_blocks=n_blocks,
    )
    model.load_state_dict(entry["model_state_dict"])
    model.eval()

    def tl(t):
        return t.detach().cpu().float().numpy().tolist()

    # fourier.B shape: (2, mapping_size) — JS transposes to (mapping_size, 2)
    B = model.fourier.B

    # input_layer: [0]=Linear, [1]=GELU, [2]=Dropout
    inp_lin = model.input_layer[0]

    # residual blocks: each has norm, down, up
    res_layers = []
    for blk in model.blocks:
        res_layers.append({
            "type": "residual_block",
            "ln_w": tl(blk.norm.weight),
            "ln_b": tl(blk.norm.bias),
            "W_down": tl(blk.down.weight),
            "b_down": tl(blk.down.bias),
            "W_up":   tl(blk.up.weight),
            "b_up":   tl(blk.up.bias),
        })

    best_acc  = float(entry.get("best_acc",  0.0))
    fresh_acc = float(entry.get("fresh_acc", 0.0))
    cx = entry.get("complexity", {})

    # Layer list consumed by app.js:
    #   [0]         fourier          { B }
    #   [1]         linear           { W, b }   fourier_out → hidden
    #   [2]         gelu
    #   [3…3+n-1]   residual_block   { ln_w, ln_b, W_down, b_down, W_up, b_up }
    #   [3+n]       linear           { W, b }   hidden → output_dim  (head)
    payload = {
        "dataset":           name,
        "num_classes":       output_dim,
        "hidden_dim":        hidden_dim,
        "scale":             scale,
        "n_blocks":          n_blocks,
        "epoch":             int(entry.get("epoch",    0)),
        "loss":              float(entry.get("loss",   0.0)),
        "best_acc":          best_acc,
        "fresh_acc":         fresh_acc,
        "complexity_score":  float(cx.get("score",  0.0)),
        "complexity_bucket": str(cx.get("bucket", "—")),
        "render_extent":     float(entry.get("render_extent", 6.0)),
        "X_mean":            tl(entry["X_mean"]),
        "X_std":             tl(entry["X_std"]),
        "layers": [
            {"type": "fourier", "B": tl(B)},
            {"type": "linear", "W": tl(inp_lin.weight), "b": tl(inp_lin.bias)},
            {"type": "gelu"},
            *res_layers,
            {"type": "linear", "W": tl(model.head.weight), "b": tl(model.head.bias)},
        ],
    }
    _payload_cache[name] = payload
    return payload


# ══════════════════════════════════════════════════════════════════════════════
# DATASET GENERATORS
# Mirror Dataset_training.py generators exactly — same defaults, same seeds.
# Used only for the visual overlay (/api/data); inference uses checkpoint stats.
# ══════════════════════════════════════════════════════════════════════════════

def _counts_per_class(total: int, n_classes: int) -> np.ndarray:
    base   = total // n_classes
    rem    = total % n_classes
    counts = np.full(n_classes, base, dtype=int)
    if rem:
        counts[:rem] += 1
    return counts

def _n_samples_for(n_classes: int, base: int) -> int:
    return max(base, n_classes * 150)


# ── Group 1: 2-class generators ───────────────────────────────────────────────

def _gen_two_spirals(n_samples, noise=0.35, n_turns=1.50, scale=5.00):
    np.random.seed(RANDOM_STATE)
    n     = n_samples // 2
    theta = np.linspace(0, n_turns * 2 * np.pi, n)
    r     = np.linspace(0.5, scale, n)
    X0 = np.column_stack([r * np.cos(theta),         r * np.sin(theta)])
    X1 = np.column_stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])
    X0 += np.random.randn(*X0.shape) * noise
    X1 += np.random.randn(*X1.shape) * noise
    return np.vstack([X0, X1]), np.hstack([np.zeros(n), np.ones(n)]).astype(int)

def _gen_circles(n_samples, noise=0.08, factor=0.50):
    return _sk_circles(n_samples=n_samples, noise=noise,
                       factor=factor, random_state=RANDOM_STATE)

def _gen_two_moons(n_samples, noise=0.15):
    return _sk_moons(n_samples=n_samples, noise=noise, random_state=RANDOM_STATE)

def _gen_xor(n_samples, scale=3.00, noise=0.10):
    np.random.seed(RANDOM_STATE)
    px    = (np.random.rand(n_samples) - 0.5) * scale * 2
    py    = (np.random.rand(n_samples) - 0.5) * scale * 2
    label = ((px > 0) != (py > 0)).astype(int)
    X = np.column_stack([
        px + np.random.randn(n_samples) * noise,
        py + np.random.randn(n_samples) * noise,
    ])
    return X, label

def _gen_sine_boundary(n_samples, scale=5.00, noise=0.15, frequency=1.00, amplitude=1.50):
    np.random.seed(RANDOM_STATE)
    px       = (np.random.rand(n_samples) - 0.5) * scale * 2
    py       = (np.random.rand(n_samples) - 0.5) * scale * 2
    boundary = amplitude * np.sin(frequency * px)
    label    = (py > boundary).astype(int)
    X = np.column_stack([
        px + np.random.randn(n_samples) * noise,
        py + np.random.randn(n_samples) * noise,
    ])
    return X, label


# ── Group 2-5: multi-class generators ─────────────────────────────────────────

def _gen_concentric_rings(n_classes, ring_width=0.80, gap=0.25, noise=0.07):
    np.random.seed(RANDOM_STATE)
    n_samples = _n_samples_for(n_classes, base=2000)
    counts    = _counts_per_class(n_samples, n_classes)
    xs, ys = [], []
    for k, n_k in enumerate(counts):
        r_inner = k * (ring_width + gap)
        r_outer = r_inner + ring_width
        r       = np.random.uniform(r_inner, r_outer, size=n_k)
        theta   = np.random.uniform(0, 2 * np.pi, size=n_k)
        Xk      = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        Xk     += np.random.randn(*Xk.shape) * noise
        xs.append(Xk); ys.append(np.full(n_k, k, dtype=int))
    return np.vstack(xs), np.hstack(ys)

def _gen_angular_wedges(n_classes, r_min=0.30, r_max=5.00, noise=0.08):
    np.random.seed(RANDOM_STATE)
    n_samples = _n_samples_for(n_classes, base=2000)
    theta  = np.random.uniform(0, 2 * np.pi, size=n_samples)
    r      = np.random.uniform(r_min, r_max, size=n_samples)
    labels = (theta / (2 * np.pi) * n_classes).astype(int) % n_classes
    X      = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    X     += np.random.randn(*X.shape) * noise
    return X, labels

def _gen_gaussian_blobs(n_classes, cluster_std=0.55):
    np.random.seed(RANDOM_STATE)
    n_samples = _n_samples_for(n_classes, base=2000)
    X, y = _sk_blobs(n_samples=n_samples, centers=n_classes,
                     cluster_std=cluster_std, random_state=RANDOM_STATE)
    return X, y.astype(int)

def _gen_multiarm_spiral(n_classes, noise=0.06, n_turns=1.50, scale=8.00):
    np.random.seed(RANDOM_STATE)
    n_samples = _n_samples_for(n_classes, base=3000)
    counts    = _counts_per_class(n_samples, n_classes)
    xs, ys = [], []
    for k, n_k in enumerate(counts):
        theta  = np.linspace(0.3, n_turns * 2 * np.pi, n_k)
        r      = np.linspace(0.5, scale, n_k)
        offset = 2 * np.pi * k / n_classes
        Xk     = np.column_stack([r * np.cos(theta + offset),
                                  r * np.sin(theta + offset)])
        Xk    += np.random.randn(*Xk.shape) * noise
        xs.append(Xk); ys.append(np.full(n_k, k, dtype=int))
    return np.vstack(xs), np.hstack(ys)

def _gen_grid_gaussians(n_classes, gap=2.20, spread=0.28):
    np.random.seed(RANDOM_STATE)
    n_samples = _n_samples_for(n_classes, base=2000)
    grid_dim  = int(np.ceil(np.sqrt(n_classes)))
    counts    = _counts_per_class(n_samples, n_classes)
    xs, ys = [], []
    label = 0
    for row in range(grid_dim):
        for col in range(grid_dim):
            if label >= n_classes:
                break
            center = np.array([col * gap, row * gap])
            n_k    = int(counts[label])
            Xk     = center + np.random.randn(n_k, 2) * spread
            xs.append(Xk); ys.append(np.full(n_k, label, dtype=int))
            label += 1
    return np.vstack(xs), np.hstack(ys)


# ── Dispatcher ─────────────────────────────────────────────────────────────────

# Maps dataset name → (generator_fn, n_classes)
_GENERATORS = {
    # 2-class
    "TwoSpirals":          (_gen_two_spirals,       2),
    "Circles":             (_gen_circles,           2),
    "TwoMoons":            (_gen_two_moons,         2),
    "XOR":                 (_gen_xor,               2),
    "SineBoundary":        (_gen_sine_boundary,     2),
    # 6-class
    "ConcentricRings_6":   (_gen_concentric_rings,  6),
    "AngularWedges_6":     (_gen_angular_wedges,    6),
    "GaussianBlobs_6":     (_gen_gaussian_blobs,    6),
    "MultiArmSpiral_6":    (_gen_multiarm_spiral,   6),
    "GridGaussians_6":     (_gen_grid_gaussians,    6),
    # 10-class
    "ConcentricRings_10":  (_gen_concentric_rings, 10),
    "AngularWedges_10":    (_gen_angular_wedges,   10),
    "GaussianBlobs_10":    (_gen_gaussian_blobs,   10),
    "MultiArmSpiral_10":   (_gen_multiarm_spiral,  10),
    "GridGaussians_10":    (_gen_grid_gaussians,   10),
    # 14-class
    "ConcentricRings_14":  (_gen_concentric_rings, 14),
    "AngularWedges_14":    (_gen_angular_wedges,   14),
    "GaussianBlobs_14":    (_gen_gaussian_blobs,   14),
    "MultiArmSpiral_14":   (_gen_multiarm_spiral,  14),
    "GridGaussians_14":    (_gen_grid_gaussians,   14),
    # 20-class
    "ConcentricRings_20":  (_gen_concentric_rings, 20),
    "AngularWedges_20":    (_gen_angular_wedges,   20),
    "GaussianBlobs_20":    (_gen_gaussian_blobs,   20),
    "MultiArmSpiral_20":   (_gen_multiarm_spiral,  20),
    "GridGaussians_20":    (_gen_grid_gaussians,   20),
}


def generate_dataset(name: str) -> tuple:
    """
    Generate overlay points for *name* using the same generator and defaults
    as Dataset_training.py.  Returns (X_list, y_list, mean, std).
    X_mean/X_std here are computed from the generated points and used only
    for the visual coordinate mapping — NOT for model inference (inference
    always uses checkpoint stats).
    """
    if name not in _GENERATORS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(_GENERATORS)}")

    fn, n_classes = _GENERATORS[name]
    if n_classes == 2:
        X, y = fn(n_samples=3000)
    else:
        X, y = fn(n_classes=n_classes)

    X    = np.array(X, dtype=float)
    mean = X.mean(0).tolist()
    std  = np.maximum(X.std(0), 1e-6).tolist()
    return X.tolist(), y.tolist(), mean, std


# ══════════════════════════════════════════════════════════════════════════════
# HTTP HANDLER
# ══════════════════════════════════════════════════════════════════════════════

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"  [{self.address_string()}] {fmt % args}")

    def _send_json(self, code: int, obj: dict):
        body   = json.dumps(obj, separators=(",", ":")).encode()
        origin = self.headers.get("Origin", "*")
        self.send_response(code)
        self.send_header("Content-Type",                 "application/json")
        self.send_header("Content-Length",               str(len(body)))
        self.send_header("Access-Control-Allow-Origin",  origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        MIME = {".html": "text/html", ".js": "application/javascript",
                ".css": "text/css",   ".json": "application/json"}
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",   MIME.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_OPTIONS(self):
        origin = self.headers.get("Origin", "*")
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        # Return patterns in group order with group metadata
        if path == "/api/patterns":
            self._send_json(200, {
                "patterns": available_patterns(),
                "groups":   DATASET_GROUPS,
            })
            return

        static = {
            "/":           BASE_DIR / "index.html",
            "/index.html": BASE_DIR / "index.html",
            "/app.js":     BASE_DIR / "app.js",
        }
        if path in static:
            fp = static[path]
            if fp.exists():
                self._send_file(fp)
            else:
                self._send_json(404, {"error": f"{fp.name} not found"})
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._body()

        # ── /api/weights — return model weights + complexity info ──────────────
        if path == "/api/weights":
            name = body.get("pattern", "").strip()
            if not name:
                self._send_json(400, {"error": "pattern name required"}); return
            try:
                self._send_json(200, get_weights_payload(name))
            except (FileNotFoundError, KeyError) as e:
                self._send_json(404, {"error": str(e)})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        # ── /api/data — return overlay points for the visualiser ───────────────
        if path == "/api/data":
            name = body.get("pattern", "").strip()
            if not name:
                self._send_json(400, {"error": "pattern name required"}); return
            try:
                X, y, mean, std = generate_dataset(name)
                self._send_json(200, {"X": X, "y": y, "X_mean": mean, "X_std": std})
            except KeyError as e:
                self._send_json(404, {"error": str(e)})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        self._send_json(404, {"error": "unknown endpoint"})


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  FourierMLP inference server")
    print(f"  Checkpoint dir : {CKPT_DIR}")

    if not CKPT_DIR.exists():
        print("\n  ⚠  Checkpoint directory not found — train first with:  python Dataset_training.py")
    else:
        try:
            reg = _get_registry()
        except Exception as e:
            print(f"\n  ⚠  Could not load checkpoint: {e}")
            reg = {}

        ckpt_keys   = set(reg.keys())
        known_names = {n for names in DATASET_GROUPS.values() for n in names}
        total       = sum(len(v) for v in DATASET_GROUPS.values())

        # Count only names that exist in checkpoint AND are in our current group table
        matched = [n for names in DATASET_GROUPS.values()
                   for n in names if n in ckpt_keys]
        print(f"  ✓  {len(matched)}/{total} pattern(s) ready")

        for group, names in DATASET_GROUPS.items():
            found = [n for n in names if n in ckpt_keys]
            for n in found:
                e  = reg[n]
                ba = e.get("best_acc",  0.0)
                fa = e.get("fresh_acc", 0.0)
                ep = e.get("epoch",     0)
                print(f"     {n:<26} acc={ba:.1f}%  fresh={fa:.1f}%  ep={ep}")
            missing = [n for n in names if n not in ckpt_keys]
            if missing:
                print(f"     [{group}] not yet trained: {', '.join(missing)}")

        stale = ckpt_keys - known_names
        if stale:
            print(f"\n  ⚠  {len(stale)} unrecognised file(s) in checkpoints dir:")
            for s in sorted(stale):
                print(f"       · {s}.pth")

    print(f"\n  http://localhost:{PORT}\n")
    HTTPServer(("", PORT), Handler).serve_forever()