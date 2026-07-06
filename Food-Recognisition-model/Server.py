"""
server.py — Food101 CNN inference server
==========================================
Serves the upload UI and one prediction endpoint.

Unlike the FourierMLP visualiser (a 2-layer MLP small enough to run its
forward pass in JS), this is a ~4.7M-parameter 4-block CNN with BatchNorm.
Hand-writing conv2d/batchnorm in JS would be slow and error-prone for no
real benefit, so the split here is:

    Python (this file)  — loads the checkpoint, preprocesses the image
                           exactly as training did, runs the real PyTorch
                           forward pass, returns predicted food + confidence.
    JS (app.js)          — upload / drag-drop / paste, preview, and
                           rendering the response. No model math in JS.

Checkpoint (written by 2_1-Food101_cnn_.py):
    <this_file's_dir>/Food101_CNN/food101_cnn.pth       — current checkpoint
    <this_file's_dir>/Food101_CNN/food101_cnn.pth.bak   — previous checkpoint (rollback copy)
    Holds: model_state_dict, optimizer_state_dict, scheduler_state_dict,
    epoch, loss. This server only needs model_state_dict + epoch/loss for
    inference, but it will fall back to the .bak file if the primary
    checkpoint is missing or fails to load — mirroring the training
    script's own load_checkpoint() recovery behaviour.
    (no class list / normalisation stats saved — reconstructed below to
    exactly match the training script.)

Architecture (mirrors 2_1-Food101_cnn_.py exactly):
    HighCapacityFood101CNN — 4 conv blocks (3→64→128, →256→256, →256→256,
    →512) each with BatchNorm+ReLU, MaxPool ÷2 per block, GAP, then a
    512→256→101 classifier head.

Preprocessing (must match training's *test* transform, MINUS the random
TrivialAugmentWide call — inference has to be deterministic):
    Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet mean/std)

Run:
    python server.py
Open:
    http://localhost:8000
"""

import base64
import io
import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import torch
from torch import nn
from PIL import Image
from torchvision import transforms

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
CKPT_PATH   = BASE_DIR / "Food101_CNN" / "food101_cnn.pth"
PORT        = 8000

# Sorted alphabetically — this is exactly the order torchvision.datasets.Food101
# assigns as class indices (`self.classes = sorted(metadata.keys())`), so it
# must match the label the model was trained against index-for-index.
CLASS_NAMES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros",
    "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee",
    "croque_madame", "cup_cakes", "deviled_eggs", "donuts", "dumplings",
    "edamame", "eggs_benedict", "escargots", "falafel", "filet_mignon",
    "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup",
    "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt",
    "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese",
    "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings",
    "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak",
    "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu",
    "tuna_tartare", "waffles",
]
assert len(CLASS_NAMES) == 101

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL  (mirrors 2_1-Food101_cnn_.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

class HighCapacityFood101CNN(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.4),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
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
# CHECKPOINT LOADING
# ══════════════════════════════════════════════════════════════════════════════

_model: nn.Module | None = None
_ckpt_meta: dict = {}


def _get_model() -> nn.Module:
    """
    Loads the checkpoint into a fresh model instance (cached in _model after
    the first call).

    Tries CKPT_PATH first. If it's missing or fails to load — e.g. an
    interrupted write on an older training-script version, or a corrupted
    file — falls back to CKPT_BACKUP (the previous good checkpoint written
    by save_checkpoint()'s rotation step). Only if both are unusable does
    this raise, so the server behaves the same way load_checkpoint() does
    in the training script.
    """
    global _model, _ckpt_meta
    if _model is not None:
        return _model

    last_error: Exception | None = None

    for label, path in (("checkpoint", CKPT_PATH)):
        if not path.exists():
            continue
        try:
            model = HighCapacityFood101CNN(input_shape=3, output_shape=101).to(DEVICE)
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            _ckpt_meta = {
                "epoch": ckpt.get("epoch", 0),
                "loss": float(ckpt.get("loss", 0.0)),
                "total_params": sum(p.numel() for p in model.parameters()),
                "source": path.name,
            }
            _model = model
            if label == "backup checkpoint":
                print(f"  ⚠  Primary checkpoint unusable — loaded {path.name} instead.")
            return _model
        except (RuntimeError, ValueError, KeyError, EOFError) as e:
            last_error = e
            print(f"  ⚠  {label.capitalize()} at {path} incompatible or corrupted — ({e})")

    if last_error is not None:
        raise RuntimeError(
            f"Both checkpoint and backup were found but failed to load: {last_error}"
        )

    raise FileNotFoundError(
        f"Checkpoint not found: {CKPT_PATH}\n"
        "Train first with:  python 2_1-Food101_cnn_.py"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  (deterministic — matches training's test transform minus
# the random TrivialAugmentWide call, which must never run at inference time)
# ══════════════════════════════════════════════════════════════════════════════

_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _decode_image(data_url: str) -> Image.Image:
    """Accepts a data URL (data:image/<fmt>;base64,....) of any image type
    PIL can read (jpg, png, webp, gif, bmp, etc.) and returns an RGB PIL image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(raw))
    return img.convert("RGB")


@torch.inference_mode()
def predict(data_url: str, top_k: int = 5) -> dict:
    model = _get_model()
    img = _decode_image(data_url)

    x = _preprocess(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_idx = torch.topk(probs, k=min(top_k, len(CLASS_NAMES)))
    top5 = [
        {"label": CLASS_NAMES[i], "prob": round(float(p) * 100, 2)}
        for p, i in zip(top_probs.tolist(), top_idx.tolist())
    ]
    return {
        "prediction": top5[0]["label"],
        "confidence": top5[0]["prob"],
        "top5": top5,
        "epoch": _ckpt_meta.get("epoch", 0),
        "total_params": _ckpt_meta.get("total_params", 0),
    }


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

        if path == "/api/status":
            try:
                _get_model()
                self._send_json(200, {
                    "ready": True,
                    "device": str(DEVICE),
                    "epoch": _ckpt_meta.get("epoch", 0),
                    "total_params": _ckpt_meta.get("total_params", 0),
                    "num_classes": len(CLASS_NAMES),
                    "checkpoint_source": _ckpt_meta.get("source", ""),
                })
            except (FileNotFoundError, RuntimeError) as e:
                self._send_json(200, {"ready": False, "error": str(e)})
            return

        static = {
            "/":           BASE_DIR / "index.html",
            "/index.html": BASE_DIR / "index.html",
            "/app.js":     BASE_DIR / "app.js",
            "/style.css":  BASE_DIR / "style.css",
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

        if path == "/api/predict":
            body = self._body()
            data_url = body.get("image", "")
            if not data_url:
                self._send_json(400, {"error": "image data required"}); return
            try:
                result = predict(data_url)
                self._send_json(200, result)
            except FileNotFoundError as e:
                self._send_json(404, {"error": str(e)})
            except RuntimeError as e:
                self._send_json(503, {"error": str(e)})
            except Exception as e:
                self._send_json(500, {"error": f"inference failed: {e}"})
            return

        self._send_json(404, {"error": "unknown endpoint"})


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  Food101 CNN inference server")
    print(f"  Checkpoint : {CKPT_PATH}")

    if not CKPT_PATH.exists():
        print("\n  ⚠  No checkpoint found — train first with:  python 2_1-Food101_cnn_.py")
    else:
        try:
            _get_model()
            print(f"  ✓  Model loaded from {_ckpt_meta['source']} on {DEVICE} | "
                  f"epoch {_ckpt_meta['epoch']} | "
                  f"{_ckpt_meta['total_params']:,} params | "
                  f"{len(CLASS_NAMES)} classes")
        except Exception as e:
            print(f"\n  ⚠  Could not load checkpoint: {e}")

    print(f"\n  http://localhost:{PORT}\n")
    HTTPServer(("", PORT), Handler).serve_forever()