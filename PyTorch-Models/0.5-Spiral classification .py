# Multiclass Spiral Classification with checkpoint save/resume
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from helper_function import plot_decision_boundary
from pathlib import Path

# ======================
# 1. Generate spiral data
# ======================
np.random.seed(0)

def create_data(points, classes):
    x = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = list(range(points * class_number, points * (class_number + 1)))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        x[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return x, y

x, y = create_data(points=1000, classes=5)

x = torch.from_numpy(x).type(torch.float32)
y = torch.from_numpy(y).type(torch.LongTensor)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)

device = "cpu"
x_train, x_test = x_train.to(device), x_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# ======================
# 2. Define model
# ======================
class SpiralModel(nn.Module):
    def __init__(self):
        super().__init__()
        # total params = 43909
        self.network = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        return self.network(x)

# ======================
# 3. Loss and accuracy
# ======================
loss_fn = nn.CrossEntropyLoss()

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100

# ======================
# 4. Checkpoint path
# ======================
model_path = Path("models")
model_path.mkdir(parents=True, exist_ok=True)
model_save_path = model_path / "spiral_model_checkpoint.pth"

# ======================
# 5. Save and load functions
# ======================
def save_checkpoint(model, optimizer, epoch, loss):
    torch.save({
        'epoch'               : epoch,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss'                : loss
    }, model_save_path)
    print(f"  >> Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer):
    if not model_save_path.exists():
        print("No checkpoint found — starting from scratch")
        return model, optimizer, 0

    # weights_only=False needed since we saved optimizer state too
    checkpoint = torch.load(model_save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss        = checkpoint['loss']
    print(f"  >> Checkpoint loaded — resuming from epoch {start_epoch}, loss was {loss:.5f}")
    return model, optimizer, start_epoch

# ======================
# 6. Training function
# ======================
def train_model(model, optimizer, start_epoch, total_epochs):
    # reduces lr by half if test loss stops improving for 100 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)
    torch.manual_seed(42)

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        y_logits = model(x_train)
        y_preds  = torch.softmax(y_logits, dim=1).argmax(dim=1)
        loss     = loss_fn(y_logits, y_train)
        acc      = accuracy_fn(y_true=y_train, y_pred=y_preds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logits = model(x_test)
            test_preds  = torch.softmax(test_logits, dim=1).argmax(dim=1)
            test_loss   = loss_fn(test_logits, y_test)
            test_acc    = accuracy_fn(y_true=y_test, y_pred=test_preds)

        scheduler.step(test_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

    return loss

# ======================
# 7. Plotting function
# ======================
def plot_results(model):
    model.eval()
    with torch.inference_mode():
        test_pred = torch.softmax(model(x_test), dim=1).argmax(dim=1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train (True Spirals)")
    plot_decision_boundary(model, x_test, y_test)
    plt.subplot(1, 2, 2)
    plt.title("Test (Predictions)")
    plot_decision_boundary(model, x_test, test_pred)
    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════
#  FIRST RUN — train from scratch and save
# ══════════════════════════════════════════════════════════════════
print("=== RUN 1 (fresh training) ===")
spiral_model = SpiralModel().to(device)
optimizer    = torch.optim.Adam(spiral_model.parameters(), lr=0.001)

final_loss = train_model(
    model        = spiral_model,
    optimizer    = optimizer,
    start_epoch  = 0,
    total_epochs = 1000
)
plot_results(spiral_model)
save_checkpoint(spiral_model, optimizer, epoch=1000, loss=final_loss)

# ══════════════════════════════════════════════════════════════════
#  CONTINUATION — load checkpoint and keep training
# ══════════════════════════════════════════════════════════════════
a = 2
while True:
    print(f"=== RUN {a} (resuming) ===")
    loaded_model     = SpiralModel().to(device)
    loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=0.001)
 
    # load weights + optimizer state from last checkpoint
    loaded_model, loaded_optimizer, start_epoch = load_checkpoint(loaded_model, loaded_optimizer)
    new_target_epoch = start_epoch + 1000
 
    final_loss = train_model(
        model        = loaded_model,
        optimizer    = loaded_optimizer,
        start_epoch  = start_epoch,       # picks up from previous epoch count
        total_epochs = new_target_epoch   # trains 1000 more epochs each run
    )
    plot_results(loaded_model)
    save_checkpoint(loaded_model, loaded_optimizer, epoch=new_target_epoch, loss=final_loss)
 
    if input("Train further? (y/n): ").strip().lower() == "n":
        break
    a += 1