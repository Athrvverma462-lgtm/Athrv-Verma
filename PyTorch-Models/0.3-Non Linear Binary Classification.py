import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from pathlib import Path
from helper_function import plot_decision_boundary

# ======================
# 1. Generate data
# ======================
SAMPLES = 1000

x, y = make_circles(SAMPLES,
                    noise=0.03,
                    random_state=42)

x = torch.from_numpy(x).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, train_size=0.80)

# ======================
# 2. Define model
# ======================
device = "cpu"

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=2, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, x):
        return self.network(x)

model = CircleModel().to(device)

# ======================
# 3. Loss function and optimizer
# ======================
# BCEWithLogitsLoss has sigmoid built in, so we pass raw logits directly
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# ======================
# 4. Accuracy function
# ======================
def accuracy_fn(y_true, y_pred):
    # count how many predictions match the true labels
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100

# ======================
# 5. Training loop
# ======================
torch.manual_seed(42)
epochs = 5000

x_train, x_test = x_train.to(device), x_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

for epoch in range(epochs):
    model.train()

    # forward pass
    y_logits = model(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # loss and accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # evaluation
    model.eval()
    with torch.inference_mode():
        test_logits = model(x_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_preds)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

# ======================
# 6. Plot decision boundaries
# ======================
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, x_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, x_test, y_test)

plt.tight_layout()
plt.show()

# ======================
# 7. Save model
# ======================
model_path = Path("models")
model_path.mkdir(parents=True, exist_ok=True)

model_save_path = model_path / "circle_model.pth"
torch.save(obj=model.state_dict(), f=model_save_path)
print(f"Model saved to: {model_save_path}")

# ======================
# 8. Load and verify saved model
# ======================
# We saved state_dict() (just weights), not the whole model,
# so we create a fresh instance and load the weights into it.
loaded_model = CircleModel().to(device)
loaded_model.load_state_dict(torch.load(f=model_save_path, weights_only=True))

loaded_model.eval()
with torch.inference_mode():
    loaded_model_preds = torch.round(torch.sigmoid(loaded_model(x_test).squeeze()))

# get original predictions to compare
model.eval()
with torch.inference_mode():
    original_preds = torch.round(torch.sigmoid(model(x_test).squeeze()))

# verify both models produce identical predictions
print("Predictions match:", torch.allclose(original_preds, loaded_model_preds))