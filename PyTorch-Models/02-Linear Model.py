import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# 1. Generate synthetic data
# ======================
WEIGHT_TRUE = 0.95
BIAS_TRUE = 0.45

x = torch.arange(start=0.1, end=1.0, step=0.02).unsqueeze(dim=1)  # shape: (45, 1)
y = WEIGHT_TRUE * x + BIAS_TRUE                                     # shape: (45, 1)

# Train/test split (80/20)
split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]

# ======================
# 2. Plotting function
# ======================
def plot_predictions(train_data, train_labels,
                     test_data, test_labels,
                     predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training data')
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing data')
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')
    plt.legend(prop={'size': 14})
    plt.show()

# ======================
# 3. Define linear regression model
# ======================
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear creates learnable weight and bias parameters
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# ======================
# 4. Initialise model, loss, optimizer
# ======================
torch.manual_seed(42)
model = LinearRegressionModel()

loss_fn = nn.L1Loss()                                        # Mean Absolute Error
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("Initial parameters:", model.state_dict())

# ======================
# 5. Training loop
# ======================
epochs = 300
train_losses = []

for epoch in range(epochs):
    model.train()

    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    train_losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f}")

print("\nFinal parameters:", model.state_dict())

# ======================
# 6. Evaluation
# ======================
model.eval()
with torch.inference_mode():
    test_preds = model(x_test)
    test_loss = loss_fn(test_preds, y_test)
    print(f"Test Loss: {test_loss:.4f}")

# ======================
# 7. Plot predictions and loss curve
# ======================
plot_predictions(x_train, y_train, x_test, y_test, predictions=test_preds)

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss (MAE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# ======================
# 8. Save model
# ======================
model_path = Path("models")
model_path.mkdir(parents=True, exist_ok=True)

model_save_path = model_path / "linear_model.pth"
torch.save(obj=model.state_dict(), f=model_save_path)
print(f"Model saved to: {model_save_path}")

# ======================
# 9. Load and verify saved model
# ======================
# We saved state_dict() (just weights), not the whole model,
# so we create a fresh instance and load the weights into it.
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(f=model_save_path, weights_only=True))

loaded_model.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model(x_test)

# Verify both models produce identical predictions
print("Predictions match:", torch.allclose(test_preds, loaded_model_preds))