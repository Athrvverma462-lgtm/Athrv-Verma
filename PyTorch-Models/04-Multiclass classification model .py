import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy
from pathlib import Path

# ======================
# 1. Generate data
# ======================
NUM_CLASSES = 10
NUM_FEATURES = 5
RANDOM_SEED = 42

# cluster_std adds randomness/spread to each cluster
x_blob, y_blob = make_blobs(n_samples=10000,
                             n_features=NUM_FEATURES,
                             centers=NUM_CLASSES,
                             cluster_std=1.5,
                             random_state=RANDOM_SEED)

x_blob = torch.from_numpy(x_blob).type(torch.float32)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(
    x_blob, y_blob, test_size=0.2, train_size=0.8
)

# ======================
# 2. Visualise raw data
# ======================
plt.figure(figsize=(10, 7))
plt.title("Raw Blob Data")
plt.scatter(x_blob[:, 0], x_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()

# ======================
# 3. Define model
# ======================
device = "cpu"

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """
        input_features  (int): number of input features
        output_features (int): number of output classes
        hidden_units    (int): number of neurons in each hidden layer
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=input_features,  out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.network(x)

model = BlobModel(input_features=NUM_FEATURES,
                  output_features=NUM_CLASSES,
                  hidden_units=64).to(device)

# ======================
# 4. Loss, optimizer, accuracy
# ======================
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

def accuracy_fn(y_true, y_pred):
    # count correct predictions out of total
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100

# ======================
# 5. Training loop
# ======================
epochs = 5000
torch.manual_seed(42)

x_blob_train, y_blob_train = x_blob_train.to(device), y_blob_train.to(device)
x_blob_test, y_blob_test   = x_blob_test.to(device),  y_blob_test.to(device)

for epoch in range(epochs):
    model.train()

    y_logits = model(x_blob_train)
    y_pred   = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train)
    acc  = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # evaluation
    model.eval()
    with torch.inference_mode():
        test_logits = model(x_blob_test)
        test_pred   = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss   = loss_fn(test_logits, y_blob_test)
        test_acc    = accuracy_fn(y_true=y_blob_test, y_pred=test_pred)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

# ======================
# 6. Plot true vs predicted
# ======================
model.eval()
with torch.inference_mode():
    y_pred = torch.softmax(model(x_blob_test), dim=1).argmax(dim=1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("True Labels")
plt.scatter(x_blob_test[:, 0].cpu(),
            x_blob_test[:, 1].cpu(),
            c=y_blob_test.cpu(),
            cmap=plt.cm.RdYlBu)

plt.subplot(1, 2, 2)
plt.title("Predicted Labels")
plt.scatter(x_blob_test[:, 0].cpu(),
            x_blob_test[:, 1].cpu(),
            c=y_pred.cpu(),
            cmap=plt.cm.RdYlBu)

plt.tight_layout()
plt.show()

# torchmetrics accuracy as a final check
accuracy_metric = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
print(f"Torchmetrics Accuracy: {accuracy_metric(y_pred, y_blob_test):.4f}")

# ======================
# 7. Save model
# ======================
model_path = Path("models")
model_path.mkdir(parents=True, exist_ok=True)

model_save_path = model_path / "blob_model.pth"
torch.save(obj=model.state_dict(), f=model_save_path)
print(f"Model saved to: {model_save_path}")

# ======================
# 8. Load and verify saved model
# ======================
# We saved state_dict() (just weights), not the whole model,
# so we create a fresh instance and load the weights into it.
loaded_model = BlobModel(input_features=NUM_FEATURES,
                         output_features=NUM_CLASSES,
                         hidden_units=64).to(device)
loaded_model.load_state_dict(torch.load(f=model_save_path, weights_only=True))

loaded_model.eval()
with torch.inference_mode():
    loaded_model_preds = torch.softmax(loaded_model(x_blob_test), dim=1).argmax(dim=1)

# verify both models produce identical predictions
print("Predictions match:", torch.allclose(y_pred.float(), loaded_model_preds.float()))