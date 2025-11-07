"""
model.py
--------
1D CNN for ECG Arrhythmia Classification using MIT-BIH dataset

Classes:
0 - Normal (N)
1 - Atrial Fibrillation (A)
2 - Premature Ventricular Contraction (V)
3 - Left Bundle Branch Block (L)
4 - Right Bundle Branch Block (R)

Author: Your Name
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# 1. Configuration
# -----------------------------
DATA_DIR = "../processed"
MODEL_PATH = "ecg_cnn.pth"
BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5

# -----------------------------
# 2. Load Dataset
# -----------------------------
print("📦 Loading preprocessed data...")
train_data = np.load(os.path.join(DATA_DIR, "ecg_train.npz"))
test_data = np.load(os.path.join(DATA_DIR, "ecg_test.npz"))

X_train, y_train = train_data["X"], train_data["y"]
X_test, y_test = test_data["X"], test_data["y"]

# Reshape for CNN input: (batch, channels, length)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Data loaded: {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples.")

# -----------------------------
# 3. Define 1D CNN Model
# -----------------------------
class ECG1DCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ECG1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 45, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# -----------------------------
# 4. Train & Evaluate
# -----------------------------
def train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for signals, labels in train_loader:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")


def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            outputs = model(signals)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["N", "A", "V", "L", "R"],
                yticklabels=["N", "A", "V", "L", "R"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# -----------------------------
# 5. Run Training
# -----------------------------
model = ECG1DCNN(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("\n🚀 Training started...")
train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS)

print("\n🔍 Evaluating model...")
evaluate_model(model, test_loader)

# -----------------------------
# 6. Save Model
# -----------------------------
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n💾 Model saved as '{MODEL_PATH}'")
print("🎉 Done! Model ready for inference.")

# Save trained model
MODEL_PATH = os.path.join(DATA_DIR, "ecg_cnn_model.pth")
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved to: {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()  # set model to inference mode


