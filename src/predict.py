import os
import numpy as np
import torch
import torch.nn as nn
from model import ECG1DCNN  # import your CNN architecture

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
MODEL_PATH = os.path.join(DATA_DIR, "ecg_cnn_model.pth")
TEST_DATA_PATH = os.path.join(DATA_DIR, "ecg_test.npz")

# Label map (adjust if you added more classes)
LABEL_MAP = {
    0: "Normal",
    1: "Atrial Fibrillation",
    2: "PVC (Premature Ventricular Contraction)",
    3: "LBBB (Left Bundle Branch Block)",
    4: "RBBB (Right Bundle Branch Block)"
}

# -----------------------------
# Load trained model
# -----------------------------
print("🧠 Loading trained ECG model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ECG1DCNN(num_classes=len(LABEL_MAP))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Load sample test data
# -----------------------------
if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"Test data not found at {TEST_DATA_PATH}. Run preprocess_data.py first!")

print("📦 Loading test ECG data...")
data = np.load(TEST_DATA_PATH)

# Handle key name differences automatically
if "X_test" in data and "y_test" in data:
    X_test, y_test = data["X_test"], data["y_test"]
elif "X" in data and "y" in data:
    X_test, y_test = data["X"], data["y"]
else:
    raise KeyError(f"Unexpected keys in {TEST_DATA_PATH}: {list(data.keys())}")

# Pick a random sample for demonstration
sample_idx = np.random.randint(0, len(X_test))
sample_signal = X_test[sample_idx]
true_label = y_test[sample_idx]

print(f"\n🔍 Predicting sample #{sample_idx}")
print(f"🩺 True Label: {LABEL_MAP[int(true_label)]}")

# -----------------------------
# Prepare signal for model
# -----------------------------
# shape: [1, 1, signal_length]
ecg_tensor = torch.tensor(sample_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# -----------------------------
# Run model prediction
# -----------------------------
with torch.no_grad():
    output = model(ecg_tensor)
    pred_class = torch.argmax(output, dim=1).item()

pred_label = LABEL_MAP[pred_class]
confidence = torch.softmax(output, dim=1)[0, pred_class].item() * 100

# -----------------------------
# Display result
# -----------------------------
print(f"\n✅ Predicted Class: {pred_label}")
print(f"📊 Confidence: {confidence:.2f}%")
