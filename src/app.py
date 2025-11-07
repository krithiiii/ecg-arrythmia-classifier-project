import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import io

# Define the same CNN model architecture (must match training)
'''class ECG1DCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ECG1DCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(64 * 187, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out
'''
class ECG1DCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ECG1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128 * 45, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load trained model (only weights, not training data)
MODEL_PATH = "processed/ecg_cnn_model.pth"
device = torch.device("cpu")

model = ECG1DCNN(num_classes=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
#model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# Mapping from class index → label
CLASS_NAMES = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Premature Ventricular Contraction",
    "Ventricular Flutter",
    "Other"
]

st.title("🫀 ECG Arrhythmia Classifier")
st.write("Upload an ECG image to detect possible arrhythmia types.")

uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded ECG Image", use_column_width=True)

    # Convert to grayscale & simulate 1D signal
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    signal = np.mean(gray, axis=0)  # reduce to 1D
    signal = (signal - np.mean(signal)) / np.std(signal)
    signal = np.expand_dims(signal, axis=(0, 1))  # shape: (1, 1, L)

    # Predict
    with torch.no_grad():
        output = model(torch.tensor(signal, dtype=torch.float32))
        _, predicted = torch.max(output.data, 1)

    st.success(f"✅ Predicted Class: **{CLASS_NAMES[predicted.item()]}**")
