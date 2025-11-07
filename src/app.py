import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy.signal import resample
from model import ECG1DCNN  # your trained model class

# --- Load model ---
MODEL_PATH = "../processed/ecg_cnn_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ECG1DCNN(num_classes=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- Class names ---
CLASSES = {
    0: "Normal Beat",
    1: "Atrial Premature Beat",
    2: "Premature Ventricular Contraction",
    3: "Fusion Beat",
    4: "Unclassifiable Beat"
}

# --- Helper: extract signal from ECG image ---
def extract_signal_from_image(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    signal = np.mean(cleaned, axis=0)
    signal = (signal - np.mean(signal)) / np.std(signal)
    signal = resample(signal, 180)
    return signal

# --- Streamlit UI ---
st.title("🩺 ECG Arrhythmia Classifier")
st.write("Upload an ECG image and let AI predict the heart rhythm class.")

uploaded_file = st.file_uploader("Upload ECG Image (.jpg, .png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded ECG Image", use_column_width=True)

    # Process ECG image → 1D signal
    signal = extract_signal_from_image(img)
    ecg_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(ecg_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    st.subheader("🧠 Prediction Result:")
    st.success(f"**{CLASSES[pred_class]}** ({probs[0][pred_class].item() * 100:.2f}% confidence)")

    st.line_chart(signal)
