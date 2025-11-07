import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

# === STEP 1: Load ECG image ===
img_path = "../data/ecg_image.jpg"
img = cv2.imread(img_path)

# === STEP 2: Convert to grayscale and invert ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = 255 - gray  # invert: waveform = white, background = black

# === STEP 3: Denoise and enhance contrast ===
blur = cv2.GaussianBlur(gray, (3, 3), 0)
_, binary = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

# === STEP 4: Morphological cleanup (remove grid lines) ===
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

# === STEP 5: Extract signal (mean intensity per column) ===
signal = np.mean(cleaned, axis=0)

# Normalize and resample (to 180 samples for your model)
signal = (signal - np.mean(signal)) / np.std(signal)
signal = resample(signal, 180)

# === STEP 6: Visualize extracted signal ===
plt.figure(figsize=(10, 4))
plt.plot(signal, color='red')
plt.title("Extracted ECG Waveform")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.show()

# === STEP 7: Save as CSV for model prediction ===
np.savetxt("../processed/new_ecg_from_image.csv", signal)
print("✅ Extracted ECG signal saved to '../processed/new_ecg_from_image.csv'")
