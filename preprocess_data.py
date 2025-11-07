"""
preprocess_data.py
------------------
Automatically downloads and preprocesses the MIT-BIH Arrhythmia dataset.

Steps:
1. Download dataset (via wfdb)
2. Filter and normalize ECG signals
3. Segment ECG beats around R-peaks
4. Label beats (N, A, V, L, R)
5. Split into train/test sets
6. Save processed data as .npz files

Author: Your Name
License: MIT
"""

import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = "data/mitdb"
OUTPUT_DIR = "../processed"
TARGET_LABELS = {'N': 0, 'A': 1, 'V': 2, 'L': 3, 'R': 4}
SEGMENT_LEN = 360  # 1-second window (since fs = 360Hz)
TEST_SIZE = 0.2
RANDOM_STATE = 42


# -----------------------------
# 1. Bandpass Filter Function
# -----------------------------
def bandpass_filter(sig, lowcut=0.5, highcut=40, fs=360, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, sig)


# -----------------------------
# 2. Load, Filter, and Segment
# -----------------------------
def process_record(record_name):
    """Load and process one ECG record."""
    record_path = os.path.join(DATA_DIR, record_name)
    try:
        record = wfdb.rdrecord(record_name, pn_dir='mitdb')
        annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
    except Exception as e:
        print(f"Error loading {record_name}: {e}")
        return [], []

    signal = record.p_signal[:, 0]  # Lead MLII
    fs = record.fs
    r_peaks = annotation.sample
    symbols = annotation.symbol

    # Apply bandpass filter & normalization
    signal = bandpass_filter(signal, fs=fs)
    signal = (signal - np.mean(signal)) / np.std(signal)

    segments, labels = [], []
    for i, r in enumerate(r_peaks):
        start = r - SEGMENT_LEN // 2
        end = r + SEGMENT_LEN // 2
        if start >= 0 and end <= len(signal):
            label = symbols[i]
            if label in TARGET_LABELS:
                segment = signal[start:end]
                segments.append(segment)
                labels.append(TARGET_LABELS[label])

    return segments, labels


# -----------------------------
# 3. Main Preprocessing Function
# -----------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📥 Downloading MIT-BIH Arrhythmia dataset...")
    wfdb.dl_database('mitdb', dl_dir=DATA_DIR)

    records = [f.replace('.dat', '') for f in os.listdir(DATA_DIR) if f.endswith('.dat')]
    print(f"✅ Found {len(records)} records.")

    all_segments, all_labels = [], []

    for rec in tqdm(records, desc="Processing Records"):
        segs, labs = process_record(rec)
        all_segments.extend(segs)
        all_labels.extend(labs)

    X = np.array(all_segments)
    y = np.array(all_labels)

    print(f"\n✅ Processed {len(X)} beats.")
    print("Shape:", X.shape, "| Labels:", y.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Save as compressed .npz files
    np.savez_compressed(os.path.join(OUTPUT_DIR, "ecg_train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "ecg_test.npz"), X=X_test, y=y_test)

    print(f"\n💾 Saved preprocessed data to '{OUTPUT_DIR}/'")
    print("   - ecg_train.npz")
    print("   - ecg_test.npz")
    print("\n🎉 Preprocessing complete!")


# -----------------------------
# 4. Run Script
# -----------------------------
if __name__ == "__main__":
    main()
print(os.getcwd())
