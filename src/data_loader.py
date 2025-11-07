import wfdb
import numpy as np
import torch
from torch.utils.data import Dataset

class MITBIHDataset(Dataset):
    """
    Custom PyTorch Dataset for MIT-BIH Arrhythmia ECG signals.
    Loads ECG waveforms and corresponding labels.
    """
    def __init__(self, record_ids, data_dir='data/', segment_length=3600):
        self.record_ids = record_ids
        self.data_dir = data_dir
        self.segment_length = segment_length
        self.samples, self.labels = self.load_data()

    def load_data(self):
        all_samples, all_labels = [], []
        for record_id in self.record_ids:
            record = wfdb.rdrecord(f'{record_id}', pn_dir='mitdb')
            annotation = wfdb.rdann(f'{record_id}', 'atr', pn_dir='mitdb')
            signal = record.p_signal[:, 0]  # Use Lead 1 (MLII)

            # Normalize
            signal = (signal - np.mean(signal)) / np.std(signal)

            # Segment signals by beats
            r_peaks = annotation.sample
            symbols = annotation.symbol

            for i, r in enumerate(r_peaks):
                start = max(0, r - self.segment_length // 2)
                end = min(len(signal), r + self.segment_length // 2)
                segment = signal[start:end]

                if len(segment) == self.segment_length:
                    label = symbols[i]
                    all_samples.append(segment)
                    all_labels.append(label)
        return np.array(all_samples), np.array(all_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(0)  # (1, L)
        y = self.encode_label(self.labels[idx])
        return x, y

    def encode_label(self, label):
        mapping = {'N': 0, 'A': 1, 'V': 2, 'L': 3, 'R': 4}
        return torch.tensor(mapping.get(label, 0), dtype=torch.long)
