# 🩺 ECG Arrhythmia Classification using AI & Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%F0%9F%A7%A0-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Deployed%20with-Streamlit-red)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

---

## 🧠 Project Overview
This project leverages **Artificial Intelligence (AI)** and **Machine Learning (ML)** to detect cardiac arrhythmias from **ECG (Electrocardiogram)** signals.  
It uses a **1D Convolutional Neural Network (CNN)** trained on the **MIT-BIH Arrhythmia Database** to classify ECG beats into categories like:

- 🟢 Normal Beat (N)  
- ❤️ Atrial Fibrillation (AF)  
- 💓 Premature Ventricular Contraction (PVC)  
- 🧩 Left Bundle Branch Block (LBBB)  
- ⚡ Right Bundle Branch Block (RBBB)

---

## 🎯 Goal
> Automatically classify ECG signals into various heart rhythm types and assist early detection of cardiac abnormalities.

---

## 📸 Screenshots & Demos

### 🔹 Streamlit Web Interface
> *Users can upload an ECG image and get instant AI-powered predictions.*

![Streamlit UI Screenshot](assets/streamlit_ui_placeholder.png)
*(Replace with your own screenshot after running the app)*

### 🔹 Sample ECG Input
![Sample ECG Signal](assets/sample_ecg_input_placeholder.jpg)

### 🔹 Model Output Example
| Predicted Class | Meaning |
|------------------|----------|
| 1 | Atrial Fibrillation |
| 0 | Normal Beat |

---

## 🧩 Architecture Overview

```mermaid
graph TD;
    A[MIT-BIH ECG Dataset] --> B[Preprocessing & Segmentation]
    B --> C[1D CNN Model (PyTorch)]
    C --> D[Model Training & Evaluation]
    D --> E[Streamlit Web App for Deployment]
