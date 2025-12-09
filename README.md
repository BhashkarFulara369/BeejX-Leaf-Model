# 🌱 BeejX LeafModel (MobileNetV2)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Professional Edge-AI Solution for Crop Disease Recognition (Offline & Mobile-First).**

## 🚀 Quick Start (5 Minutes)

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Put your raw crop images in `data/` and run the organization script:
```bash
python src/scripts/organize.py
```
*Creates a flattened `data_processed/` directory ready for training.*

### 3. Train
```bash
python src/train.py
```
*Outputs `model.tflite` to `exports/` folder.*

---

## 🏗️ Architecture
We use **MobileNetV2** (Quantized) for < 4MB model size, enabling real-time inference on low-end Android devices in rural areas.
*   **[Read Full Architecture Docs](docs/ARCHITECTURE.md)** regarding Input Size, Alpha, and Hyperparameters.
*   **[Read Mock Pitching Suggestion](SUGGESTIONS.md)** for Hackathon Judges.

## 🛠️ Tech Stack
*   **Core**: TensorFlow 2.x, Keras
*   **Pipeline**: Custom `BeejXDataLoader` with Real-time Augmentation (Rotation/Zoom).
*   **Optimization**: Post-training Quantization (Float32 -> Int8).
*   **Handling Imbalance**: Algorithmic Class Weighting (sklearn).

## 📂 Project Structure
```
LeafModel/
├── configs/             # YAML configuration (Hyperparameters)
├── data_processed/      # (Generated) Clean Training Data
├── docs/                # Architecture & Dataset documentation
├── exports/             # Final TFLite models
├── src/
│   ├── models/          # MobileNetV2 definition
│   ├── scripts/         # Data cleaning & loading utilities
│   └── train.py         # Main training loop
└── requirements.txt     # Dependencies
```

## 📊 Data Version Control (DVC)
We recommend initializing DVC to track the `data_processed` folder:
```bash
dvc init
dvc add data_processed
```
