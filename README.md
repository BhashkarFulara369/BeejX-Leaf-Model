# BeejX LeafModel (MobileNetV2)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Edge-AI Solution for Crop Disease Recognition (Offline & Mobile-First).**

## Quick Start


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

## Architecture
We use **MobileNetV2** (Quantized) for < 2.5MB model size, enabling real-time inference on low-end Android devices in rural areas.
*   **[Read Full Architecture Docs](docs/ARCHITECTURE.md)** regarding Input Size, Alpha, and Hyperparameters.


## Tech Stack
*   **Core**: TensorFlow 2.19.0, Keras
*   **Pipeline**: Custom `BeejXDataLoader` with Real-time Augmentation (Rotation/Zoom).
*   **Optimization**: Post-training Quantization (Float32 -> Int8).
*   **Handling Imbalance**: Algorithmic Class Weighting (sklearn).


## Data Version Control (DVC)

Used data version control ---
`data_processed` folder:
```bash
dvc init
dvc add data_processed
```
