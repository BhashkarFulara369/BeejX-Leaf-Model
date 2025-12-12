# Model Architecture & Configuration Explained

This document explains **WHY** we chose specific settings in `config.yaml`. Use this information to answer judges' technical questions.

## 1. Why MobileNetV2?
We chose **MobileNetV2** for its efficiency on mobile devices compared to larger models.

| Model | Size (MB) | Speed (FPS) | Accuracy | Best Use |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | **14 MB** | **Real-time** | **High** | **Mobile Offline** |
| ResNet50 | 98 MB | Slow | Very High | Cloud |
| VGG16 | 528 MB | Very Slow | High | Legacy |

**Selection Criteria**: MobileNetV2 was selected for its balance of lightweight architecture (3.5MB quantised) and high accuracy for offline edge deployment.

## 2. Configuration Parameters Explained

### `model` Section
-   **`input_shape: [224, 224, 3]`**
    -   **Why 224?**: This is the global standard resolution for ImageNet models. Most pre-trained "brains" expect exactly this size.
    -   **Are my images 224x224?**: No, your images are likely larger (e.g., 1920x1080).
    -   **The Magic**: Our `loader.py` automatically **resizes** every image to 224x224 before feeding it to the model. You don't need to resize them manually.
    -   **`3`**: Represents the RGB color channels (Red, Green, Blue).

-   **`alpha: 1.0`**
    -   **Width Multiplier**.
    -   `1.0`: Standard size.
    -   `0.35`: Extremely tiny model (runs on smartwatches or microcontrollers).
    -   We use `1.0` because modern phones handle it easily, and it's much smarter than `0.35`.

-   **`weights: "imagenet"`**
    -   **Transfer Learning**. instead of training from a blank brain (random noise), we start with a brain that has already seen **14 million images** (cats, dogs, cars, trees).
    -   It already knows "edges", "shapes", and "textures". We just teach it the final layer: "This texture is Late Blight".

-   **`dropout: 0.2`**
    -   **The "Forget" Rate**. During training, we randomly turn off 20% of the neurons.
    -   **Why?**: This forces the model to not rely on just one feature (e.g., "yellow pixel in top left"). It forces it to learn robust patterns, preventing **Overfitting**.

### `training` Section
-   **`batch_size: 32`**
    -   We feed 32 images at a time to the GPU/CPU.
    -   Higher (64, 128) is faster but uses more RAM.
    -   Lower (8, 16) is slower but more stable. `32` is the Goldilocks number for 16GB RAM.

-   **`learning_rate: 0.0001`**
    -   **How fast we learn**.
    -   Since we use Pre-trained weights (`imagenet`), we want to learn **slowly** (`0.0001`) to rigorously fine-tune the model, rather than running fast (`0.01`) and destroying the pre-learned knowledge.

-   **`validation_split: 0.2`**
    -   **80/20 Rule**. We train on 80% of data and test on 20%.
    -   Crucial for proving the model works on "Unseen Data".

### `augmentation` Section
-   **`rotation_range: 0.2`** & **`zoom_range: 0.2`**
    -   Since we have small datasets (e.g., 68 Nematode images), we fake more data.
    -   The loader randomly rotates (20%) and zooms (20%) images every epoch.
    -   This means the model never sees the exact same image twice!
