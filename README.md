# Image Colorization with LAB Space and CNN

A deep learning project that automatically colorizes grayscale images using a **U-Net style Convolutional Neural Network** trained in the **LAB color space**. The model learns to predict the color channels (`a*` and `b*`) from the lightness channel (`L`) and reconstructs full-color images.

## Overview

The project trains a custom CNN to map grayscale (L channel) inputs to color (ab channels) outputs using the LAB color space — a perceptually uniform space well-suited for colorization tasks. After training, the model predicts colors for unseen grayscale images and visualizes results alongside ground truth comparisons and a regression scatter analysis.

## Project Structure

```
Image-Colorization-main/
└── Image Colorization 2.ipynb    # Main Jupyter Notebook
```

## Dataset

- **Structure expected:**
  - `train_color/` — color training images
  - `train_black/` — grayscale training images
  - `test_color/` — color test images
  - `test_black/` — grayscale test images
- **Format:** `.jpg` or `.png`
- **Image Size:** resized to `256 × 256`
- **Max images loaded:** 1000 (configurable)

## Workflow

### 1. Preprocessing
- Load images and resize to `256 × 256`
- Normalize RGB to `[0, 1]` range
- Convert RGB → LAB color space using OpenCV
- Extract:
  - **L channel** (lightness) → model input, normalized to `[0, 1]`
  - **ab channels** (color) → model target, normalized to `[0, 1]`
- Split into train (90%) and test (10%) sets

### 2. Model Architecture — U-Net Style CNN

```
Input (256×256×1 — L channel)
  → Conv2D(64) + MaxPooling        [Encoder]
  → Conv2D(128) + MaxPooling       [Encoder]
  → Conv2D(256)                    [Bottleneck]
  → UpSampling + Conv2D(128)       [Decoder]
  → UpSampling + Conv2D(64)        [Decoder]
  → Conv2D(2, activation='tanh')   [Output — ab channels]
```

- **Optimizer:** Adam
- **Loss:** Mean Squared Error (MSE)
- **Epochs:** 50
- **Batch Size:** 8

### 3. Training & Analysis
- Train model and track training vs. validation loss
- Plot loss curves over epochs
- Print total trainable parameter count

### 4. Prediction & Visualization
- Predict `ab` channels for test images
- Reconstruct full color images by merging predicted `ab` with input `L`
- Convert LAB → RGB for display
- Show side-by-side comparison: **Grayscale Input | Predicted | Ground Truth**

### 5. Regression Scatter Analysis
- Denormalize predicted and true `ab` values
- Fit a linear regression line on predicted vs. true color values
- Plot scatter with viridis colormap and regression line for model accuracy insight

## Requirements

- Python 3.x
- Jupyter Notebook or JupyterLab
- numpy
- opencv-python
- matplotlib
- tensorflow / keras
- scikit-learn
- tqdm

Install all dependencies with:

```bash
pip install numpy opencv-python matplotlib tensorflow scikit-learn tqdm jupyter
```

## Usage

```bash
git clone https://github.com/your-username/Image-Colorization.git
cd Image-Colorization
jupyter notebook "Image Colorization 2.ipynb"
```

Update `DATASET_PATH` in Cell 2 to point to your dataset directory before running.

## Outputs

- **Training/Validation Loss Plot** — MSE loss curve over 50 epochs
- **Model Summary** — layer-by-layer architecture and parameter count
- **Side-by-side Panels** — grayscale input, predicted color, and ground truth (3 test samples)
- **Regression Scatter Plot** — predicted vs. true ab values with regression line and colormap

## Key Concepts

| Concept | Purpose |
|---|---|
| LAB Color Space | Separates lightness (L) from color (ab), making colorization a well-defined prediction task |
| U-Net Style CNN | Encoder-decoder architecture that preserves spatial structure through upsampling |
| MSE Loss | Minimizes pixel-level color prediction error between predicted and true ab channels |
| LAB → RGB Conversion | Reconstructs viewable color images from model predictions |
| Regression Scatter | Visualizes how closely predicted colors match ground truth values |
