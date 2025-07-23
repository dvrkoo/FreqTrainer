# Deepfake Detector in Wavelet and Pixel Space

This repository contains a refactored and modular PyTorch implementation for training deepfake detection models. It supports training on both raw pixel data (RGB, YCbCr) and wavelet packet transformations, allowing for the exploration of frequency-based artifacts introduced by generative models.

The codebase is designed to be easily extensible, configurable, and reproducible, leveraging modern best practices for deep learning projects.

## Key Features

- **Modular Design**: The code is split into logical modules for configuration (`config.py`), data handling (`data_loader.py`), training/evaluation logic (`engine.py`), and utilities (`utils.py`).
- **Extensible Model Support**: Easily add new model architectures. Comes with support for:
  - ResNet50 (with variable input channels)
  - A simple CNN baseline
  - Late-Fusion ResNet for dual-stream inputs
  - Cross-Attention models
- **Flexible Data Handling**: Supports multiple data representations (pixels, wavelets) and can combine them using `DoubleDataset` (for dual-stream models) or `CombinedDataset` (for channel-wise concatenation).
- **Experiment Tracking**: Integrated support for both [Comet.ml](https://www.comet.com/) and [TensorBoard](https://www.tensorflow.org/tensorboard) for logging metrics, parameters, and models.
- **Reproducibility**: Easily set a global random seed for reproducible results.
- **Configurable Training**: All key hyperparameters and settings are exposed as command-line arguments.

## Getting Started

Follow these steps to set up your environment, prepare the data, and run the training script.

### 1. Prerequisites

- Python 3.8+
- PyTorch
- NVIDIA GPU with CUDA and cuDNN (highly recommended for performance)

### 2. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/dvrkoo/FreqTrainer.git
    cd repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies from `requirements.txt`:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Comet.ml (Optional but Recommended):**
    This project is integrated with Comet for experiment tracking. If you plan to use it, configure your API key. You can find it in your Comet project settings.
    ```bash
    # You can set this as an environment variable or in a .comet.config file
    export COMET_API_KEY="YourCometAPIKey"
    ```

## How to Run

The `train.py` script is the single entry point for running experiments. Its behavior is controlled entirely by command-line arguments.

### Basic Training Command

This example trains a ResNet model, enabling both `tqdm` progress bars and TensorBoard logging.

```bash
python train.py \
    --data-prefix ./data/224_neuraltextures_wavelets \
    --model resnet \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --pbar \
```
