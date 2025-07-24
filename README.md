# Deepfake Detector in Wavelet and Pixel Space

This repository contains a refactored and modular PyTorch implementation for training deepfake detection models. It supports training on both raw pixel data (RGB, YCbCr) and wavelet packet transformations, allowing for the exploration of frequency-based artifacts introduced by generative models.

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

### Cross-Forgery Generalization Results

The following table presents a critical evaluation of our two main approaches: a model trained directly on image **Pixels** versus a model trained on the **Discrete Wavelet Transform (DWT)** of the images.

The primary goal of this experiment is to measure **generalization**. Specifically, we test how well a model trained to detect _one_ specific type of manipulation (e.g., Deepfake) performs when it is tested against _other, unseen_ manipulation methods (e.g., FaceSwap, NeuralTextures). This is known as a "cross-forgery" evaluation and is a robust test of a model's real-world utility.

#### How to Read the Table

- **Rows (Training Method)**: Each row represents a model trained exclusively on a single forgery technique from the FaceForensics++ (FF++) dataset. For example, the row `**FaceSwap**` shows the results for a model that has only ever seen real images and images manipulated with the FaceSwap method during training.

- **Columns (Testing Method)**: The columns show the performance of that trained model when tested against different forgery datasets.
  - **TNR (True Negative Rate)**: This measures the model's ability to correctly identify **real (pristine)** images. A high TNR means the model has a low false-positive rate.
  - **TPR (True Positive Rate)**: This measures the model's accuracy at correctly detecting the forgery type listed in the column header. This is the detection accuracy.

- **Cell Format (`Pixel / DWT`)**: Each cell contains two values separated by a slash `/`.
  - The first value is the performance of the **Pixel-based** model.
  - The second value is the performance of the **DWT-based** model.
  - The **bold** value indicates which of the two models performed better for that specific test.

#### Key Takeaways

1.  **In-Domain Performance (The Diagonal)**: If you look at the diagonal from top-left to bottom-right (e.g., training on `Deepfake` and testing on `Deepfake TPR`), you can see the model's performance on the _exact task it was trained for_. Both models perform very well here, as expected.

2.  **Generalization to Unseen Fakes (The Off-Diagonal)**: This is the most important part of the evaluation. Notice that for nearly all off-diagonal TPRs, the **DWT model's score is bolded**. This demonstrates a crucial finding: **the wavelet-based (DWT) model generalizes far better to unseen manipulation techniques.** The pixel-based model, while effective on the data it was trained on, is very brittle and often fails completely when shown a new type of forgery.

3.  **Identifying Real Images (TNR)**: The pixel-based model consistently achieves a higher TNR. This suggests it is slightly more robust at identifying pristine images, while the DWT model is more sensitive and might have a slightly higher tendency to flag real images as fake. However, the DWT model's superior generalization in detecting fakes often makes this a worthy trade-off.

---

**Table: Cross-forgery evaluation between Pixel and DWT methods on the c23 compressed FF++ dataset.**
_(For each table entry, the `Pixel / DWT` results are reported, with the bold value highlighting the best-performing one.)_

| Training Method    |      TNR (%)      | Deepfake TPR (%)  | Face2Face TPR (%) | FaceSwap TPR (%)  | Neural Textures TPR (%) | FaceShifter TPR (%) |
| :----------------- | :---------------: | :---------------: | :---------------: | :---------------: | :---------------------: | :-----------------: |
| **Deepfake**       | **98.64** / 94.93 | **97.57** / 95.79 | 7.79 / **11.79**  |  0.00 / **1.93**  |    14.07 / **19.21**    |  7.43 / **15.64**   |
| **Face2Face**      | **98.36** / 94.50 | 31.14 / **33.00** | **97.07** / 96.14 |  3.07 / **7.79**  |     6.29 / **8.36**     |   1.07 / **4.07**   |
| **FaceSwap**       | **98.21** / 92.00 |  1.14 / **2.00**  | 4.57 / **15.50**  | **96.79** / 93.29 |     0.36 / **2.43**     |   0.57 / **4.50**   |
| **NeuralTextures** | **93.07** / 83.00 | 45.50 / **65.29** | 23.00 / **27.14** | 4.50 / **10.00**  |    **89.14** / 79.14    |  22.43 / **35.00**  |
| **FaceShifter**    | **99.57** / 92.79 | 1.86 / **29.07**  |  0.29 / **6.71**  |  0.00 / **3.29**  |    1.57 / **13.86**     |  97.71 / **98.29**  |
