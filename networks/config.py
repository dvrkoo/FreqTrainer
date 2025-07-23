import argparse


def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Train a deepfake detector")

    # --- Data and Model Paths ---
    group = parser.add_argument_group("Data and Model Paths")
    group.add_argument(
        "--data-prefix",
        type=str,
        nargs="+",
        default=["/home/nick/ff_crops/224_neuraltextures_crops_packets_haar_reflect_1"],
        help="Shared prefix of the data paths.",
    )

    # --- Model Architecture ---
    group = parser.add_argument_group("Model Architecture")
    group.add_argument(
        "--model",
        choices=["regression", "cnn", "resnet"],
        default="resnet",
        help="The model type to use.",
    )
    group.add_argument("--nclasses", type=int, default=2, help="Number of classes.")
    group.add_argument(
        "--cross", action="store_true", help="Use Cross-Attention model."
    )
    group.add_argument("--late", action="store_true", help="Use Late-Fusion model.")

    # --- Training Hyperparameters ---
    group = parser.add_argument_group("Training Hyperparameters")
    group.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs."
    )
    group.add_argument(
        "--batch-size", type=int, default=32, help="Input batch size for training."
    )
    group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    group.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for the optimizer.",
    )
    group.add_argument(
        "--class-weights",
        type=float,
        nargs="+",
        default=None,
        help="Class weights for the loss function.",
    )

    # --- Data Preprocessing ---
    group = parser.add_argument_group("Data Preprocessing")
    group.add_argument(
        "--ycbcr", action="store_true", help="Convert images to YCbCr space."
    )
    group.add_argument(
        "--perturbation", action="store_true", help="Use perturbed images."
    )
    group.add_argument(
        "--single-channel",
        action="store_true",
        help="Use a single channel from wavelet packets.",
    )
    group.add_argument(
        "--upscale", action="store_true", help="Upscale wavelet packets."
    )

    # --- Execution and Logging ---
    group = parser.add_argument_group("Execution and Logging")
    group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    group.add_argument(
        "--num-workers", type=int, default=2, help="Number of data loader workers."
    )
    group.add_argument("--pbar", action="store_true", help="Enable progress bars.")
    group.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging."
    )
    group.add_argument(
        "--validation-interval",
        type=int,
        default=1,
        help="Run validation every N epochs.",
    )

    return parser.parse_args()
