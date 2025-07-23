import torch
import numpy as np
import os
import torch.nn.functional as F


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_path(args):
    """Generates a model file path based on arguments."""
    suffix = ""
    if args.ycbcr:
        suffix += "_ycbcr"
    if args.perturbation:
        suffix += "_perturbed"

    filename = (
        f"{args.data_prefix[0].split('/')[-1]}_{args.model}_"
        f"lr{args.learning_rate}{suffix}.pt"
    )

    log_dir = "./log/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return os.path.join(log_dir, filename)


def preprocess_batch(batch, args):
    """Applies necessary preprocessing steps to a batch of images."""
    if args.ycbcr:
        # Assuming batch shape is [B, H, W, C]
        y_channel = batch[..., 0]
        return y_channel.unsqueeze(-1)
    if args.single_channel:
        # Assuming batch shape is [B, 4, H, W, C] for packets
        first_band = batch[:, 3, :, :]
        return first_band.unsqueeze(1)
    if args.upscale:
        # This logic was in the original code but never used.
        # Assuming it's for upscaling wavelet packets
        B = batch.shape[0]
        reshaped = batch.permute(0, 1, 4, 2, 3).reshape(-1, 3, 112, 112)
        upscaled = F.interpolate(
            reshaped, size=(224, 224), mode="bilinear", align_corners=False
        )
        return upscaled.view(B, 4, 3, 224, 224).permute(0, 1, 3, 4, 2)
    return batch
