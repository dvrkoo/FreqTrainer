"""Code to create wavelet packet plots."""

from matplotlib.colors import PowerNorm
import argparse

import cv2
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from pywt._doc_utils import _2d_wp_basis_coords


from wavelet_math import (
    compute_packet_rep_2d,
    compute_pytorch_packet_representation_2d_tensor,
)


device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def draw_2d_wp_basis(shape, keys, fmt="k", plot_kwargs=None, ax=None, label_levels=0):
    """Plot a 2D representation of a WaveletPacket2D basis.

    Based on: pywt._doc_utils.draw_2d_wp_basis
    """
    coords, centers = _2d_wp_basis_coords(shape, keys)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    for coord in coords:
        ax.plot(coord[0], coord[1], fmt)
    ax.set_axis_off()
    ax.axis("square")
    if label_levels > 0:
        for key, c in centers.items():
            if len(key) <= label_levels:
                ax.text(
                    c[0],
                    c[1],
                    "".join(key),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=6,
                )
    return fig, ax


def read_pair(path_real, path_fake):
    """Load an image pair into numpy arrays.

    Args:
        path_real (str): A path to a real image.
        path_fake (str): Another path to a generated image.

    Returns:
        tuple: Two numpy arrays for each image.
    """
    face = cv2.cvtColor(cv2.imread(path_real), cv2.COLOR_BGR2RGB) / 255.0
    fake_face = cv2.cvtColor(cv2.imread(path_fake), cv2.COLOR_BGR2RGB) / 255.0
    return face, fake_face


def compute_packet_rep_img(image, wavelet_str, max_lev):
    """Compute the packet representation of an input image.

    Args:
        image (np.array): An image of shape [height, widht, channel]
        wavelet_str (str): A string indicating the desired wavelet according
            to the pywt convention. I.e. 'haar.'
        max_lev (int): The level up to which the packet representation should be
            computed. I.e. 3.

    Raises:
        ValueError: If the image shape does not have 2 or 3 dimensions.

    Returns:
        np.array: A stacked version of the wavelet packet representation.

    # noqa: DAR401
    """
    if len(image.shape) == 3:
        channels_lst = []
        for channel in range(3):
            channels_lst.append(
                compute_packet_rep_2d(image[:, :, channel], wavelet_str, max_lev)
            )
        return np.stack(channels_lst, axis=-1)
    elif len(image.shape) != 2:
        raise ValueError(f"invalid image shape: {image.shape}")
    else:
        return compute_packet_rep_2d(image, wavelet_str, max_lev)


def main():
    """Compute wavelet packets of fake images for visual comparison with RGB."""
    parser = argparse.ArgumentParser(
        description="Plot wavelet decomposition of fake images in RGB."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./visualize/",
        help="Path to the folder containing the data (default: ./visualize/)",
    )
    parser.add_argument(
        "--ycbcr",
        action="store_true",
        help="Use YCbCr color space instead of RGB for wavelet decomposition (default: False)",
    )
    args = parser.parse_args()

    print(args)

    # Only consider fake images (forgeries)
    forgery_labels = [
        "real",
        "FaceShifter",
        "DeepFake",
        "NeuralTextures",
        "Face2Face",
        "FaceSwap",
    ]
    fake_images = [
        args.data_dir + "/original.png",
        args.data_dir + "/faceshifter.png",
        args.data_dir + "/deepfake.png",
        args.data_dir + "/neuraltextures.png",
        args.data_dir + "/face2face.png",
        args.data_dir + "/faceswap.png",
    ]

    # Preprocessing function (reads image, resizes, converts color)
    def preprocess_image(img_path, use_ycbcr=False):
        img = cv2.imread(img_path)
        rgb_img = cv2.resize(img, (224, 224))  # Keep RGB for visualization
        if use_ycbcr:
            # Use only the Y channel
            proc_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        else:
            proc_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        proc_img = cv2.resize(proc_img, (224, 224)).astype(np.float32)
        return rgb_img, proc_img

    # Process each forgery
    fake_data = [preprocess_image(img, args.ycbcr) for img in fake_images]
    fake_rgb, fake_gray = zip(*fake_data)

    # Compute wavelet decomposition for each forgery
    wavelet = "haar"
    max_lev = 1
    fake_decompositions = []
    for img in fake_gray:
        # Add batch & channel dimensions for torch
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        packets = compute_pytorch_packet_representation_2d_tensor(
            img_tensor, wavelet_str=wavelet, max_lev=max_lev
        )
        # Remove extra dims and move to numpy
        decomposition = torch.squeeze(packets).cpu().numpy()
        fake_decompositions.append(decomposition)

    # Define subband titles (first subplot will show the original RGB image)
    subbands = ["RGB", "LL/A", "LH/H", "HL/V", "HH/D"]

    # For each forgery, create a separate plot
    for label, rgb_img, dec in zip(forgery_labels, fake_rgb, fake_decompositions):
        # Create a figure with 5 subplots in one row
        fig, axes = plt.subplots(1, 5, figsize=(20, 4), gridspec_kw={"wspace": 0.1})

        # Set titles for each subplot
        if label == forgery_labels[0]:
            for j, subband in enumerate(subbands):
                axes[j].set_title(subband, fontsize=14)

        # First subplot: original RGB image
        axes[0].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        axes[0].axis("off")

        # Compute absolute values of the wavelet packets
        abs_packets = np.abs(dec)
        # Plot each of the 4 decomposition subbands
        for j in range(4):
            if j == 0:
                # For LL band, apply logarithmic scaling
                ll_band = np.log1p(abs_packets[j])
                axes[j + 1].imshow(
                    ll_band,
                    cmap="gray",
                    norm=colors.Normalize(vmin=ll_band.min(), vmax=ll_band.max()),
                )
            else:
                axes[j + 1].imshow(
                    abs_packets[j],
                    cmap="gray",
                    norm=PowerNorm(gamma=0.1),
                )
            axes[j + 1].axis("off")

        # Add a suptitle for the forgery label
        # fig.suptitle(label, fontsize=16)
        fig.text(0.1, 0.5, label, fontsize=16, rotation=90, va="center")
        plt.tight_layout(pad=0.1)
        plt.savefig(f"wavelet_decomposition_{label}.png", dpi=300, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()
