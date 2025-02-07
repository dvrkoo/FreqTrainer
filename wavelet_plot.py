"""Code to create wavelet packet plots."""

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
    """Compute wavelet packets of real and generated images for visual comparison with RGB."""
    parser = argparse.ArgumentParser(
        description="Plot wavelet decomposition of real and fake images in RGB."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./visualize/",
        help="Path to the folder containing the data (default: ./data/)",
    )
    parser.add_argument(
        "--real-data",
        type=str,
        default="A_ffhq",
        help="Folder name for real data (default: A_ffhq)",
    )
    parser.add_argument(
        "--fake-data",
        type=str,
        default="B_stylegan",
        help="Folder name for fake data (default: B_stylegan)",
    )
    parser.add_argument(
        "--ycbcr",
        action="store_true",
        help="Use YCbCr color space instead of RGB for wavelet decomposition (default: False)",
    )
    args = parser.parse_args()

    print(args)

    # Image file paths
    real_images = [args.data_dir + "/original.png"]
    fake_images = [
        args.data_dir + "/faceshifter.png",
        args.data_dir + "/deepfake.png",
        args.data_dir + "/neuraltextures.png",
        args.data_dir + "/face2face.png",
        args.data_dir + "/faceswap.png",
    ]

    # Load and preprocess images
    def preprocess_image(img_path, use_ycbcr=False):
        img = cv2.imread(img_path)
        rgb_img = cv2.resize(img, (224, 224))  # Keep RGB for visualization
        if use_ycbcr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]  # Y channel only
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale conversion
        img = cv2.resize(img, (224, 224)).astype(np.float32)
        return rgb_img, img

    all_images = [
        preprocess_image(img, args.ycbcr) for img in real_images + fake_images
    ]

    # Separate RGB and grayscale
    rgb_images, gray_images = zip(*all_images)

    # Wavelet decomposition
    wavelet = "haar"
    max_lev = 1
    decompositions = []
    for img in gray_images:
        img_tensor = (
            torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        )  # Add batch & channel dims
        packets = compute_pytorch_packet_representation_2d_tensor(
            img_tensor, wavelet_str=wavelet, max_lev=max_lev
        )
        decompositions.append(torch.squeeze(packets).cpu().numpy())

    # Plot setup
    num_images = len(all_images)
    fig, axes = plt.subplots(
        num_images,
        5,
        figsize=(20, 4 * num_images),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1], "wspace": 0, "hspace": 0.01},
    )

    subbands = ["RGB", "LL/A", "LH/H", "HL/V", "HH/D"]
    scale_min = np.min([np.abs(dec).min() for dec in decompositions]) + 1e-4
    scale_max = np.max([np.abs(dec).max() for dec in decompositions])

    # Add subband titles
    for j, subband in enumerate(subbands):
        axes[0, j].set_title(subband, fontsize=14)

    # Row labels
    row_labels = ["Real"] + [
        "FaceShifter",
        "DeepFake",
        "NeuralTextures",
        "Face2Face",
        "FaceSwap",
    ]

    # Plot images and decompositions
    for i, (rgb_img, dec) in enumerate(zip(rgb_images, decompositions)):
        # Add row label in a dedicated column or as text overlay
        row_label = row_labels[i]
        axes[i, 0].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        axes[i, 0].axis("off")

        # Add the row label as a text overlay (preserves it)
        axes[i, 0].text(
            -0.1,
            0.5,  # Position slightly to the left of the image
            row_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=14,
            transform=axes[i, 0].transAxes,
        )

        abs_packets = np.abs(dec)
        for j in range(4):  # Loop through subbands
            if j == 0:  # LL band
                ll_band = np.log1p(
                    abs_packets[j]
                )  # Log scaling for brightness reduction
                axes[i, j + 1].imshow(
                    ll_band,
                    cmap="gray",
                    norm=colors.Normalize(vmin=ll_band.min(), vmax=ll_band.max()),
                )
            else:  # LH, HL, HH bands
                axes[i, j + 1].imshow(
                    abs_packets[j],
                    norm=colors.LogNorm(vmin=scale_min, vmax=scale_max),
                    cmap="gray",
                )
            axes[i, j + 1].axis("off")

    # Finalize layout
    plt.tight_layout(pad=0.1)
    plt.savefig("wavelet_decomposition_rgb.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
