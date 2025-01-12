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
    """Compute some wavelet packets of real and generated images for visual comparison."""
    parser = argparse.ArgumentParser(
        description="Plot wavelet decomposition of real and fake imgs"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./visualize/",
        help="path of folder containing the data (default: ./data/)",
    )
    parser.add_argument(
        "--real-data",
        type=str,
        default="A_ffhq",
        help="name of folder with real data (default: A_ffhq)",
    )
    parser.add_argument(
        "--fake-data",
        type=str,
        default="B_stylegan",
        help="name of folder with fake data (default: B_stylegan)",
    )
    parser.add_argument(
        "--ycbcr",
        action="store_true",
        help="use YCbCr color space instead of RGB (default: False)",
    )
    args = parser.parse_args()

    print(args)

    pairs = []
    pairs.append(
        read_pair(
            args.data_dir + "/original.png",
            args.data_dir + "/faceshifter.png",
        )
    )
    pairs.append(
        read_pair(
            args.data_dir + "/original.png",
            args.data_dir + "/deepfake.png",
        )
    )
    real_images = [args.data_dir + "/original.png"]
    fake_images = [
        args.data_dir + "/faceshifter.png",
        args.data_dir + "/deepfake.png",
        args.data_dir + "/neuraltextures.png",
        args.data_dir + "/face2face.png",
        args.data_dir + "faceswap.png",
    ]
<<<<<<< Updated upstream

    if args.ycbcr:  # Load and process images
        real_images = [
            cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2YCrCb) for img in real_images
        ]
        fake_images = [
            cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2YCrCb) for img in fake_images
        ]
    else:
        real_images = [
            cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in real_images
        ]
        fake_images = [
            cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in fake_images
        ]

=======
    # real_images = [
    #     cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in real_images
    # ]
    # fake_images = [
    #     cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in fake_images
    # ]
    # let's make real and fake in ycbcr instead
    real_images = [
        cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2YCrCb) for img in real_images
    ]
    fake_images = [
        cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2YCrCb) for img in fake_images
    ]
>>>>>>> Stashed changes
    all_images = real_images + fake_images
    wavelet = "haar"
    max_lev = 1
    decompositions = []
    for img in all_images:
        # img = img[:, :, 0]
        img_resized = cv2.resize(img, (224, 224))
        img_grayscale = (
            torch.from_numpy(np.mean(img_resized, -1).astype(np.float32))
            .unsqueeze(0)
            .to(device)
        )
        packets = compute_pytorch_packet_representation_2d_tensor(
            img_grayscale, wavelet_str=wavelet, max_lev=max_lev
        )
        decompositions.append(torch.squeeze(packets).cpu().numpy())

    # Set up the grid for plotting
    num_images = len(all_images)
    fig, axes = plt.subplots(
        num_images,
        5,
        figsize=(20, 5 * num_images),
        gridspec_kw={"width_ratios": [0.5, 1, 1, 1, 1]},
    )

    subbands = ["LL/A", "LH/H", "HL/V", "HH/D"]
    scale_min = np.min([np.abs(dec).min() for dec in decompositions]) + 2e-4
    scale_max = np.max([np.abs(dec).max() for dec in decompositions])

    # cmap = "cividis"  # Color map for visualization
    # cmap = "viridis"
    # cmap = "inferno"
    cmap = "magma"

    # Add subband labels at the top
    for j in range(4):
        axes[0, j + 1].set_title(subbands[j], fontsize=14)

    row_labels = [
        "Real",
        "FaceShifter",
        "DeepFake",
        "NeuralTextures",
        "Face2Face",
        "FaceSwap",
    ]
    # Plot each image's decomposition
    for i, dec in enumerate(decompositions):
        row_label = row_labels[i]
        axes[i, 0].axis("off")  # Leftmost column for labels
        axes[i, 0].text(
            0.5,
            0.5,
            row_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=14,
            transform=axes[i, 0].transAxes,
        )

        abs_packets = np.abs(dec)
        for j in range(4):  # Loop through subbands
            axes[i, j + 1].imshow(
                abs_packets[j],
                norm=colors.LogNorm(vmin=scale_min, vmax=scale_max),
                cmap=cmap,
            )
            axes[i, j + 1].axis("off")

    # Adjust layout
    plt.tight_layout()
    # plt.show()
    plt.savefig("wavelet_decomposition.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
