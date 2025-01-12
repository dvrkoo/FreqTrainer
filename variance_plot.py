from data_loader import NumpyDataset
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def custom_collate(batch):
    # Assuming the key for images is 'image', adjust if it's different
    # print(batch[0].keys())
    key = "packets1"  # or whatever key your dataset uses for images
    images = torch.stack([item[key] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return {key: images, "label": labels}


def get_suffix(perturbation, ycbcr):
    suffix = ""
    if ycbcr:
        suffix += "_ycbcr"
    if perturbation:
        suffix += "_perturbed"
    return suffix


bands = ["LL", "LH", "HL", "HH"]


def create_data_loaders(
    data_prefix: str, batch_size: int, ycbcr=False, perturbation=False, test=False
) -> tuple:
    """Create the data loaders needed for training.

    The test set is created outside a loader.

    Args:
        data_prefix (str): Where to look for the data.

    Raises:
        RuntimeError: Raised if the prefix is incorrect.

    Returns:
        tuple: (train_data_loader, val_data_loader, test_data_set)
    """
    data_set_list = []
    print(data_prefix)
    data_prefix = [data_prefix]
    for data_prefix_el in data_prefix:
        print(data_prefix_el)
        # with open(f"{data_prefix_el}_train/mean_std.pkl", "rb") as file:
        # mean, std = pickle.load(file)
        # mean = torch.from_numpy(mean.astype(np.float32))
        # std = torch.from_numpy(std.astype(np.float32))

        # print("mean", mean, "std", std)
        key = "image"
        if "raw" in data_prefix_el.split("_"):
            key = "raw"
        elif "packets" in data_prefix_el.split("_"):
            key = "packets" + data_prefix_el.split("_")[-1]
        elif "fourier" in data_prefix_el.split("_"):
            key = "fourier"
        # check if dir exists
        if os.path.exists(data_prefix_el + "_train" + get_suffix(perturbation, ycbcr)):
            train_data_set = NumpyDataset(
                (data_prefix_el + "_train" + get_suffix(perturbation, ycbcr)),
                # mean=mean,
                # std=std,
                key=key,
            )
        else:
            train_data_set = None
        if os.path.exists(data_prefix_el + "_val" + get_suffix(perturbation, ycbcr)):
            val_data_set = NumpyDataset(
                (data_prefix_el + "_val" + get_suffix(perturbation, ycbcr)),
                # mean=mean,
                # std=std,
                key=key,
            )
        else:
            val_data_set = None
        test_data_set = NumpyDataset(
            (data_prefix_el + "_test" + get_suffix(perturbation, ycbcr)),
            # mean=mean,
            # std=std,
            key=key,
        )
        data_set_list.append((train_data_set, val_data_set, test_data_set))
        print(len(data_set_list))

    if len(data_set_list) == 1:
        print("----------------------")
        if os.path.exists(data_prefix_el + "_train" + get_suffix(perturbation, ycbcr)):
            train_data_loader = DataLoader(
                data_set_list[0][0],
                batch_size=batch_size,
                shuffle=True,
                num_workers=3,
                collate_fn=custom_collate,
            )
        else:
            train_data_loader = None
        if os.path.exists(data_prefix_el + "_val" + get_suffix(perturbation, ycbcr)):
            val_data_loader = DataLoader(
                data_set_list[0][1],
                batch_size=batch_size,
                shuffle=False,
                num_workers=3,
                collate_fn=custom_collate,
            )
        else:
            val_data_loader = None

        test_data_set = DataLoader(
            data_set_list[0][2],
            batch_size=batch_size,
            shuffle=False,
            num_workers=3,
            collate_fn=custom_collate,
        )
        print(len(test_data_set))

    return train_data_loader, val_data_loader, test_data_set


def main():
    data_prefix = [
        "/home/nick/ff_crops/224_deepfake_crops_packets_haar_reflect_1",
        "/home/nick/ff_crops/224_faceshifter_crops_packets_haar_reflect_1",
        "/home/nick/ff_crops/224_neuraltextures_crops_packets_haar_reflect_1",
        "/home/nick/ff_crops/224_faceswap_crops_packets_haar_reflect_1",
        "/home/nick/ff_crops/224_face2face_crops_packets_haar_reflect_1",
    ]

    for folder in data_prefix:
        # Create data loaders
        train_data_loader, _, _ = create_data_loaders(
            folder,
            256,
            False,
            False,
        )

        variances_real = [
            [] for _ in range(4)
        ]  # Variances for real images (one list per wavelet band)
        variances_fake = [
            [] for _ in range(4)
        ]  # Variances for fake images (one list per wavelet band)

        # Iterate over batches
        for batch in tqdm(train_data_loader, desc=f"Processing {folder}", unit="batch"):
            batch_images = batch[train_data_loader.dataset.key].to(
                "cuda", non_blocking=True
            )
            batch_labels = batch["label"].to("cuda", non_blocking=True)

            # Separate images by labels
            real_images = batch_images[batch_labels == 0]
            fake_images = batch_images[batch_labels == 1]

            # Calculate per-band variance for real images
            for img in real_images:
                # Variance per spatial dimension, then average over channels
                band_variances = torch.var(img.float(), dim=(1, 2))  # Shape: [4, 3]
                band_variances = band_variances.mean(dim=1)  # Shape: [4]
                for i, var in enumerate(band_variances):
                    variances_real[i].append(var.item())

            # Calculate per-band variance for fake images
            for img in fake_images:
                band_variances = torch.var(img.float(), dim=(1, 2))  # Shape: [4, 3]
                band_variances = band_variances.mean(dim=1)  # Shape: [4]
                for i, var in enumerate(band_variances):
                    variances_fake[i].append(var.item())
        max_real_freq = 0
        max_fake_freq = 0
        for i in range(4):
            # Compute histograms for real and fake images
            real_hist, _ = np.histogram(variances_real[i], bins=50)
            fake_hist, _ = np.histogram(variances_fake[i], bins=50)
            max_real_freq = max(max_real_freq, real_hist.max())
            max_fake_freq = max(max_fake_freq, fake_hist.max())

        y_max = (
            max(max_real_freq, max_fake_freq) * 1.2
        )  # Add a 20% margin for extra space
        # Plot histogram for real images
        plt.figure(figsize=(16, 8))
        for band_idx in range(4):
            plt.subplot(1, 4, band_idx + 1)
            plt.hist(
                variances_real[band_idx],
                bins=50,
                alpha=0.7,
                color="blue",
                edgecolor="black",
                label=f"Band {band_idx + 1}",
            )
            plt.title(f"Real Images - Band {band_idx + 1}")
            plt.xlabel("Variance")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True)
            plt.ylim(0, y_max)  # Fixed y-axis limit for better comparability
        plt.tight_layout()
        plt.savefig(f"./{folder.split('/')[4].split('_')[1]}_real_variance.png")
        plt.close()

        # Plot histogram for fake images
        plt.figure(figsize=(16, 8))
        for band_idx in range(4):
            plt.subplot(1, 4, band_idx + 1)
            plt.hist(
                variances_fake[band_idx],
                bins=50,
                alpha=0.7,
                color="red",
                edgecolor="black",
                label=f"Band {band_idx + 1}",
            )
            plt.title(f"Fake Images - Band {band_idx + 1}")
            plt.xlabel("Variance")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True)
            plt.ylim(0, y_max)  # Fixed y-axis limit for better comparability
        plt.tight_layout()
        plt.savefig(f"./{folder.split('/')[4].split('_')[1]}_fake_variance.png")
        plt.close()


if __name__ == "__main__":
    main()
