"""
Code to load and compose datasets from pre-processed NumPy files.

This module provides several Dataset classes for PyTorch:
- NumpyDataset: The base class for loading data from a single directory
  containing .npy files, labels.npy, and paths.npy.
- MergedNumpyDataset: A class that loads data from two separate directories,
  merges them, and removes duplicates based on file paths.
- DoubleDataset: A composer class that takes two existing datasets and returns
  items as separate images (e.g., for late-fusion models).
- CombinedDataset: A composer class that takes two existing datasets and
  concatenates their image data into a single tensor.
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["NumpyDataset", "MergedNumpyDataset", "DoubleDataset", "CombinedDataset"]


def _parse_numpy_directory(data_dir: Path) -> Tuple[List[Path], np.ndarray, np.ndarray]:
    """
    Scans a directory for .npy files and separates data, labels, and paths.

    Args:
        data_dir (Path): The directory to scan.

    Returns:
        A tuple containing:
        - List[Path]: A list of paths to the image data files.
        - np.ndarray: An array of labels.
        - np.ndarray: An array of original file paths.

    Raises:
        ValueError: If the directory contains no .npy files or is missing
                    'labels.npy'.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    file_list = sorted(data_dir.glob("*.npy"))
    if not file_list:
        raise ValueError(f"No .npy files found in {data_dir}")

    label_file = data_dir / "labels.npy"
    path_file = data_dir / "paths.npy"

    if not label_file.exists():
        raise ValueError(f"labels.npy not found in {data_dir}")

    # Treat all .npy files that are not labels or paths as image data
    image_files = [f for f in file_list if f not in (label_file, path_file)]

    labels = np.load(label_file)
    # Allow pickle for loading paths, which might be arrays of strings
    paths = (
        np.load(path_file, allow_pickle=True)
        if path_file.exists()
        else np.array([None] * len(labels))
    )

    if len(image_files) != len(labels):
        raise ValueError(
            f"Mismatch between number of images ({len(image_files)}) and "
            f"labels ({len(labels)}) in {data_dir}"
        )

    return image_files, labels, paths


class NumpyDataset(Dataset):
    """
    Creates a PyTorch Dataset to load pre-processed numpy arrays from a directory.
    """

    def __init__(
        self,
        data_dir: str,
        key: str = "image",
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            data_dir (str): Path to the pre-processed folder with numpy files.
            key (str): The dictionary key for the input ('x') component.
            mean (Optional[torch.Tensor]): Pre-computed mean for normalization.
            std (Optional[torch.Tensor]): Pre-computed std for normalization.
        """
        self.data_dir = Path(data_dir)
        print(f"Loading data from {self.data_dir}")

        self.image_files, self.labels, self.paths = _parse_numpy_directory(
            self.data_dir
        )

        self.key = key
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a single data sample from the dataset.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            A dictionary containing the image, label, and file path.
        """
        image_path = self.image_files[idx]
        image = np.load(image_path).astype(np.float32)
        image = torch.from_numpy(image)

        # Normalize the data if mean and std are provided
        if self.mean is not None and self.std is not None:
            image = (image - self.mean) / self.std

        label = torch.tensor(int(self.labels[idx]))
        path = self.paths[idx]

        return {self.key: image, "label": label, "file_path": path}


class MergedNumpyDataset(Dataset):
    """
    A Dataset that merges two data directories, de-duplicating entries.

    This class is designed for a specific use case where two datasets (e.g.,
    one with original images, one with manipulated images) need to be combined
    into a single dataset, ensuring no duplicates based on file paths.
    """

    def __init__(
        self,
        data_dir1: str,
        data_dir2: str,
        key: str = "image",
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            data_dir1 (str): Path to the first data directory.
            data_dir2 (str): Path to the second data directory.
            key (str): The dictionary key for the input ('x') component.
            mean (Optional[torch.Tensor]): Pre-computed mean for normalization.
            std (Optional[torch.Tensor]): Pre-computed std for normalization.
        """
        print(f"Merging data from {data_dir1} and {data_dir2}")
        img1, lbl1, path1 = _parse_numpy_directory(Path(data_dir1))
        img2, lbl2, path2 = _parse_numpy_directory(Path(data_dir2))

        combined_images = img1 + img2
        combined_labels = np.concatenate((lbl1, lbl2))
        combined_paths = np.concatenate((path1, path2))

        # De-duplication logic: use a dictionary to keep only unique entries
        unique_entries = {}
        for img, label, path in zip(combined_images, combined_labels, combined_paths):
            # The key for uniqueness is the file path
            unique_entries[path] = (img, label, path)

        print(f"Total unique files after merging: {len(unique_entries)}")

        # Unpack the unique entries back into lists
        self.image_files = [item[0] for item in unique_entries.values()]
        self.labels = np.array([item[1] for item in unique_entries.values()])
        self.paths = np.array([item[2] for item in unique_entries.values()])

        self.key = key
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieves a single data sample from the merged dataset."""
        image_path = self.image_files[idx]
        image = np.load(image_path).astype(np.float32)
        image = torch.from_numpy(image)

        if self.mean is not None and self.std is not None:
            image = (image - self.mean) / self.std

        label = torch.tensor(int(self.labels[idx]))
        path = self.paths[idx]

        return {self.key: image, "label": label, "file_path": path}


class DoubleDataset(Dataset):
    """
    A composer Dataset that pairs items from two other Datasets.

    This is useful for models that require two separate inputs (e.g., late fusion),
    such as wavelet data and pixel data. It assumes both datasets are aligned
    and have the same length and paths.
    """

    def __init__(self, dataset1: NumpyDataset, dataset2: NumpyDataset):
        """
        Args:
            dataset1 (NumpyDataset): The first dataset.
            dataset2 (NumpyDataset): The second dataset.
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        if len(dataset1) != len(dataset2):
            raise ValueError("Datasets must have the same length.")
        if not np.array_equal(self.dataset1.paths, self.dataset2.paths):
            raise ValueError(
                "Datasets must have the same file paths in the same order."
            )

    def __len__(self) -> int:
        return len(self.dataset1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary with images from both datasets and a single label.
        """
        sample1 = self.dataset1[idx]
        sample2 = self.dataset2[idx]

        return {
            "image1": sample1[self.dataset1.key],
            "image2": sample2[self.dataset2.key],
            "label": sample1["label"],  # Labels are assumed to be identical
        }


class CombinedDataset(Dataset):
    """
    A composer Dataset that concatenates items from two other Datasets.

    This is useful for models that take a single, multi-channel input formed
    by combining data from different sources.
    """

    def __init__(
        self, dataset1: NumpyDataset, dataset2: NumpyDataset, key: str = "image"
    ):
        """
        Args:
            dataset1 (NumpyDataset): The first dataset.
            dataset2 (NumpyDataset): The second dataset.
            key (str): The dictionary key for the final combined image.
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.key = key

        if len(dataset1) != len(dataset2):
            raise ValueError("Datasets must have the same length.")
        if not np.array_equal(self.dataset1.paths, self.dataset2.paths):
            raise ValueError(
                "Datasets must have the same file paths in the same order."
            )

    def __len__(self) -> int:
        return len(self.dataset1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary with a concatenated image and its label.
        """
        sample1 = self.dataset1[idx]
        sample2 = self.dataset2[idx]

        image1 = sample1[self.dataset1.key]
        image2 = sample2[self.dataset2.key]

        # This reshaping logic is specific to the original code's presumed data shapes.
        # It assumes image1 is wavelet data [4, 112, 112, 3] -> [112, 112, 12]
        image1_reshaped = image1.permute(1, 2, 0, 3).reshape(112, 112, -1)

        # Concatenate along the channel dimension
        concatenated_image = torch.cat((image1_reshaped, image2), dim=-1)

        return {
            self.key: concatenated_image,
            "label": sample1["label"],
            "file_path": sample1["file_path"],
        }


def compute_mean_std(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes mean and standard deviation for all images in a dataset.

    Args:
        dataset (Dataset): A PyTorch Dataset where each item is a dictionary
                           containing an "image" key.

    Returns:
        A tuple containing the mean and standard deviation tensors.
    """
    print("Calculating mean and std... This may take a while.")
    # Use a data loader for potentially faster iteration
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, num_workers=4, shuffle=False
    )

    # Use Welford's algorithm for stable online variance calculation
    count = 0
    mean = 0.0
    m2 = 0.0

    for batch in loader:
        images = batch["image"]  # Assuming default key
        batch_size = images.size(0)
        # Reshape to (batch_size, -1) to calculate stats over all pixels
        images = images.view(batch_size, -1)

        delta = images - mean
        mean += torch.sum(delta / (count + images.size(0)), dim=0)
        delta2 = images - mean
        m2 += torch.sum(delta * delta2, dim=0)
        count += batch_size

    if count < 2:
        return mean, torch.zeros_like(mean)

    std = torch.sqrt(m2 / (count - 1))

    return mean, std


# data_utils.py
import os
import torch
from torch.utils.data import DataLoader
from data_loader import (
    DoubleDataset,
    CombinedDataset,
    NumpyDataset,
)  # Assuming these are in data_loader.py


def get_suffix(perturbation, ycbcr):
    """Creates a filename suffix based on data flags."""
    suffix = ""
    if ycbcr:
        suffix += "_ycbcr"
    if perturbation:
        suffix += "_perturbed"
    return suffix


def custom_collate(batch):
    """Custom collate function to handle file paths."""
    key = "packets1"  # Or dynamically find key
    images = torch.stack([item[key] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    file_paths = [item["file_path"] for item in batch]
    return {key: images, "label": labels, "file_path": file_paths}


def create_data_loaders(args):
    """Creates train, validation, and test data loaders."""
    data_prefix = args.data_prefix
    batch_size = args.batch_size
    ycbcr = args.ycbcr
    perturbation = args.perturbation
    num_workers = args.num_workers

    # This logic can be further simplified, but for now, we keep it as is.
    data_set_list = []
    for data_prefix_el in data_prefix:
        key = "image"
        if "raw" in data_prefix_el:
            key = "raw"
        elif "packets" in data_prefix_el:
            key = "packets" + data_prefix_el.split("_")[-1]

        suffix = get_suffix(perturbation, ycbcr)
        train_path = f"{data_prefix_el}_train{suffix}"
        val_path = f"{data_prefix_el}_val{suffix}"
        test_path = f"{data_prefix_el}_test{suffix}"

        train_data_set = (
            NumpyDataset(train_path, key=key) if os.path.exists(train_path) else None
        )
        val_data_set = (
            NumpyDataset(val_path, key=key) if os.path.exists(val_path) else None
        )
        test_data_set = NumpyDataset(test_path, key=key)

        data_set_list.append((train_data_set, val_data_set, test_data_set))

    if len(data_set_list) == 1:
        train_ds, val_ds, test_ds = data_set_list[0]
        train_loader = (
            DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=custom_collate,
            )
            if train_ds
            else None
        )
        val_loader = (
            DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=custom_collate,
            )
            if val_ds
            else None
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=custom_collate,
        )
    elif len(data_set_list) == 2:
        # Handle DoubleDataset logic
        train_loader = DataLoader(
            DoubleDataset(data_set_list[0][0], data_set_list[1][0]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            DoubleDataset(data_set_list[0][1], data_set_list[1][1]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            CombinedDataset(data_set_list[0][2], data_set_list[1][2]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        raise ValueError("Unsupported number of data prefixes.")

    return train_loader, val_loader, test_loader
