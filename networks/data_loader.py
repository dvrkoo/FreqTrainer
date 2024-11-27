"""
Code to load numpy files into memory for further processing with PyTorch.

Written with the numpy based data format
of https://github.com/RUB-SysSec/GANDCTAnalysis in mind.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["NumpyDataset", "CombinedDataset"]


from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Tuple


class DoubleNumpyDataset(Dataset):
    """Create a data loader to load pre-processed numpy arrays from one or two datasets into memory."""

    def __init__(
        self,
        data_dir1: str,
        data_dir2: Optional[str] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        key: Optional[str] = "image",
    ):
        """Create a Numpy-dataset object from one or two datasets.
        Args:
            data_dir1: A path to the first pre-processed folder with numpy files.
            data_dir2: A path to the second pre-processed folder with numpy files. Defaults to None.
            mean: Pre-computed mean to normalize with. Defaults to None.
            std: Pre-computed standard deviation to normalize with. Defaults to None.
            key: The key for the input or 'x' component of the dataset. Defaults to "image".
        Raises:
            ValueError: If an unexpected file name is given or directories are empty.
        """
        self.data_dir1 = Path(data_dir1)
        self.data_dir2 = Path(data_dir2) if data_dir2 else None
        self.mean = mean
        self.std = std
        self.key = key

        if self.data_dir2 is None or str(self.data_dir2) == "":
            # Act as NumpyDataset if data_dir2 is absent or empty
            self._init_single_dataset(self.data_dir1)
        else:
            # Process both datasets
            print(f"Loading data from {self.data_dir1} and {self.data_dir2}")
            self.images, self.labels, self.paths = self._process_datasets(
                self.data_dir1, self.data_dir2
            )

    def _init_single_dataset(self, data_dir: Path):
        """Initialize as a single dataset (NumpyDataset behavior)."""
        print(f"Loading data from {data_dir}")
        self.file_lst = sorted(data_dir.glob("*.npy"))
        if not self.file_lst:
            raise ValueError(f"No .npy files found in {data_dir}")

        self.label_file = next(
            (f for f in self.file_lst if f.name == "labels.npy"), None
        )
        self.path_file = next((f for f in self.file_lst if f.name == "paths.npy"), None)
        if self.label_file is None:
            raise ValueError(f"labels.npy not found in {data_dir}")

        self.images = [
            f for f in self.file_lst if f not in (self.label_file, self.path_file)
        ]
        self.labels = np.load(self.label_file)
        self.paths = np.load(self.path_file, allow_pickle=True)

        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Mismatch between number of images ({len(self.images)}) and labels ({len(self.labels)})"
            )

    def _process_datasets(
        self, dir1: Path, dir2: Path
    ) -> Tuple[List[Path], np.ndarray, np.ndarray]:
        """Process two datasets and remove duplicates."""
        file_lst1 = sorted(dir1.glob("*.npy"))
        file_lst2 = sorted(dir2.glob("*.npy"))

        if not file_lst1 or not file_lst2:
            raise ValueError(f"No .npy files found in {dir1} or {dir2}")

        # Process each directory
        images1, labels1, paths1 = self._process_single_dataset(dir1, file_lst1)
        images2, labels2, paths2 = self._process_single_dataset(dir2, file_lst2)

        # Combine and remove duplicates
        combined_images = images1 + images2
        combined_labels = np.concatenate((labels1, labels2))
        combined_paths = np.concatenate((paths1, paths2))

        # Remove duplicates based on file names
        unique_files = {}
        original_files = {}

        for img, label, path in zip(combined_images, combined_labels, combined_paths):
            path_prefix = str(path).split("/")[-2]
            if path_prefix == "original":
                file_name = img.name
                if file_name not in original_files:
                    original_files[file_name] = (img, label, path)
            else:
                unique_files[path] = (img, label, path)
        for file_name, (img, label, path) in original_files.items():
            unique_files[path] = (img, label, path)

        # Add all unique original files to the unique_files dict

        print(f"Total unique files: {len(unique_files)}")  # Unpack the unique files
        images = [item[0] for item in unique_files.values()]
        labels = np.array([item[1] for item in unique_files.values()])
        paths = np.array([item[2] for item in unique_files.values()])

        return images, labels, paths

    def _process_single_dataset(
        self, data_dir: Path, file_lst: List[Path]
    ) -> Tuple[List[Path], np.ndarray, np.ndarray]:
        """Process a single dataset directory."""
        label_file = next((f for f in file_lst if f.name == "labels.npy"), None)
        path_file = next((f for f in file_lst if f.name == "paths.npy"), None)

        if label_file is None:
            raise ValueError(f"labels.npy not found in {data_dir}")

        images = [f for f in file_lst if f not in (label_file, path_file)]
        labels = np.load(label_file)
        paths = np.load(path_file, allow_pickle=True)

        if len(images) != len(labels):
            raise ValueError(
                f"Mismatch between number of images ({len(images)}) and labels ({len(labels)}) in {data_dir}"
            )

        return images, labels, paths

    def __len__(self) -> int:
        """Return the data set length."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """Get a dataset element."""
        img_path = self.images[idx]
        image = np.load(img_path)
        image = torch.from_numpy(image.astype(np.float32))

        if self.mean is not None:
            image = (image - self.mean) / self.std

        label = self.labels[idx]
        label = torch.tensor(int(label))
        path = self.paths[idx]

        sample = {self.key: image, "label": label, "file_path": path}
        return sample


class NumpyDataset(Dataset):
    """Create a data loader to load pre-processed numpy arrays into memory."""

    def __init__(
        self,
        data_dir: str,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        key: Optional[str] = "image",
    ):
        """Create a Numpy-dataset object.

        Args:
            data_dir: A path to a pre-processed folder with numpy files.
            mean: Pre-computed mean to normalize with. Defaults to None.
            std: Pre-computed standard deviation to normalize with. Defaults to None.
            key: The key for the input or 'x' component of the dataset.
                Defaults to "image".

        Raises:
            ValueError: If an unexpected file name is given or directory is empty.

        # noqa: DAR401
        """
        self.data_dir = Path(data_dir)
        print(f"Loading data from {self.data_dir}")

        # Find all .npy files
        self.file_lst = sorted(self.data_dir.glob("*.npy"))

        if not self.file_lst:
            raise ValueError(f"No .npy files found in {self.data_dir}")

        # Find the labels file
        self.label_file = next(
            (f for f in self.file_lst if f.name == "labels.npy"), None
        )
        self.path_file = next((f for f in self.file_lst if f.name == "paths.npy"), None)
        if self.label_file is None:
            raise ValueError(f"labels.npy not found in {self.data_dir}")

        # Remove the labels file from the image list
        self.images = [
            f for f in self.file_lst if f not in (self.label_file, self.path_file)
        ]

        # Load labels
        self.labels = np.load(self.label_file)
        self.paths = np.load(self.path_file, allow_pickle=True)

        # Validate that we have the same number of images and labels
        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Mismatch between number of images ({len(self.images)}) and labels ({len(self.labels)})"
            )

        self.mean = mean
        self.std = std
        self.key = key

    def __len__(self) -> int:
        """Return the data set length."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """Get a dataset element.

        Args:
            idx (int): The element index of the data pair to return.

        Returns:
            [dict]: Returns a dictionary with the self.key
                    default ("image") and "label" keys.
        """
        img_path = self.images[idx]
        image = np.load(img_path)
        image = torch.from_numpy(image.astype(np.float32))
        # normalize the data.
        if self.mean is not None:
            image = (image - self.mean) / self.std
        label = self.labels[idx]
        label = torch.tensor(int(label))
        path = self.paths[idx]
        sample = {self.key: image, "label": label, "file_path": path}
        return sample


class DoubleDataset(Dataset):
    """Load data from two Numpy datasets using a single object."""

    def __init__(self, dataset1: NumpyDataset, dataset2: NumpyDataset):
        """Create a merged dataset, combining two numpy datasets based on the same image paths.

        Args:
            dataset1 (NumpyDataset): The first dataset (e.g., wavelet data).
            dataset2 (NumpyDataset): The second dataset (e.g., pixel data).
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len = len(dataset1)

        # Ensure that both datasets have matching image paths
        self._validate_paths()

    def _validate_paths(self):
        """Ensure that the image paths in both datasets match."""
        if not np.array_equal(self.dataset1.paths, self.dataset2.paths):
            raise ValueError("Datasets have mismatched image paths!")

    def __len__(self) -> int:
        """Return the dataset length."""
        return self.len

    def __getitem__(self, idx: int) -> dict:
        """Get a dataset element.

        Args:
            idx (int): The index of the data pair to return.

        Returns:
            dict: A dictionary containing images from both datasets and the corresponding label.
        """
        # Retrieve images from both datasets using the same index
        image1 = self.dataset1.__getitem__(idx)[self.dataset1.key]
        image2 = self.dataset2.__getitem__(idx)[self.dataset2.key]

        # Use the label from the first dataset (assuming labels are the same)
        label = self.dataset1.__getitem__(idx)["label"]

        # Return both images and the label
        return {
            "image1": image1,  # Image from dataset1
            "image2": image2,  # Image from dataset2
            "label": label,
        }


class CombinedDataset(Dataset):
    """Load data from two Numpy datasets using a single object."""

    def __init__(self, dataset1: NumpyDataset, dataset2: NumpyDataset):
        """Create a merged dataset, combining two numpy datasets based on the same image paths.

        Args:
            dataset1 (NumpyDataset): First NumpyDataset object.
            dataset2 (NumpyDataset): Second NumpyDataset object.
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # Ensure both datasets have the same paths
        if not all(
            path1 == path2 for path1, path2 in zip(dataset1.paths, dataset2.paths)
        ):
            raise ValueError("Datasets have different paths.")

        self.len = len(dataset1)  # Both datasets should have the same length
        self.key = "image"

    def __len__(self) -> int:
        """Return the dataset length."""
        return self.len

    def __getitem__(self, idx: int) -> dict:
        """Get a dataset element.

        Args:
            idx (int): The element index of the data pair to return.

        Returns:
            dict: Returns a dictionary with concatenated images from both datasets and "label".
        """
        image1 = self.dataset1.__getitem__(idx)[self.dataset1.key]
        image1 = image1.permute(1, 2, 0, 3).reshape(112, 112, -1)
        image2 = self.dataset2.__getitem__(idx)[self.dataset2.key]

        # print(image1.shape, image2.shape)

        # image2 = image2.unsqueeze(-1).expand(-1, -1, 3)
        # image2 = image2.unsqueeze(-1).expand(-1, -1, 3)

        concatenated_image = torch.cat(
            (image1, image2), dim=-1
        )  # Concatenate along channel dimension
        label = self.dataset1.__getitem__(idx)[
            "label"
        ]  # Use label from the first dataset

        return {self.key: concatenated_image, "label": label, "file_path": idx}


def main():
    """Compute dataset mean and standard deviation and store it."""
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Calculate mean and std")
    parser.add_argument(
        "dir",
        type=str,
        help="path of training data for which mean and std are computed",
    )
    args = parser.parse_args()

    print(args)

    data = NumpyDataset(args.dir)

    def compute_mean_std(data_set: Dataset) -> tuple:
        """Compute mean and stad values by looping over a dataset.

        Args:
            data_set (Dataset): A torch style dataset.

        Returns:
            tuple: the raw_data, as well as mean and std values.
        """
        # compute mean and std
        img_lst = []
        for img_no in range(data_set.__len__()):  # type: ignore[attr-defined]
            img_lst.append(data_set.__getitem__(img_no)["image"])
        img_data = torch.stack(img_lst, 0)

        # average all axis except the color channel
        axis = tuple(np.arange(len(img_data.shape[:-1])))
        # calculate mean and std in double to avoid precision problems
        mean = torch.mean(img_data.double(), axis).float()
        std = torch.std(img_data.double(), axis).float()
        return img_data, mean, std

    data, mean, std = compute_mean_std(data)

    print("mean", mean)
    print("std", std)
    file_name = f"{args.dir}/mean_std.pkl"
    with open(file_name, "wb") as f:
        pickle.dump([mean.numpy(), std.numpy()], f)
    print("stored in", file_name)


if __name__ == "__main__":
    main()
