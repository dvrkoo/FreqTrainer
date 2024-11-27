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

# Set the seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        self.data_dir = data_dir
        self.file_lst = sorted(Path(data_dir).glob("./*.npy"))
        if not self.file_lst:
            raise ValueError(f"No .npy files found in {self.data_dir}")

        # Find the labels file
        self.label_file = next(
            (f for f in self.file_lst if f.name == "labels.npy"), None
        )
        self.path_file = next((f for f in self.file_lst if f.name == "paths.npy"), None)
        if self.label_file is None:
            raise ValueError(f"labels.npy not found in {self.data_dir}")
        if self.path_file is None:
            raise ValueError(f"paths.npy not found in {self.data_dir}")
        # Remove the labels file from the image list
        self.images = [
            f for f in self.file_lst if f not in (self.label_file, self.path_file)
        ]

        # Load labels
        self.labels = np.load(self.label_file)
        self.paths = np.load(self.path_file, allow_pickle=True)
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
        sample = {self.key: image, "label": label}
        return sample


class CombinedDataset(Dataset):
    """Load data from multiple Numpy datasets ensuring matching image paths."""

    def __init__(self, sets: list):
        """Create a merged dataset, ensuring that the image paths match.

        Args:
            sets (list): A list of NumpyDataset objects.
        """
        self.sets = sets
        self.len = len(sets[0])
        self._validate_paths()  # Ensure that paths match between datasets
        self.paths_dict = self._create_paths_dict()

    def _validate_paths(self):
        """Ensure that all datasets have matching image paths."""
        reference_paths = self.sets[0].paths
        for dataset in self.sets[1:]:
            if dataset.paths != reference_paths:
                raise ValueError("Datasets have mismatched image paths!")

    def _create_paths_dict(self):
        """Create a dictionary mapping paths to their index."""
        paths_dict = {}
        for dataset in self.sets:
            for idx in range(len(dataset)):
                path = dataset.paths[idx]
                if path in paths_dict:
                    raise ValueError(f"Path conflict found: {path}")
                paths_dict[path] = idx
        return paths_dict

    def __len__(self) -> int:
        """Return the dataset length."""
        return self.len

    def __getitem__(self, idx: int) -> dict:
        """Get a dataset element.

        Args:
            idx (int): The element index of the data pair to return.

        Returns:
            dict: Returns a dictionary with images from both datasets and "label".
        """
        path = self.sets[0].paths[idx]  # Use the path from the first dataset
        images = {}

        for i, dataset in enumerate(self.sets):
            dataset_idx = self.paths_dict[path]
            image = dataset.__getitem__(dataset_idx)[dataset.key]
            images[f"image_set_{i+1}"] = image  # Store each image in a separate key

        label = self.sets[0].__getitem__(idx)[
            "label"
        ]  # Use label from the first dataset

        return {"images": images, "label": label}


class CombinedDataset(Dataset):
    """Load data from multiple Numpy datasets using a single object."""

    def __init__(self, sets: list):
        """Create a merged dataset, combining many numpy datasets.

        Args:
            sets (list): A list of NumpyDataset objects.
        """
        self.sets = sets
        self.len = len(sets[0])
        self.paths_dict = self._create_paths_dict()

    def _create_paths_dict(self):
        """Create a dictionary mapping paths to their index."""
        paths_dict = {}
        for dataset in self.sets:
            for idx in range(len(dataset)):
                path = dataset.paths[idx]
                if path in paths_dict:
                    raise ValueError(f"Path conflict found: {path}")
                paths_dict[path] = idx
        return paths_dict

    def __len__(self) -> int:
        """Return the dataset length."""
        return self.len

    def __getitem__(self, idx: int) -> dict:
        """Get a dataset element.

        Args:
            idx (int): The element index of the data pair to return.

        Returns:
            dict: Returns a dictionary with concatenated images and "label".
        """
        path = self.sets[0].paths[idx]  # Use the path from the first dataset
        concatenated_images = []

        for dataset in self.sets:
            dataset_idx = self.paths_dict[path]
            image = dataset.__getitem__(dataset_idx)[dataset.key]
            concatenated_images.append(image)
        print("shape1", concatenated_images[0].shape())
        print("shape2", concatenated_images[1].shape())
        concatenated_images = torch.cat(
            concatenated_images, dim=1
        )  # Concatenate along channel dimension
        label = self.sets[0].__getitem__(idx)[
            "label"
        ]  # Use label from the first dataset

        return {"image": concatenated_images, "label": label}


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
