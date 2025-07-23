"""Source code to train deepfake detectors in wavelet and pixel space."""

from torch.optim.lr_scheduler import LambdaLR
import argparse
import os
import pickle
from typing import Any, Tuple
from data_loader import DoubleDataset
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from data_loader import CombinedDataset, NumpyDataset
from models import CNN, Regression, compute_parameter_total
from resnet import ResNet50, LateFusionResNet, CrossAttentionModel

experiment = Experiment(
    api_key="WQRfjlovs7RSjYUmjlMvNt3PY", project_name="general", workspace="dvrkoo"
)
hyper_params = {
    "learning_rate": 1e-3,
    "batch_size": 16,
}

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


experiment.log_parameters(hyper_params)


def _parse_args():
    """Parse cmd line args for training an image classifier."""
    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument(
        "--cross",
        action="store_true",
    )
    parser.add_argument(
        "--late",
        action="store_true",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
    )
    parser.add_argument(
        "--single-channel",
        action="store_true",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
    )
    parser.add_argument(
        "--features",
        choices=["raw", "packets", "all-packets", "fourier", "all-packets-fourier"],
        default="packets",
        help="the representation type",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="input batch size for testing",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="learning rate for optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="weight decay for optimizer (default: 0)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="number of epochs (default: 10)"
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=200,
        help="number of training steps after which the model is tested on the validation data set (default: 200)",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        nargs="+",
        default=["/home/nick/ff_crops/224_neuraltextures_crops_packets_haar_reflect_1"],
        help="shared prefix of the data paths (default: ./data/source_data_packets)",
    )
    parser.add_argument(
        "--nclasses", type=int, default=2, help="number of classes (default: 2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="the random seed pytorch works with."
    )

    parser.add_argument(
        "--model",
        choices=["regression", "cnn", "resnet"],
        default="resnet",
        help="The model type chosse regression or CNN. Default: Regression.",
    )

    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="enables a tensorboard visualization.",
    )

    parser.add_argument(
        "--pbar",
        action="store_true",
        help="enables progress bars",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes started by the test and validation data loaders. The training data_loader "
        "uses three times this argument many workers. Hence, this argument should probably be chosen below 10. "
        "Defaults to 2.",
    )

    parser.add_argument(
        "--class-weights",
        type=float,
        metavar="CLASS_WEIGHT",
        nargs="+",
        default=None,
        help="If specified, training samples are weighted based on their class "
        "in the loss calculation. Expects one weight per class.",
    )
    # add ycbcr option
    parser.add_argument(
        "--ycbcr",
        action="store_true",
        help="convert images to YCbCr space",
    )
    parser.add_argument(
        "--perturbation",
        action="store_true",
        help=" perturbed images ",
    )

    # one should not specify normalization parameters and request their calculation at the same time
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--calc-normalization",
        action="store_true",
        help="calculates mean and standard deviation used in normalization"
        "from the training data",
    )
    return parser.parse_args()


def val_test_loop(
    data_loader,
    model: torch.nn.Module,
    loss_fun,
    make_binary_labels: bool = True,
    _description: str = "Validation",
    pbar: bool = False,
    ycbcr: bool = False,
    single_channel: bool = False,
) -> Tuple[float, Any]:
    """Test the performance of a model on a data set by calculating the prediction accuracy and loss of the model.

    Args:
        e.g. a test or validation set in a data split.
        data_loader (DataLoader): A DataLoader loading the data set on which the performance should be measured,
        model (torch.nn.Module): The model to evaluate.
        loss_fun: The loss function, which is used to measure the loss of the model on the data set
        make_binary_labels (bool): If flag is set, we only classify binarily, i.e. whether an image is real or fake.
            In this case, the label 0 encodes 'real'. All other labels are cosidered fake data, and are set to 1.

    Returns:
        Tuple[float, Any]: The measured accuracy and loss of the model on the data set.
    """
    with torch.no_grad():
        model.eval()
        val_total = 0

        val_ok = 0.0
        for val_batch in iter(data_loader):
            # if type(data_loader.dataset) is CombinedDataset:
            #     batch_images = {
            #         key: val_batch[key].to("cuda", non_blocking=True)
            #         for key in data_loader.dataset.key
            #     }
            # else:
            # TODO: uncomment if not late fusion
            batch_images = val_batch[data_loader.dataset.key].to(
                "cuda", non_blocking=True
            )
            batch_labels = val_batch["label"].to("cuda", non_blocking=True)
            # if ycbcr:
            #     y_channel = batch_images[..., 0]
            #     batch_images = y_channel.unsqueeze(-1)
            #     # batch_images = extract_y_channel(batch_images)
            # if single_channel:
            #     first_band = batch_images[:, 3, :, :]
            #     batch_images = first_band.unsqueeze(1)
            #     # if args.late:
            # image_set_1 = val_batch["image1"].to("cuda")
            # image_set_2 = val_batch["image2"].to("cuda")
            # batch_labels = val_batch["label"].to("cuda")
            # out = model(image_set_1, image_set_2)
            out = model((batch_images))
            if make_binary_labels:
                batch_labels[batch_labels > 0] = 1
            val_loss = loss_fun(torch.squeeze(out), batch_labels)
            ok_mask = torch.eq(torch.max(out, dim=-1)[1], batch_labels)
            val_ok += torch.sum(ok_mask).item()
            val_total += batch_labels.shape[0]
        val_acc = val_ok / val_total
        print("acc", val_acc, "ok", val_ok, "total", val_total)
    return val_acc, val_loss


def custom_collate(batch):
    # Assuming the key for images is 'image', adjust if it's different
    # print(batch[0].keys())
    key = "packets1"  # or whatever key your dataset uses for images
    images = torch.stack([item[key] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    file_paths = [item["file_path"] for item in batch]
    return {key: images, "label": labels, "file_path": file_paths}


def upscale_wavelet_packets(batch):
    # Assume batch is of shape [B, 4, 112, 112, 3]
    # B is the batch size
    B = batch.shape[0]

    # Reshape to [B*4, 3, 112, 112] to treat each packet as a separate image
    reshaped = batch.permute(0, 1, 4, 2, 3).reshape(-1, 3, 112, 112)

    # Upscale using bilinear interpolation
    upscaled = F.interpolate(
        reshaped, size=(224, 224), mode="bilinear", align_corners=False
    )

    # Reshape back to [B, 4, 224, 224, 3]
    result = upscaled.view(B, 4, 3, 224, 224).permute(0, 1, 3, 4, 2)

    return result


def get_suffix(perturbation, ycbcr):
    suffix = ""
    if ycbcr:
        suffix += "_ycbcr"
    if perturbation:
        suffix += "_perturbed"
    return suffix


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
        print(len(test_data_set))
        data_set_list.append((train_data_set, val_data_set, test_data_set))

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

    elif len(data_set_list) == 2:
        # Combine datasets
        train_data_loader = DataLoader(
            DoubleDataset(data_set_list[0][0], data_set_list[1][0]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
        )
        val_data_loader = DataLoader(
            DoubleDataset(data_set_list[0][1], data_set_list[1][1]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=3,
        )
        test_data_set = CombinedDataset(data_set_list[0][2], data_set_list[1][2])
    else:
        raise RuntimeError("Failed to load data from the specified prefixes.")
    return train_data_loader, val_data_loader, test_data_set


def compute_energy_for_batch(batch):
    """
    Compute the energy vector for each image in a batch.

    Args:
        batch (Tensor): A batch of images with shape (batch_size, num_channels, height, width).

    Returns:
        Tensor: A tensor containing the energy vector for each image in the batch.
                Shape: (batch_size, num_channels)
    """
    # Compute the energy for each channel (band) in the image
    energy_vector = torch.sum(
        batch**2, dim=[2, 3]
    )  # Sum over height and width dimensions

    return energy_vector


# Example usage:
# Assume freq1, freq2, freq3, and freq4 are tensors of shape (batch_size, 1, height, width)
def compute_energy_vector(freq1, freq2, freq3, freq4):
    """
    Compute the combined energy vector for a batch of images across four frequency bands.

    Args:
        freq1, freq2, freq3, freq4 (Tensor): Tensors of shape (batch_size, 1, height, width)
                                             representing different frequency bands.

    Returns:
        Tensor: A tensor containing the energy vectors for each image in the batch.
                Shape: (batch_size, 4)
    """
    # Compute energy for each frequency band
    energy1 = compute_energy_for_batch(freq1)
    energy2 = compute_energy_for_batch(freq2)
    energy3 = compute_energy_for_batch(freq3)
    energy4 = compute_energy_for_batch(freq4)

    # Stack energies to create the energy vector
    energy_vector = torch.cat(
        [energy1, energy2, energy3, energy4], dim=1
    )  # Shape: (batch_size, 4)

    return energy_vector


def main():
    """Trains a model to classify images."""
    args = _parse_args()
    print(args)
    experiment.set_name(
        args.data_prefix[0].split("/")[-1]
        + get_suffix(args.perturbation, args.ycbcr)
        + "_a_h_v_d"
    )

    if args.class_weights and len(args.class_weights) != args.nclasses:
        raise ValueError(
            f"The number of class_weights ({len(args.class_weights)}) must equal "
            f"the number of classes ({args.nclasses})"
        )

    # Fix the seed for reproducible results.
    torch.manual_seed(args.seed)

    make_binary_labels = args.nclasses == 2
    train_data_loader, val_data_loader, test_data_set = create_data_loaders(
        args.data_prefix, args.batch_size, args.ycbcr, args.perturbation
    )
    # if args.late:
    #     train_image_data_loader, val_image_data_loader, test_image_data_loader = (
    #         create_data_loaders(
    #             args.data_prefix[1], args.batch_size, args.ycbcr, args.perturbation
    #         )
    #     )

    validation_list = []
    loss_list = []
    accuracy_list = []
    step_total = 0
    best_val_acc = 0  # Initialize the best validation accuracy
    best_model_path = ""  # Initialize the path to save the best model

    if args.model == "cnn":
        model = CNN(args.nclasses, args.features).to("cuda")
        print("feature is ", args.features)
    elif args.model == "resnet":
        if args.ycbcr or args.single_channel:
            model = ResNet50(2, 1).to("cuda")
        # elif args.cross:
        # model = ResNetWithAttention(Bottleneck, [3, 4, 6, 3], num_classes=10)
        if args.late:
            model = LateFusionResNet(num_classes=2).to("cuda")
        if args.cross:
            model = CrossAttentionModelFreq(2048).to("cuda")
        else:
            model = ResNet50(2, 3).to("cuda")
    else:
        model = Regression(args.nclasses).to("cuda")

    print("model parameter count:", compute_parameter_total(model))

    if args.tensorboard:
        writer_str = (
            f"runs/params_test2/{args.model}/FaceSwap/{args.batch_size}/"
            f"{args.data_prefix[0].split('/')[-1]}/{args.learning_rate}_{args.seed}"
        )
        writer = SummaryWriter(writer_str, max_queue=100)

    if args.class_weights:
        loss_fun = torch.nn.NLLLoss(weight=torch.tensor(args.class_weights).to("cuda"))
    else:
        # loss_fun = torch.nn.CrossEntropyLoss()
        loss_fun = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def warmup_scheduler(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs  # Linear warmup
        else:
            return 1.0  # Keep the learning rate constant or switch to another scheduler

    warmup_epochs = 10  # Number of warmup epochs

    # Learning rate scheduler
    # scheduler = LambdaLR(optimizer, lr_lambda=warmup_scheduler)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[5, 10], gamma=0.1
    )

    for e in tqdm(
        range(args.epochs), desc="Epochs", unit="epochs", disable=not args.pbar
    ):
        # Iterate over training data.
        for it, batch in enumerate(
            tqdm(
                iter(train_data_loader),
                desc="Training",
                unit="batches",
                disable=not args.pbar,
            )
        ):
            model.train()
            optimizer.zero_grad()

            if type(train_data_loader.dataset) is CombinedDataset:
                batch_images = {
                    key: batch[key].to("cuda", non_blocking=True)
                    for key in train_data_loader.dataset.key
                }
            else:
                batch_images = batch[train_data_loader.dataset.key].to(
                    "cuda", non_blocking=True
                )

                batch_labels = batch["label"].to("cuda", non_blocking=True)
            if args.ycbcr:
                y_channel = batch_images[..., 0]
                batch_images = y_channel.unsqueeze(-1)
            if args.single_channel:
                first_band = batch_images[:, 3, :, :]
                batch_images = first_band.unsqueeze(1)
                # batch_images = extract_y_channel(batch_images)
            if args.late or args.cross:
                # image_set_1 = batch["image1"][:, [2, 3], :, :].to("cuda")
                # image_set_2 = batch["image2"].to("cuda")
                freq1 = batch["packets1"][:, [0], :, :].to("cuda")
                freq2 = batch["packets1"][:, [1], :, :].to("cuda")
                freq3 = batch["packets1"][:, [2], :, :].to("cuda")
                freq4 = batch["packets1"][:, [3], :, :].to("cuda")
                energy_vector = compute_energy_vector(freq1, freq2, freq3, freq4)
                batch_labels = batch["label"].to("cuda")
                out = model(freq1, freq2, freq3, freq4, energy_vector)
            else:
                batch_images = batch[train_data_loader.dataset.key].to(
                    "cuda", non_blocking=True
                )
                out = model(batch_images)

            if make_binary_labels:
                batch_labels = batch["label"].to("cuda")
                batch_labels[batch_labels > 0] = 1

            if args.late or args.cross:
                out = model(image_set_1, image_set_2)
            else:
                out = model((batch_images))
            loss = loss_fun((out), batch_labels)
            # ok_mask = torch.eq(torch.max(out, dim=-1)[1], batch_labels)
            ok_mask = torch.eq(
                torch.max(out, dim=-1)[1], torch.argmax(batch_labels, dim=-1)
            )
            acc = torch.sum(ok_mask.type(torch.float32)) / len(batch_labels)

            if it % 10 == 0:
                print(
                    "e",
                    e,
                    "it",
                    it,
                    "total",
                    step_total,
                    "loss",
                    loss.item(),
                    "acc",
                    acc.item(),
                )
            loss.backward()
            optimizer.step()
            step_total += 1
            loss_list.append([step_total, e, loss.item()])
            accuracy_list.append([step_total, e, acc.item()])

            if args.tensorboard:
                writer.add_scalar("loss/train", loss.item(), step_total)
                writer.add_scalar("accuracy/train", acc.item(), step_total)
                if step_total == 0:
                    writer.add_graph(model, batch_images)

        # scheduler.step()
        val_acc, val_loss = val_test_loop(
            val_data_loader,
            model,
            loss_fun,
            make_binary_labels=make_binary_labels,
            pbar=args.pbar,
            ycbcr=args.ycbcr,
            cross=args.cross,
            single_channel=args.single_channel,
        )
        validation_list.append([step_total, e, val_acc])

        # Check if the current validation accuracy is the best one
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_file = (
                "./log/"
                + args.data_prefix[0].split("/")[-1]
                + "_"
                + str(args.learning_rate)
                + "_"
                # + f"{args.epochs}e"
                + "_"
                + str(args.model)
                + "_no_mean"
            )
            if args.ycbcr:
                model_file += "_ycbcr"
            if args.perturbation:
                model_file += "_perturbed"
            torch.save(model.state_dict(), model_file + ".pt")
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}")

        if args.tensorboard:
            writer.add_scalar("loss/validation", val_loss, e)
            writer.add_scalar("accuracy/validation", val_acc, e)
            writer.add_scalar("epochs", e, step_total)

    if not os.path.exists("./log/"):
        os.makedirs("./log/")
    model_file = (
        "./log/"
        + args.data_prefix[0].split("/")[-1]
        + "_"
        + str(args.learning_rate)
        + "_"
        + f"{args.epochs}e"
        + "_"
        + str(args.model)
    )
    if args.ycbcr:
        model_file += "_ycbcr"
    if args.perturbation:
        model_file += "_perturbed"
    if args.cross:
        model_file += "_cross"
    # save_model(model, model_file + "_" + str(args.seed) + ".pt")
    # print(model_file, " saved.")

    # Run over the test set.
    print("Training done testing....")
    # Load the best model for final evaluation (optional)
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")

    # if type(test_data_set) is list:
    #     test_data_set = CombinedDataset(test_data_set)
    #
    test_data_loader = DataLoader(
        test_data_set,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
    )
    with torch.no_grad():
        test_acc, test_loss = val_test_loop(
            test_data_loader,
            model,
            loss_fun,
            make_binary_labels=make_binary_labels,
            pbar=not args.pbar,
            _description="Testing",
            ycbcr=args.ycbcr,
            single_channel=args.single_channel,
        )
        print("test acc", test_acc)

    if args.tensorboard:
        writer.add_scalar("accuracy/test", test_acc, step_total)
        writer.add_scalar("loss/test", test_loss, step_total)

    log_model(experiment, model=model, model_name="TheModel")

    if args.tensorboard:
        writer.close()


def _save_stats(
    model_file: str,
    loss_list: list,
    accuracy_list: list,
    validation_list: list,
    test_acc: float,
    args,
    iterations_per_epoch: int,
):
    stats_file = model_file + "_" + str(args.seed) + ".pkl"
    try:
        res = pickle.load(open(stats_file, "rb"))
    except OSError as e:
        res = []
        print(
            e,
            "stats.pickle does not exist, \
              creating a new file.",
        )
    res.append(
        {
            "train_loss": loss_list,
            "train_acc": accuracy_list,
            "val_acc": validation_list,
            "test_acc": test_acc,
            "args": args,
            "iterations_per_epoch": iterations_per_epoch,
        }
    )
    pickle.dump(res, open(stats_file, "wb"))
    print(stats_file, " saved.")


if __name__ == "__main__":
    main()
