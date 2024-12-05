import os
import torch
from resnet import ResNet50
from resnet import LateFusionResNet
from resnet import CrossAttentionModel
from train_classifier import create_data_loaders
from resnet import CrossAttentionModel, CrossAttentionModelFreq
from train_classifier import compute_energy_vector
import argparse

parser = argparse.ArgumentParser(description="args")
parser.add_argument("--cross", action="store_true"),
parser.add_argument(
    "--cross",
    action="store_true",
)
parser.add_argument(
    "--late",
    action="store_true",
)
parser.add_argument(
    "--ycbcr",
    action="store_true",
    help="convert images to YCbCr space",
)
parser.add_argument(
    "--perturbation",
    action="store_true",
    help="perturbation",
)
parser.add_argument(
    "--single-channel",
    action="store_true",
)

args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load all models and their datasets
models = {}
val_data_loaders = {}

for folder in os.listdir("./log"):
    folder_path = os.path.join("./log", folder)
    if os.path.isdir(folder_path) and folder != "ycbcr":
        model_name = folder_path.split("/")[-1]
        model_path = os.path.join(
            "./log",
            f"224_{model_name}_packets_haar_reflect_1_5e-05__resnet_v_d",
        )
        if args.ycbcr:
            model_path += "_ycbcr"
        if args.perturbation:
            model_path += "_perturbed"
        model_path += ".pt"
        # Load the model
        model = (
            ResNet50(2, 1).to("cuda")
            if args.ycbcr or args.single_channel
            else (
                CrossAttentionModel(2048).to("cuda")
                if args.cross
                else ResNet50(2, 3).to("cuda")
            )
        )
        #
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        print("folder", folder)
        models[folder] = model

        # Load the validation data loader for this model
        data_prefix = [
            # f"/home/nick/ff_crops/224_{model_name}_crops_packets_haar_reflect_1",
            f"/home/nick/Face/c40/output/224_{model_name}_packets_haar_reflect_1",
        ]
        _, val_data_loader, _ = create_data_loaders(
            data_prefix, 16, ycbcr=args.ycbcr, perturbation=False
        )
        val_data_loaders[folder] = val_data_loader

        print(
            f"Loaded model and validation data loader for: {folder} with folder {data_prefix}"
        )


def calculate_tpr_tnr(tp, fn, tn, fp):
    tpr = tp / (tp + fn) if tp + fn > 0 else 0
    tnr = tn / (tn + fp) if tn + fp > 0 else 0
    return tpr * 100, tnr * 100


# Cross-testing of each model on all validation datasets
for model_name, model in models.items():
    for dataset_name, val_loader in val_data_loaders.items():
        val_ok = 0.0
        val_total = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        positive_indices = []
        negative_indices = []
        loss_fun = torch.nn.CrossEntropyLoss()
        # val_test_loop(val_loader, model, loss_fun, ycbcr=True)
        with torch.no_grad():
            for batch_idx, val_batch in enumerate(val_loader):
                # batch_images = val_batch[val_loader.dataset.key].to(
                #     device, non_blocking=True
                # )
                # batch_labels = val_batch["label"].to(device, non_blocking=True)
                # if args.ycbcr:
                #     y_channel = batch_images[..., 0]
                #     batch_images = y_channel.unsqueeze(-1)
                # if args.single_channel:
                #     batch_images = batch_images[:, 3, :, :].unsqueeze(1)

                batch_images_1 = val_batch[val_loader.dataset.key].to(device)
                # batch_images_2 = val_batch["image2"].to(device)
                batch_labels = val_batch["label"].to(device)
                out = model(batch_images_1)
                batch_labels[batch_labels > 0] = 1
                out = model(batch_images_1)
                predicted_labels = torch.max(out, dim=-1)[1]
                # file_paths = val_batch["file_path"]

                # Track true positives, true negatives, false positives, false negatives
                for image_idx, (true_label, pred_label) in enumerate(
                    zip(batch_labels, predicted_labels)
                ):
                    if true_label == 1:
                        if pred_label == 1:
                            # positive_indices.append(file_paths[image_idx])
                            tp += 1
                        else:
                            fn += 1
                    else:
                        if pred_label == 0:
                            # negative_indices.append(file_paths[image_idx])
                            tn += 1
                        else:
                            fp += 1

                ok_mask = torch.eq(predicted_labels, batch_labels)
                val_ok += torch.sum(ok_mask).item()
                val_total += batch_labels.shape[0]

            accuracy = 100 * val_ok / val_total
            tpr, tnr = calculate_tpr_tnr(tp, fn, tn, fp)

        print(f"Model {model_name} on {dataset_name} dataset:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"True Positive Rate (TPR): {tpr:.2f}")
        print(f"True Negative Rate (TNR): {tnr:.2f}")

        # Save positive indices
        torch.save(
            positive_indices,
            f"./tpnr/{model_name}_{dataset_name}_positive_indices.pt",
        )
        torch.save(
            negative_indices,
            f"./tpnr/{model_name}_{dataset_name}_negative_indices.pt",
        )
