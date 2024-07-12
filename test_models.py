import os
import torch
from resnet import ResNet50
from train_classifier import create_data_loaders

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load all models and their datasets
models = {}
val_data_loaders = {}

for folder in os.listdir("./log"):
    folder_path = os.path.join("./log", folder)
    if os.path.isdir(folder_path):
        model_path = os.path.join(
            folder_path, "224_freq_packets_haar_reflect_1_0.001_20e_resnet_42.pt"
        )

        # Load the model
        model = ResNet50(2, 3).to("cuda")

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        models[folder] = model

        # Load the validation data loader for this model
        data_prefix = [
            f"/home/nick/ff_crops/{folder}_crops/224_freq_packets_haar_reflect_1"
        ]
        _, val_data_loader, _ = create_data_loaders(data_prefix, 64)
        val_data_loaders[folder] = val_data_loader

        print(f"Loaded model and validation data loader for: {folder}")

# Cross-testing of each model on all validation datasets
for model_name, model in models.items():
    for dataset_name, val_loader in val_data_loaders.items():
        val_ok = 0.0
        val_total = 0
        with torch.no_grad():
            for val_batch in val_loader:
                batch_images = val_batch[val_loader.dataset.key].to(
                    "cuda", non_blocking=True
                )
                batch_labels = val_batch["label"].to("cuda", non_blocking=True)
                out = model(batch_images)
                batch_labels[batch_labels > 0] = 1
                loss_fun = torch.nn.CrossEntropyLoss()
                val_loss = loss_fun(torch.squeeze(out), batch_labels)
                ok_mask = torch.eq(torch.max(out, dim=-1)[1], batch_labels)
                val_ok += torch.sum(ok_mask).item()
                val_total += batch_labels.shape[0]
            accuracy = val_ok / val_total
        print(f"Model {model_name} on {dataset_name} dataset: {accuracy:.2f}% accuracy")
