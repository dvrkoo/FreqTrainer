import torch
import os


def ensure_string_set(loaded_set):
    return set(str(item) for item in loaded_set)


def load_pos_neg_idx(model_name, dataset_name, pixel):
    prefix = "./tpnr/"
    if pixel:
        prefix = "../../FaceForensicsTrainer/tpnr/"

    pos_path_freq = f"{model_name}_{dataset_name}_positive_indices.pt"
    neg_path_freq = f"{model_name}_{dataset_name}_negative_indices.pt"
    neg_path_freq = prefix + neg_path_freq
    pos_path_freq = prefix + pos_path_freq
    if os.path.exists(pos_path_freq) and os.path.exists(neg_path_freq):
        return ensure_string_set((torch.load(pos_path_freq))), ensure_string_set(
            set(torch.load(neg_path_freq))
        )
    else:
        print(f"No positive indices found for {model_name} on {dataset_name}")
        return set()


# Example usage to find intersections
model_names = [
    "neuraltextures",
    "pixel_neuraltextures",
]  # Replace with actual model names
dataset_names = ["deepfake"]  # Replace with actual dataset names

positive_indices_sets = []
negative_indices_sets = []

for model_name in model_names:
    pixel = False
    print_pref = "Freq "
    if model_name.split("_")[0] == "pixel":
        model_name = model_name.split("_")[1]
        print_pref = "Pixel "
        pixel = True

    for dataset_name in dataset_names:
        positive_indices, negative_indices = load_pos_neg_idx(
            model_name, dataset_name, pixel
        )
        # print(positive_indices)
        print(print_pref, "positives: ", len(positive_indices))
        print(print_pref, "negatives: ", len(negative_indices))
        positive_indices_sets.append(positive_indices)
        negative_indices_sets.append(negative_indices)

# Find intersections
if positive_indices_sets:
    common_positive_indices = set.intersection(*positive_indices_sets)
    common_negative_indices = set.intersection(*negative_indices_sets)
    print(
        f"Common positive indices across all models and datasets: {len(common_positive_indices)}"
    )
    print(
        f"Common negative indices across all models and datasets: {len(common_negative_indices)}"
    )
    # print(common_positive_indices)
else:
    print("No positive indices sets to compare.")
