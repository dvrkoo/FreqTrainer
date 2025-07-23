import torch
import os
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

# Local imports from our refactored modules
from config import parse_args
from data_loader import create_data_loaders
from engine import train_one_epoch, evaluate
from utils import set_seed, get_model_path
from models import (
    CNN,
    Regression,
    ResNet50,
    LateFusionResNet,
    CrossAttentionModel,
    compute_parameter_total,
)


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
    """Main function to orchestrate the training and evaluation process."""
    args = parse_args()
    print("Script arguments:", args)

    # --- Setup ---
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    experiment = Experiment(
        api_key="WQRfjlovs7RSjYUmjlMvNt3PY", project_name="general", workspace="dvrkoo"
    )
    experiment.log_parameters(vars(args))
    experiment.set_name(f"{args.model}_{args.data_prefix[0].split('/')[-1]}")

    # --- Data ---
    train_loader, val_loader, test_loader = create_data_loaders(args)

    # --- Model ---
    # Simplified model selection
    if args.model == "resnet":
        in_channels = 1 if (args.ycbcr or args.single_channel) else 3
        if args.late:
            model = LateFusionResNet(num_classes=args.nclasses).to(device)
        elif args.cross:
            model = CrossAttentionModel(2048).to(device)  # Assuming feature dim
        else:
            model = ResNet50(num_classes=args.nclasses, channels=in_channels).to(device)
    elif args.model == "cnn":
        model = CNN(args.nclasses, args.features).to(
            device
        )  # Assuming features are handled in CNN
    else:
        model = Regression(args.nclasses).to(device)

    print(f"Model: {args.model}, Parameters: {compute_parameter_total(model):,}")

    # --- Loss and Optimizer ---
    loss_weights = (
        torch.tensor(args.class_weights).to(device) if args.class_weights else None
    )
    loss_fun = torch.nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # --- Training Loop ---
    best_val_acc = 0.0
    best_model_path = get_model_path(args)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fun, optimizer, device, args, epoch
        )
        experiment.log_metric("accuracy/train", train_acc, step=epoch)
        experiment.log_metric("loss/train", train_loss, step=epoch)

        if epoch % args.validation_interval == 0 and val_loader:
            val_acc, val_loss = evaluate(
                model, val_loader, loss_fun, device, args, description="Validation"
            )
            experiment.log_metric("accuracy/validation", val_acc, step=epoch)
            experiment.log_metric("loss/validation", val_loss, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"New best model saved to {best_model_path} with accuracy: {best_val_acc:.4f}"
                )

    # --- Final Evaluation ---
    print("Training finished. Running final evaluation on the test set...")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("No best model found. Using the last epoch's model for testing.")

    test_acc, test_loss = evaluate(
        model, test_loader, loss_fun, device, args, description="Testing"
    )
    experiment.log_metric("accuracy/test", test_acc)
    experiment.log_metric("loss/test", test_loss)

    log_model(experiment, model=model, model_name=args.model)
    print("Process complete.")


if __name__ == "__main__":
    main()
