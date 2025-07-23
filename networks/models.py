"""This module contains code for deepfake detection models."""

import numpy as np
import torch
from torch import nn
from resnet import ResNet50
import torchvision.models as models


def compute_parameter_total(net: torch.nn.Module) -> int:
    """Compute the parameter total of the input net.

    Args:
        net (torch.nn.Module): The model containing the
            parameters to count.

    Returns:
        int: The parameter total.
    """
    total = 0
    for p in net.parameters():
        if p.requires_grad:
            print(p.shape)
            total += np.prod(p.shape)  # type: ignore
    return total


class CNN(torch.nn.Module):
    """CNN models used for packet or pixel classification."""

    def __init__(self, classes: int, feature: str = "image"):
        """Create a convolutional neural network (CNN) model.

        Args:
            classes (int): The number of classes or sources to classify.
            feature (str)): A string which tells us the input feature
                we are using.
        """
        super().__init__()
        self.feature = feature
        print("feature", feature)

        if feature == "packets":
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(192, 24, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(24, 24, 6),
                torch.nn.ReLU(),
                torch.nn.Conv2d(24, 24, 9),
                torch.nn.ReLU(),
            )
            self.linear = torch.nn.Linear(24, classes)
        elif feature == "all-packets" or feature == "all-packets-fourier":
            if feature == "all-packets-fourier":
                self.scale1 = torch.nn.Sequential(
                    torch.nn.Conv2d(6, 8, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AvgPool2d(2, 2),
                )
            else:
                self.scale1 = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 8, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AvgPool2d(2, 2),
                )
            self.scale2 = torch.nn.Sequential(
                torch.nn.Conv2d(20, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(2, 2),
            )
            self.scale3 = torch.nn.Sequential(
                torch.nn.Conv2d(64, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(2, 2),
            )
            self.scale4 = torch.nn.Sequential(
                torch.nn.Conv2d(224, 32, 3, 1, padding=1), torch.nn.ReLU()
            )
            self.linear = torch.nn.Linear(32 * 16 * 16, classes)
        else:
            # assume an 128x128x3 image input.
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 8, 3),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(2, 2),
                torch.nn.Conv2d(8, 16, 3),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(2, 2),
                torch.nn.Conv2d(16, 32, 3),
                torch.nn.ReLU(),
            )
            self.linear = torch.nn.Linear(32 * 28 * 28, classes)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x) -> torch.Tensor:
        """Compute the CNN forward pass.

        Args:
            x (torch.Tensor or dict): An input image of shape
                [batch_size, packets, height, width, channels]
                for packet inputs and
                [batch_size, height, width, channels]
                else.

        Returns:
            torch.Tensor: A logsoftmax scaled output of shape
                [batch_size, classes].

        """
        # x = generate_packet_image_tensor(x)
        if self.feature == "packets":
            # batch_size, packets, height, width, channels
            shape = x.shape
            # batch_size, height, width, packets, channels
            x = x.permute([0, 2, 3, 1, 4])
            # batch_size, height, width, packets*channels
            to_net = x.reshape([shape[0], shape[2], shape[3], shape[1] * shape[4]])
            # HACK: batch_size, packets*channels, height, width
        elif self.feature == "all-packets":
            to_net = x["raw"]
        elif self.feature == "all-packets-fourier":
            to_net = torch.cat([x["raw"], x["fourier"]], dim=-1)
        else:
            to_net = x

        to_net = to_net.permute([0, 3, 1, 2])

        if self.feature == "all-packets" or self.feature == "all-packets-fourier":
            res = self.scale1(to_net)
            packets = [
                torch.reshape(
                    x[key].permute([0, 2, 3, 1, 4]),
                    [x[key].shape[0], x[key].shape[2], x[key].shape[3], -1],
                ).permute(0, 3, 1, 2)
                for key in ["packets1", "packets2", "packets3"]
            ]
            # shape: batch_size, packet_channels, height, widht, color_channels
            # cat along channel dim1.
            to_net = torch.cat([packets[0], res], dim=1)
            res = self.scale2(to_net)
            to_net = torch.cat([packets[1], res], dim=1)
            res = self.scale3(to_net)
            to_net = torch.cat([packets[2], res], dim=1)
            out = self.scale4(to_net)
            out = torch.reshape(out, [out.shape[0], -1])
            out = self.linear(out)
        else:
            out = self.layers(to_net)
            out = torch.reshape(out, [out.shape[0], -1])
            out = self.linear(out)
        return self.logsoftmax(out)


class Regression(torch.nn.Module):
    """A shallow linear-regression model."""

    def __init__(self, classes: int):
        """Create the regression model.

        Args:
            classes (int): The number of classes or sources to classify.
        """
        super().__init__()
        self.linear = torch.nn.Linear(49152, classes)

        # self.activation = torch.nn.Sigmoid()
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the regression forward pass.

        Args:
            x (torch.Tensor): An input tensor of shape
                [batch_size, ...]

        Returns:
            torch.Tensor: A logsoftmax scaled output of shape
                [batch_size, classes].
        """
        x_flat = torch.reshape(x, [x.shape[0], -1])
        return self.logsoftmax(self.linear(x_flat))


def save_model(model: torch.nn.Module, path):
    """Save the state dict of the model to the specified path.

    Args:
        model (torch.nn.Module): model to store
        path: file path of the storage file
    """
    torch.save(model.state_dict(), path)


def initialize_model(model: torch.nn.Module, path):
    """Initialize the given model from a stored state dict file.

    Args:
        model (torch.nn.Module): model to initialize
        path: file path of the storage file
    """
    model.load_state_dict(torch.load(path))


## Cross attention models
# ResNet model with cross-attention mechanism for late fusion of two datasets
class CrossAttentionModel(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        """Model with cross-attention between two datasets."""
        super(CrossAttentionModel, self).__init__()
        # ResNet models for both datasets
        # wavelet
        self.resnet1 = ResNet50(2, 3)
        # images
        self.resnet2 = models.resnet50(pretrained=False)
        self.resnet2 = nn.Sequential(*list(self.resnet2.children())[:-1])  # remove F
        # self.resnet2.conv1 = nn.Conv2d(
        #     12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        # )

        # Cross-attention layer
        self.cross_attention1 = CrossAttention(feature_dim, num_heads=num_heads)
        self.cross_attention2 = CrossAttention(feature_dim, num_heads=num_heads)

        # Classifier after attention
        self.fc = nn.Linear(
            feature_dim + feature_dim, 2
        )  # Assume binary classification

    def forward(self, image1, image2):
        image2 = image2.permute([0, 3, 1, 2])

        # Extract features from both images
        features1 = self.resnet1(image1)  # Output shape: (batch_size, feature_dim)
        features2 = self.resnet2(image2)  # Output shape: (batch_size, feature_dim)

        # Flatten the features
        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)

        # Reshape for cross-attention (add sequence dimension for attention to work)
        features1 = features1.unsqueeze(0)  # Shape: (1, batch_size, feature_dim)
        features2 = features2.unsqueeze(0)  # Shape: (1, batch_size, feature_dim)

        # Apply cross-attention: image1 features attend to image2 features, and vice versa
        attn_output1 = self.cross_attention1(features1, features2, features2)
        attn_output2 = self.cross_attention2(features2, features1, features1)

        # Remove sequence dimension and combine both attention outputs
        combined_features = torch.cat(
            [attn_output1.squeeze(0), attn_output2.squeeze(0)], dim=1
        )

        # Pass through a fully connected layer for classification
        output = self.fc(combined_features)

        return output


# Cross-attention mechanism
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        """Initialize cross-attention mechanism.

        Args:
            embed_dim (int): Dimensionality of the embeddings.
            num_heads (int): Number of attention heads.
        """
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        """Forward pass for cross-attention.

        Args:
            query (Tensor): The query tensor, typically from dataset1's features.
            key (Tensor): The key tensor, typically from dataset2's features.
            value (Tensor): The value tensor, typically from dataset2's features.

        Returns:
            Tensor: The output of the cross-attention mechanism.
        """
        # Apply multi-head attention: Query attends to the key-value pairs.
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output


class LateFusionResNet(nn.Module):
    def __init__(self, num_classes):
        super(LateFusionResNet, self).__init__()

        # Load two ResNet models (e.g., ResNet18 and ResNet50)
        # wavelet
        self.resnet1 = ResNet50(2, 3)
        # images
        self.resnet2 = models.resnet50(pretrained=False)

        # Remove the final fully connected layers from both
        # self.resnet1 = nn.Sequential(
        # *list(self.resnet1.children())[:-1]
        # )  # remove FC layer
        self.resnet2 = nn.Sequential(
            *list(self.resnet2.children())[:-1]
        )  # remove FC layer

        # Fusion layer: Concatenate features from both models
        self.fc = nn.Linear(
            2048 + 2048, num_classes
        )  # 512 for ResNet18 and 2048 for ResNet50

    def forward(self, x1, x2):
        # Pass inputs through both ResNet models
        features1 = self.resnet1(x1)  # Output of ResNet50
        # print(x2.shape)
        x2 = x2.permute([0, 3, 1, 2])
        features2 = self.resnet2(x2)  # Output of ResNet50

        # Flatten the features from both ResNets
        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)

        # Concatenate the features (late fusion)
        combined_features = torch.cat((features1, features2), dim=1)

        # Final classification layer
        output = self.fc(combined_features)
        return output
