import torch.nn as nn
from torchvision import models


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(
        self, ResBlock, layer_list, num_classes, num_channels=1, feature="packets"
    ):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.feature = feature
        if self.feature == "packets":
            # Modify input layer for packets
            self.conv1 = nn.Conv2d(
                1 * num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        elif self.feature == "image":
            # No modification needed for image input
            self.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            raise ValueError(
                "Invalid feature type. Supported types are 'packets' and 'image'."
            )

        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        shape = x.shape
        # print(shape)
        # Batch size, height, width, packets, channels
        x = x.permute([0, 2, 3, 1, 4])  # we don't ned to do that with combined dataset
        # Reshape to [batch_size, height, width, packets*channels]
        to_net = x.reshape([shape[0], shape[2], shape[3], shape[1] * shape[4]])
        to_net = to_net.permute([0, 3, 1, 2])  # change back to to_net for normal net
        # Apply initial convolutional layer
        x = self.relu(self.batch_norm1(self.conv1(to_net)))
        # x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.max_pool(x)

        # Proceed with the rest of the ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Perform average pooling and flatten
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # remove for late fusion
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * ResBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )

        layers.append(
            ResBlock(
                self.in_channels, planes, i_downsample=ii_downsample, stride=stride
            )
        )
        self.in_channels = planes * ResBlock.expansion

        for _ in range(1, blocks):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


class ResNetPre(nn.Module):
    def __init__(
        self, ResBlock, layer_list, num_classes, num_channels=1, feature="packets"
    ):
        super(ResNetPre, self).__init__()
        self.in_channels = 64
        self.feature = feature

        # Load the pretrained ResNet model from PyTorch
        pretrained_model = models.resnet50(pretrained=True)

        if self.feature == "packets":
            # Modify input layer for packets (change the number of input channels)
            self.conv1 = nn.Conv2d(
                1 * num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        elif self.feature == "image":
            # Use the standard ResNet input layer for images
            self.conv1 = pretrained_model.conv1
        else:
            raise ValueError(
                "Invalid feature type. Supported types are 'packets' and 'image'."
            )

        # Use pretrained batch norm and relu layers
        self.batch_norm1 = pretrained_model.bn1
        self.relu = pretrained_model.relu
        self.max_pool = pretrained_model.maxpool

        # Copy pretrained layers for deeper layers
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = pretrained_model.avgpool
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

        # Initialize other layers from pretrained model if needed
        self.load_pretrained_layers(pretrained_model)

    def load_pretrained_layers(self, pretrained_model):
        """Copy pretrained weights, excluding layers with shape mismatch."""
        # Replace the layers that have the same shape
        state_dict = pretrained_model.state_dict()

        # Remove conv1 weights if the number of input channels is different
        if self.feature == "packets":
            state_dict.pop("conv1.weight")

        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")

        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        shape = x.shape
        x = x.permute([0, 2, 3, 1, 4])  # adjust tensor dimensions
        to_net = x.reshape([shape[0], shape[2], shape[3], shape[1] * shape[4]])
        to_net = to_net.permute([0, 3, 1, 2])  # switch back to [N, C, H, W] format

        # Forward pass through the layers
        x = self.relu(self.batch_norm1(self.conv1(to_net)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * ResBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )

        layers.append(
            ResBlock(
                self.in_channels, planes, i_downsample=ii_downsample, stride=stride
            )
        )
        self.in_channels = planes * ResBlock.expansion

        for _ in range(1, blocks):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet50(num_classes, channels=192):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)


def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)


def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels)


# Now modify the ResNet class to include the cross-attention mechanism
