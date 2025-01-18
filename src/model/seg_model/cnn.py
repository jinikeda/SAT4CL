## author: Jin Ikeda
## create: 2025 1 17
## des: the simple CNNs (Convolutional Neural Networks)


import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn(nn.Module):
    def __init__(self, num_bands, num_classes, dropout_prob=0.2, pooling_Flag = False):
        super(cnn, self).__init__()

        self.name = 'cnn'
        self.pooling_Flag = pooling_Flag

        # Define convolutional layers as a sequential module
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(num_bands, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2) if pooling_Flag else nn.Identity(),  # Conditional pooling
                nn.Dropout(p=dropout_prob)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2) if pooling_Flag else nn.Identity(),  # Conditional pooling
                nn.Dropout(p=dropout_prob)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2) if pooling_Flag else nn.Identity(),  # Conditional pooling
                nn.Dropout(p=dropout_prob)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2) if pooling_Flag else nn.Identity(),  # Conditional pooling
                nn.Dropout(p=dropout_prob)
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2) if pooling_Flag else nn.Identity(),  # Conditional pooling
                nn.Dropout(p=dropout_prob)
            ),
            nn.Sequential(
                nn.Conv2d(64, num_classes, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob)
            )
        )

    def forward(self, x):
        original_size = x.shape[2:]  # Save original spatial dimensions (height, width)
        # print(f"Input shape: {x.shape}")

        # Pass through all layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # print(f"After layer {i + 1} ({layer}): {x.shape}")

            if self.pooling_Flag:
                x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)

        # Apply activation for classification
        if x.size(1) > 1:
            x = F.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)

        return x
