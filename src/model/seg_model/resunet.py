## author: Jin Ikeda
## create: 2024.12.02
## des: the ResUNet model.


import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# def conv1x1_bn_relu(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 1, 1, 0),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True)
#     )
#
# def conv3x3_bn_relu(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, 1, 1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True)
#     )
#
# def dwconv3x3_bn_relu(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
#             kernel_size=3, stride=1, padding=1, groups=in_channels),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True)
#     )

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()

        # Global Average Pooling: Reduces spatial dimensions to 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers for channel-wise attention
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # Reduce channels
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # Restore channels
            nn.Sigmoid(), # Sigmoid activation for channel weights
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # Batch size, Channels, Height, Width
        y = self.avg_pool(x).view(b, c)  # Output shape: (batch, channels)
        y = self.fc(y).view(b, c, 1, 1)  # Reshape to match input dimensions

        # Rescaling: Multiply input by attention weights
        return x * y.expand_as(x)  # Element-wise multiplication

# multiple scales by applying convolutions with different dilation (atrous) rates
class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rates=[1, 2, 4, 8, 16]): # to get complicated features rather than global context
        super(ASPP, self).__init__()

        # Dynamically create ASPP blocks based on the rates
        self.aspp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_dims, out_dims, kernel_size=3, stride=1,
                    padding=rate, dilation=rate
                ),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_dims)
            ) for rate in rates
        ])

        # Combine ASPP block outputs with a 1x1 convolution
        self.output = nn.Conv2d(len(rates) * out_dims, out_dims, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        # Pass input through each ASPP block
        aspp_outputs = [block(x) for block in self.aspp_blocks]

        # Concatenate the results along the channel dimension
        x_cat = torch.cat(aspp_outputs, dim=1)

        # Pass concatenated results through the final output layer
        out = self.output(x_cat)
        return out

    def _init_weights(self):
        # Initialize weights for Conv2D and BatchNorm layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# class ASPP(nn.Module):
#     def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
#         super(ASPP, self).__init__()
#
#         self.aspp_block1 = nn.Sequential(
#             nn.Conv2d(
#                 in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
#             ),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(out_dims),
#         )
#         self.aspp_block2 = nn.Sequential(
#             nn.Conv2d(
#                 in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
#             ),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(out_dims),
#         )
#         self.aspp_block3 = nn.Sequential(
#             nn.Conv2d(
#                 in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
#             ),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(out_dims),
#         )
#
#         self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
#         self._init_weights()
#
#     def forward(self, x):
#         x1 = self.aspp_block1(x)
#         x2 = self.aspp_block2(x)
#         x3 = self.aspp_block3(x)
#         out = torch.cat([x1, x2, x3], dim=1)
#         return self.output(out)
#
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


####-----------for the Resunet-----------####
class ResUnetPlusPlus(nn.Module):
    def __init__(self, num_bands, num_classes, filters=[16, 32, 64, 128, 256]):
        super(ResUnetPlusPlus, self).__init__()
        self.num_classes = num_classes

        # Input Layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(num_bands, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(num_bands, filters[0], kernel_size=3, padding=1)
        )

        # Encoder
        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        # Bridge
        self.aspp_bridge = ASPP(filters[3], filters[4])

        # Decoder
        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        # Output Layer
        self.aspp_out = ASPP(filters[1], filters[0])
        self.output_layer = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)

        logits = self.output_layer(x9)

        if self.num_classes > 1:
            out = F.softmax(logits, dim=1)  # Multi-class segmentation
        else:
            out = torch.sigmoid(logits)  # Binary segmentation

        return out

class ResUNet(nn.Module):
    def __init__(self, num_bands, num_classes, filters=[32, 64, 128, 256]):
        super(ResUNet, self).__init__()
        self.num_classes = num_classes

        # Input Layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(num_bands, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(num_bands, filters[0], kernel_size=3, padding=1)
        )

        # Encoder
        self.residual_conv_1 = ResidualConv(filters[0], filters[1], stride=2, padding=1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], stride=2, padding=1)

        # Bridge
        self.bridge = ResidualConv(filters[2], filters[3], stride=2, padding=1)

        # Decoder
        self.upsample_1 = Upsample(filters[3], filters[3], kernel=2, stride=2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], stride=1, padding=1)

        self.upsample_2 = Upsample(filters[2], filters[2], kernel=2, stride=2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], stride=1, padding=1)

        self.upsample_3 = Upsample(filters[1], filters[1], kernel=2, stride=2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], stride=1, padding=1)

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        # Input Layer
        x1 = self.input_layer(x) + self.input_skip(x)

        # Encoder
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)

        # Bridge
        x4 = self.bridge(x3)

        # Decoder
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        x10 = self.up_residual_conv3(x9)

        # Output Layer
        logits = self.output_layer(x10)

        # Activation
        if self.num_classes > 1:
            out = F.softmax(logits, dim=1)  # Multi-class segmentation
        else:
            out = torch.sigmoid(logits)  # Binary segmentation

        return out