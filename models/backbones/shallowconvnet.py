import torch.nn as nn
import torch
from models.backbones.layers import Conv2dWithConstraint


class ShallowConvNet(nn.Module):
    def __init__(self, args):
        super(ShallowConvNet, self).__init__()
        self.temporal = Conv2dWithConstraint(
            args.in_channels,
            args.depth,
            kernel_size=[1, args.sfreq // 10],
            padding="same",
            max_norm=2.0,
        )
        self.spatial = Conv2dWithConstraint(
            args.depth,
            args.depth,
            kernel_size=[args.num_channels, 1],
            padding="valid",
            max_norm=2.0,
        )
        self.bn = nn.BatchNorm2d(args.depth)

        self.avg_pool = nn.AvgPool2d(
            kernel_size=[1, int(args.sfreq * 0.3)], stride=[1, int(args.sfreq * 0.06)]
        )

        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = torch.square(x)
        x = self.avg_pool(x)
        x = torch.log(torch.clamp(x, min=1e-06))
        x = self.dropout(x)

        return x
