import math
import torch.nn as nn


class EmbeddingBlock(nn.Module):
    def __init__(self, args):
        super(EmbeddingBlock, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                args.EMBEDDING.in_channels,
                6,
                kernel_size=[1, args.sfreq // 4],
                padding="same",
            ),
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 24, kernel_size=[args.num_channels, 1], padding="valid"),
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=[1, 8], stride=[1, 8]),
            nn.Dropout2d(args.dropout_rate // 2),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                24, 24, kernel_size=[1, args.sfreq // 8], groups=24, padding="same"
            ),
            nn.Conv2d(24, 24, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=[1, 12], stride=[1, 12]),
            nn.Dropout2d(args.dropout_rate),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)

        return x
