import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, args):
        super(EEGNet, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(1, args.F1, kernel_size=[1, args.sfreq // 2], bias=False),
            nn.BatchNorm2d(args.F1),
            nn.Conv2d(
                args.F1,
                args.F1,
                kernel_size=[args.num_channels, 1],
                groups=args.F1,
                bias=False,
            ),
            nn.BatchNorm2d(args.F1),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=[1, 4]),
            nn.Dropout2d(args.dropout_rate),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                args.F1,
                args.F1,
                kernel_size=[1, args.sfreq // 8],
                groups=args.F1,
                bias=False,
            ),
            nn.BatchNorm2d(args.F1),
            nn.ELU(inplace=True),
            nn.Conv2d(args.F1, args.F2, kernel_size=[1, 1], groups=1, bias=False),
            nn.BatchNorm2d(args.F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=[1, 8]),
            nn.Dropout2d(args.dropout_rate),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)

        return x
