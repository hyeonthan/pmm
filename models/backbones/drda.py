import torch
import torch.nn as nn
import numpy as np
from models.backbones.pmm import PMM

from models.backbones.shallowconvnet import ShallowConvNet

torch.autograd.set_detect_anomaly(True)


class GENERATOR(nn.Module):
    def __init__(self, args):
        super(GENERATOR, self).__init__()
        self.args = args

        self.pmm_module = PMM(args)

        self.feature_extractor = ShallowConvNet(args)

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1860, 64),
        )
        self.fc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(64, 16),
        )
        self.fc3 = nn.Linear(64, args.num_classes)

    def forward(self, x):
        x, _, _, _ = self.pmm_module(x)
        features = self.feature_extractor(x)
        logits = self.fc1(features)
        representation = self.fc2(logits)
        class_logits = self.fc3(logits)

        return features, class_logits, representation


class DISCRIMINATOR(nn.Module):
    def __init__(self, args):
        super(DISCRIMINATOR, self).__init__()
        self.args = args

        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1860, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        domain_logits = self.discriminator(x)
        domain_out = self.sigmoid(domain_logits)

        return domain_out, domain_logits
