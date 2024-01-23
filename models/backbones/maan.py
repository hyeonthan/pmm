import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
import numpy as np

from models.backbones.embeddingblock import EmbeddingBlock
from models.backbones.pmm import PMM


class MAAN(nn.Module):
    def __init__(self, args):
        super(MAAN, self).__init__()
        self.args = args

        self.pmm_module = PMM(args)

        self.learnable_kernel = Parameter(torch.ones(args.MAL, 22, 1))
        self.feature_extractor = EmbeddingBlock(args)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240, args.num_classes),
        )
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240, args.num_domain),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x, _, _, _ = self.pmm_module(x)

        channel_feature_maps = torch.repeat_interleave(x, self.args.MAL, dim=1)
        attentive_map = torch.softmax(self.learnable_kernel, dim=1)

        refined_feature_maps = channel_feature_maps * attentive_map

        features = self.feature_extractor(refined_feature_maps)

        # features = self.feature_extractor(channel_feature_maps)
        class_logit = self.classifier(features)
        domain_prob = self.discriminator(features.detach())

        return class_logit, domain_prob
