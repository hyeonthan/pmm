import torch
import torch.nn as nn
import numpy as np

from models.backbones.embeddingblock import EmbeddingBlock
from models.backbones.pmm import PMM


class DJDAN(nn.Module):
    def __init__(self, args):
        super(DJDAN, self).__init__()
        self.args = args

        self.pmm_module = PMM(args)

        self.feature_extractor = EmbeddingBlock(args)

        self.class_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240, args.num_classes),
        )
        self.conditional_discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240, args.num_domain),
            nn.Softmax(dim=1),
        )
        self.marginal_discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240, args.num_domain),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x, _, _, _ = self.pmm_module(x)

        features = self.feature_extractor(x)
        class_logits = self.class_classifier(features)

        pc_list = self.pred_prob_ditribution(class_logits.detach())
        product_features = self.product_operator(features.detach(), pc_list)

        conditional_logits = []  # return list
        for product_feature in product_features:
            conditional_logits.append(self.conditional_discriminator(product_feature))
        marginal_logits = self.marginal_discriminator(features.detach())

        return class_logits, conditional_logits, marginal_logits

    def product_operator(self, features, pc_list):
        features_list = []
        for pc in pc_list:
            reshape_pc = pc.view(-1, 1, 1, 1)

            features_list.append(features * reshape_pc)

        product_features = features_list
        return product_features

    def pred_prob_ditribution(self, x):
        pc_list = []
        for class_idx in range(self.args.num_classes):
            pc = np.array([x[i][class_idx].item() for i in range(x.size(0))])

            pc = torch.from_numpy(pc).type(torch.FloatTensor).to("cuda")
            pc_list.append(pc)

        return pc_list
