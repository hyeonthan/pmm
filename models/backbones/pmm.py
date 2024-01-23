import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class PMM(nn.Module):
    """Proxy-based masking module"""

    def __init__(self, args):
        super(PMM, self).__init__()

        self.args = args

        self.mask_score_encoder = nn.Sequential(
            nn.Conv2d(
                args.PMM.in_channels,
                args.dim,
                kernel_size=[1, args.sfreq // 16],
                padding="valid",
            ),
            nn.BatchNorm2d(args.dim),
            nn.Conv2d(
                args.dim,
                args.dim,
                kernel_size=[args.num_channels, 1],
                padding="valid",
            ),
            nn.BatchNorm2d(args.dim),
            nn.ELU(),
            nn.Dropout2d(args.dropout_rate),
        )

        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def get_scaling_score(self, temporal_score_map):
        temporal_score_map += 1
        temporal_score_map /= 2
        temporal_score_map = 1 - temporal_score_map

        return temporal_score_map

    def forward(self, x):
        related_signals = x[..., self.args.non_related_period :]
        non_related_signals = x[..., : self.args.non_related_period]

        _, _, _, T = related_signals.shape

        embedded_related_signals = self.mask_score_encoder(related_signals)
        embedded_non_related_signals = self.mask_score_encoder(non_related_signals)
        center_point = embedded_non_related_signals.mean(dim=-1, keepdim=True)

        temporal_score_map = self.cosine_similarity(
            center_point, embedded_related_signals
        )
        temporal_score_map = self.get_scaling_score(temporal_score_map)

        temporal_score_map = torch.unsqueeze(temporal_score_map, 1)
        interpolated_temporal_score_map = F.interpolate(
            temporal_score_map, size=(1, T), mode="bilinear"
        )

        output = related_signals * interpolated_temporal_score_map

        return output, _, interpolated_temporal_score_map, _
