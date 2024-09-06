import torch
from torch import nn

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode


class WhiteCat2D(ClusteringLayer, nn.Module):
    def __init__(self, left_in_channels: int, right_in_channels: int):
        super(WhiteCat2D, self).__init__()
        self.left_in_channels = left_in_channels
        self.right_in_channels = right_in_channels
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)
        self.previous_codebook = None

        self.quantization_count = None
        self.forward_mode = Forward_Mode.normal
        self.is_last_layer = False

    def _whitebox_forward(self, input) -> torch.Tensor:
        left, right = input
        right += (self.last_codebook[0][0] // 2)
        out = torch.cat([left, right], dim=-1)
        return out

    def _normal_forward(self, input) -> torch.Tensor:
        left, right = input
        out = torch.cat([left, right], dim=-1)
        return out
