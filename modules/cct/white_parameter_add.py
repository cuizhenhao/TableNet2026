import numpy as np
import torch
import torch.nn as nn

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode
from torch.nn import Parameter


class WhiteParameterAdd(ClusteringLayer, nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(WhiteParameterAdd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.last_codebook = np.zeros(out_channels, dtype=np.object)
        self.codebook = np.zeros(out_channels, dtype=np.object)
        self.white_table = np.zeros(out_channels, dtype=np.object)
        self.weight = Parameter(torch.zeros((1, out_channels, in_channels)), requires_grad=False)

        self.previous_codebook = None
        self.quantization_count = None
        self.forward_mode = Forward_Mode.normal

    def _whitebox_forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input.cpu().numpy().astype(np.uint8)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                out[:, i, j] = self.white_table[i][j][out[:, i, j]]
        return torch.Tensor(out)

    def _normal_forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input + self.weight
        return out
