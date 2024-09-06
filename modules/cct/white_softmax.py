import numpy as np
import torch
import torch.nn as nn

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode


class WhiteSoftmax(ClusteringLayer, nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super(WhiteSoftmax, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.last_codebook = np.zeros(in_channels, dtype=np.object)
        self.codebook = np.zeros(in_channels, dtype=np.object)
        self.previous_codebook = None
        self.white_table = np.zeros(in_channels, dtype=np.object)
        self.quantization_count = None
        self.kwargs = kwargs
        self.forward_mode = Forward_Mode.normal

    def _whitebox_forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input.cpu().numpy().astype(np.uint8)
        for i in range(self.in_channels):
            sum = 0
            for j in range(self.out_channels):
                sum = self.white_table[0]['add'][j][sum, out[:, i, j]]
            for j in range(self.out_channels):
                out[:, i, j] = self.white_table[0]['softmax'][sum, out[:, i, j]]
        return torch.Tensor(out)

    def _normal_forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.softmax(dim=-1)
        out = torch.zeros_like(input)
        for i in range(self.in_channels):
            sum = torch.zeros_like(input[:, i, 0])
            for j in range(self.out_channels):
                sum = sum + torch.exp(input[:, i, j])
            for j in range(self.out_channels):
                out[:, i, j] = torch.exp(input[:, i, j]) / sum
        return out
