import numpy as np
import torch
import torch.nn as nn

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode


class WhiteTranspose(ClusteringLayer, nn.Module):
    def __init__(self, in_channels, out_channels, dim0, dim1) -> None:
        super(WhiteTranspose, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim0 = dim0
        self.dim1 = dim1

        self.last_codebook = np.zeros(out_channels, dtype=np.object)
        self.codebook = np.zeros(out_channels, dtype=np.object)
        self.white_table = np.zeros(out_channels, dtype=np.object)

        self.previous_codebook = None
        self.quantization_count = None
        self.forward_mode = Forward_Mode.normal

    def _whitebox_forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input.transpose(self.dim0, self.dim1)
        out = out.cpu().numpy().astype(np.uint8)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                out[:, i, j] = self.white_table[i][j][out[:, i, j]]
        return torch.Tensor(out)

    def _normal_forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(self.dim0, self.dim1)
