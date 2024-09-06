import numpy as np
import torch
import torch.nn as nn

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode


class WhiteMul(ClusteringLayer, nn.Module):
    def __init__(self, in_features: int, set_codebook=None) -> None:
        super(WhiteMul, self).__init__()

        self.in_features = in_features

        self.codebook = np.zeros((in_features), dtype=np.object)
        self.last_codebook = np.zeros(in_features, dtype=np.object)
        self.previous_codebook = None

        self.white_table = np.zeros(in_features, dtype=np.object)
        self.quantization_count = None
        self.forward_mode = Forward_Mode.normal
        self.set_codebook = set_codebook

    def _whitebox_forward(self, input) -> torch.Tensor:
        left_input, right_input = input
        left_input = left_input.cpu().numpy().astype(np.uint8)
        right_input = right_input.cpu().numpy().astype(np.uint8)

        out = np.zeros((left_input.shape[0], self.in_features), dtype=np.uint8)

        for i in range(self.in_features):
            out[:, i] = self.white_table[i][left_input[:, i], right_input[:, i]]

        return torch.Tensor(out)

    def _normal_forward(self, input: torch.Tensor) -> torch.Tensor:
        left_input, right_input = input
        out = left_input * right_input
        return out
