import numpy as np
import torch
import torch.nn as nn

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode

class WhiteActivation(ClusteringLayer, nn.Module):
    def __init__(self, num_features: int, activate_function,merge_last_codebook = False,set_codebook=None, **kwargs) -> None:
        super(WhiteActivation, self).__init__()
        self.num_features = num_features
        self.last_codebook = np.zeros(num_features, dtype=np.object)
        self.codebook = np.zeros(num_features, dtype=np.object)
        self.previous_codebook = None
        self.white_table = np.zeros(num_features, dtype=np.object)
        self.quantization_count = None
        self.activate_function = activate_function
        self.kwargs = kwargs
        self.forward_mode = Forward_Mode.normal
        self.merge_last_codebook = merge_last_codebook
        self.set_codebook = set_codebook

    def _whitebox_forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input.cpu().numpy().astype(np.uint8)
        for i in range(self.num_features):
            out[:, i] = self.white_table[i][out[:, i]]

        return torch.Tensor(out)

    def _normal_forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.activate_function(input, **self.kwargs)
        return out
