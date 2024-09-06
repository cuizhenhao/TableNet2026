import numpy as np
import torch
import torch.nn as nn

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode


class WhiteBatchNorm2d(ClusteringLayer, nn.BatchNorm2d):
    def __init__(self, num_features: int, activate_function = None, is_last_layer = False) -> None:
        super(WhiteBatchNorm2d, self).__init__(num_features)
        self.last_codebook = np.zeros(num_features, dtype=np.object)
        self.codebook = np.zeros(num_features, dtype=np.object)
        self.previous_codebook = None
        # self.track_running_stats = False # 白盒化重训练要加上这个，不在改动统计值（影响也不大）
        self.white_table = np.zeros(num_features, dtype=np.object)
        self.quantization_count = None
        self.activate_function = activate_function
        self.forward_mode = Forward_Mode.normal
        self.is_last_layer = is_last_layer

    def _whitebox_forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input.cpu().numpy().astype(np.uint8)
        if self.is_last_layer:
            new_outputs = np.zeros_like(out, dtype=np.float32)
        for i in range(self.num_features):
            out[:, i] = self.white_table[i][out[:, i]]

            if self.is_last_layer:
                new_outputs[:, i] = self.last_codebook[i][1][out[:, i]]
        if self.is_last_layer:
            return torch.Tensor(new_outputs)
        return torch.Tensor(out)

    def _normal_forward(self, input: torch.Tensor) -> torch.Tensor:
        out = nn.BatchNorm2d.forward(self, input)
        if self.activate_function is not None:
            out = self.activate_function(out)
        return out
