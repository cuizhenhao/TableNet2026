import numpy as np
import torch
import torch.nn as nn

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode


class WhiteLayerNorm(ClusteringLayer, nn.LayerNorm):
    def __init__(self, num_features, normalized_shape, is_last_layer=False) -> None:
        super(WhiteLayerNorm, self).__init__(normalized_shape)
        self.num_features = num_features
        self.last_codebook = np.zeros(num_features, dtype=np.object)
        self.codebook = np.zeros(num_features, dtype=np.object)
        self.previous_codebook = None
        self.white_table = np.zeros(num_features, dtype=np.object)
        self.quantization_count = None
        self.forward_mode = Forward_Mode.normal
        self.is_last_layer = is_last_layer

    def _whitebox_forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input.cpu().numpy().astype(np.uint8)

        for i in range(self.num_features):
            out[:, i] = self.white_table[i][out[:, i]]
        return torch.Tensor(out)

    def _normal_forward(self, input: torch.Tensor) -> torch.Tensor:
        self.eval()
        # 计算input的均值和方差
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True)
        normalized_tensor = (input - mean) / (std + self.eps)  # 添加一个小的常数以防止除零错误
        out = self.weight * normalized_tensor + self.bias  # 缩放和偏移操作
        return out
