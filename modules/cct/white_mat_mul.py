import numpy as np
import torch
import torch.nn as nn

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode


class WhiteMatMul(ClusteringLayer, nn.Module):
    def __init__(self, in_features: int, out_features: int, mid_features: int, scale: float, activate_function=None,
                 **kwargs) -> None:
        super(WhiteMatMul, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.mid_features = mid_features
        self.scale = scale
        self.activation = activate_function
        self.kwargs = kwargs

        self.codebook = np.zeros((in_features), dtype=np.object)
        self.last_codebook = np.zeros(in_features, dtype=np.object)
        self.previous_codebook = None

        self.white_table = np.zeros(1, dtype=np.object)
        self.quantization_count = None
        self.forward_mode = Forward_Mode.normal

    def _whitebox_forward(self, input) -> torch.Tensor:
        left_input, right_input = input
        left_input = left_input.cpu().numpy().astype(np.uint8)
        right_input = right_input.cpu().numpy().astype(np.uint8)

        out = np.zeros((left_input.shape[0], self.in_features, self.out_features), dtype=np.uint8)

        mul_table = self.white_table[0]['mul']
        add_tables = self.white_table[0]['add']

        for i in range(self.in_features):
            for j in range(self.out_features):
                mul_results = []
                for k in range(self.mid_features):
                    mul_results.append(mul_table[left_input[:, i, k], right_input[:, k, j]])

                add_results = mul_results
                for table in add_tables:
                    new_add_results = []
                    for index in range(len(add_results) // 2):
                        new_add_results.append(table[add_results[index * 2], add_results[index * 2 + 1]])
                    add_results = new_add_results
                out[:, i, j] = add_results[0]
        return torch.Tensor(out)

    def _normal_forward(self, input: torch.Tensor) -> torch.Tensor:
        left_input, right_input = input
        out = left_input.matmul(right_input)
        if self.scale is not None:
            out = out * self.scale

        if self.activation is not None:
            out = self.activation(out, **self.kwargs)
        return out
