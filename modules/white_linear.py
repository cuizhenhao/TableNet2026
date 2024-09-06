import numpy as np
import torch
import torch.nn as nn

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode

# 当input可以是2维度，也可以是3维，不过要求input所有编码是一样的
class WhiteLinear(ClusteringLayer, nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, activate_function=None,
                 is_last_layer=False, merge_last_codebook_flag=False, set_codebook=None) -> None:
        super(WhiteLinear, self).__init__(in_features, out_features, bias)

        self.codebook = np.zeros((out_features, in_features), dtype=np.object)
        self.last_codebook = np.zeros(out_features, dtype=np.object)
        self.previous_codebook = None
        self.merge_last_codebook_flag = merge_last_codebook_flag
        self.white_table = np.zeros(out_features, dtype=np.object)
        self.quantization_count = None
        self.activate_function = activate_function
        self.forward_mode = Forward_Mode.normal  # 0 for normal forward, 1 for clustering forward, 2 for whitebox forward
        self.is_last_layer = is_last_layer
        self.weight_mask = nn.Parameter(torch.ones(self.weight.shape).byte(), requires_grad=False)
        self.set_codebook = set_codebook
        if bias:
            self.bias_mask = nn.Parameter(torch.ones(self.bias.shape).byte(), requires_grad=False)

    def _whitebox_forward(self, input: torch.Tensor) -> torch.Tensor:
        input_arr = input.cpu().numpy().astype(np.uint8)
        out = np.zeros((*input.size()[:-1], self.out_features), dtype=np.uint8)
        if self.is_last_layer:
            new_outputs = np.zeros_like(out, dtype=np.float32)

        for i in range(self.out_features):
            for index, table in self.white_table[i]:
                out[..., i] = table[out[..., i], input_arr[..., index]]

            if self.is_last_layer:
                new_outputs[:, i] = self.last_codebook[i][1][out[:, i]]  # (codebook_length, codebook)

        if self.is_last_layer:
            return torch.Tensor(new_outputs)
        return torch.Tensor(out)

    def _normal_forward(self, input: torch.Tensor) -> torch.Tensor:

        weight = self.weight * self.weight_mask.float()
        out = input.matmul(weight.t())
        if self.bias is not None:
            out += self.bias * self.bias_mask.float()
        if self.activate_function is not None:
            out = self.activate_function(out)
        return out
