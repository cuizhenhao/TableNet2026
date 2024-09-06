import numpy as np
import torch
import torch.nn as nn

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode
from utils.bin import high4bit, low4bit, mean_cluster, l3_h4bit, l4_h3bit, l4_h4bit


class WhiteMiniLinear(ClusteringLayer, nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, activate_function = None, is_last_layer = False) -> None:
        super(WhiteMiniLinear, self).__init__(in_features, out_features, bias)

        self.codebook = np.zeros((out_features, in_features), dtype=np.object)
        self.last_codebook = np.zeros(out_features, dtype=np.object)
        self.previous_codebook = None

        self.white_table = np.zeros(out_features, dtype=np.object)
        self.quantization_count = None
        self.activate_function = activate_function
        self.forward_mode = Forward_Mode.normal  #0 for normal forward, 1 for clustering forward, 2 for whitebox forward
        self.is_last_layer = is_last_layer
        self.weight_mask = nn.Parameter(torch.ones(self.weight.shape).byte(), requires_grad=False)
        if bias:
            self.bias_mask = nn.Parameter(torch.ones(self.bias.shape).byte(), requires_grad=False)
        self.sp_rate = 1.1
        self.min_max = np.zeros((out_features, in_features), dtype=np.object)

    def _whitebox_forward(self, input: torch.Tensor) -> torch.Tensor:
        input_arr = input.cpu().numpy().astype(np.uint8)
        out = np.zeros((input_arr.shape[0], self.out_features), dtype=np.uint8)
        if self.is_last_layer:
            new_outputs = np.zeros_like(out, dtype=np.float32)
        for i in range(self.out_features):
            # i = 270
            total_add_num = self.weight_mask[i].sum()
            add_num = 0
            for index, tables in self.white_table[i]:
                add_num+=1
                if add_num < total_add_num * self.sp_rate:
                    wx1,wx2 = tables[0][out[:,i]],tables[1][input_arr[:, index]]
                    l1, h1 = l4_h4bit(wx1)
                    l2, h2 = l4_h4bit(wx2)
                    b1 = tables[2][l1, l2]
                    b2 = tables[3][h1, h2]
                    out[:, i] = tables[4][b1,b2]
                else:
                    pass
            if self.is_last_layer:
                new_outputs[:, i] = self.last_codebook[i][1][out[:, i]]#(codebook_length, codebook)

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
