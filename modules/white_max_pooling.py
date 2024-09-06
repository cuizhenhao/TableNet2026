import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode


class WhiteMaxPool(ClusteringLayer, nn.Module):
    def __init__(self, in_channels: int, pool_size, stride, padding: int = 0, activate_function=None,
                 is_last_layer=False):
        super(WhiteMaxPool, self).__init__()
        self.in_channels = in_channels
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride

        self.last_codebook = np.zeros(in_channels, dtype=np.object)
        self.codebook = np.zeros((in_channels, self.pool_size[0] * self.pool_size[1] - 1), dtype=np.object)
        self.previous_codebook = None

        self.white_table = np.zeros((in_channels, self.pool_size[0] * self.pool_size[1] - 1), dtype=np.object)
        self.quantization_count = None
        self.activate_function = activate_function
        self.forward_mode = Forward_Mode.normal
        self.is_last_layer = is_last_layer
        self.padding = padding
        self.directions = self._get_directions()
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)

    def set_info(self, info):
        info.id += 1
        self.info = copy.copy(info)
        self.info.net_type = "max_pooling"

    def _get_directions(self):
        directions = []
        for x in range(self.pool_size[0]):
            for y in range(self.pool_size[1]):
                directions.append([x, y])
        return directions[1:]

    def _whitebox_forward(self, input: torch.Tensor) -> torch.Tensor:  # bug?
        input_arr = input.cpu().numpy().astype(np.uint8)
        expanded_padding = ((0, 0), (0, 0), (self.padding, self.padding),
                            (self.padding, self.padding))
        input_arr = np.pad(input_arr, expanded_padding, mode="edge")

        step1 = input_arr.shape[2] - self.pool_size[0] + 1
        step2 = input_arr.shape[3] - self.pool_size[1] + 1

        out = input_arr[:, :, :step1:self.stride[0], :step2:self.stride[1]].copy()
        if self.is_last_layer:
            new_outputs = np.zeros_like(out, dtype=np.float32)
        for inch in range(self.in_channels):
            for i, direction in enumerate(self.directions):
                a = out[:, inch]
                b = input_arr[:, inch, direction[0]:direction[0] + step1:self.stride[0],
                    direction[1]:direction[1] + step2:self.stride[1]]
                out[:, inch] = self.white_table[inch, i][a, b]

            if self.is_last_layer:
                new_outputs[:, inch] = self.last_codebook[inch][1][out[:, inch]]
        if self.is_last_layer:
            y = torch.Tensor(new_outputs)
        else:
            y = torch.Tensor(out)
        return y

    def _normal_forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.max_pool2d(input, self.pool_size, stride=self.stride, padding=self.padding)
        if self.activate_function is not None:
            out = self.activate_function(out)
        return out