import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode

class WhitePool(ClusteringLayer, nn.Module):
    def __init__(self, in_channels: int, pool_size: int, stride: int = None, padding: int = 0, activate_function=None,
                 is_last_layer=False):
        super(WhitePool, self).__init__()
        self.last_codebook = np.zeros(in_channels, dtype=np.object)
        self.codebook = np.zeros((in_channels, pool_size * pool_size - 1), dtype=np.object)
        self.previous_codebook = None

        self.white_table = np.zeros((in_channels, pool_size * pool_size - 1), dtype=np.object)
        self.quantization_count = None
        self.activate_function = activate_function
        self.forward_mode = Forward_Mode.normal
        self.is_last_layer = is_last_layer

        self.in_channels = in_channels
        self.pool_size = pool_size
        self.stride = stride or pool_size
        self.padding = padding
        self.directions = self._get_directions(pool_size)
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)

    def _get_directions(self, pool_size: int):
        directions = []
        for x in range(pool_size):
            for y in range(pool_size):
                directions.append([x, y])
        return directions[1:]

    def _whitebox_forward(self, input: torch.Tensor) -> torch.Tensor:
        input_arr = input.cpu().numpy().astype(np.uint8)
        expanded_padding = ((0, 0), (0, 0), (self.padding, self.padding),
                            (self.padding, self.padding))
        input_arr = np.pad(input_arr, expanded_padding, mode="edge")

        step = input_arr.shape[2] - self.pool_size + 1
        out = input_arr[:, :, :step:self.stride, :step:self.stride].copy()
        if self.is_last_layer:
            new_outputs = np.zeros_like(out, dtype=np.float32)
        for inch in range(self.in_channels):
            for i, direction in enumerate(self.directions):
                a = out[:, inch]
                b = input_arr[:, inch, direction[0]:direction[0] + step:self.stride,
                    direction[1]:direction[1] + step:self.stride]
                out[:, inch] = self.white_table[inch, i][a, b]

            if self.is_last_layer:
                new_outputs[:, inch] = self.last_codebook[inch][1][out[:, inch]]
        if self.is_last_layer:
            y = torch.Tensor(new_outputs)
        else:
            y = torch.Tensor(out)
        return y

    def _normal_forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.avg_pool2d(input, self.pool_size, stride=self.stride, padding=self.padding)

        if self.activate_function is not None:
            out = self.activate_function(out)
        return out
