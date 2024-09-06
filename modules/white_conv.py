import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode

class WhiteConv2(ClusteringLayer, nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        padding_mode: str = 'replicate', # 注意大多数模型用的都是zero的padding模式，如果是zero模式，需要重训练原本的模型变成replicate模式，否则会带来2~3个点的精度损失
        activate_function=None,
        bias: bool = True,
        is_last_layer = False,
        groups: int = 1,
    ):
        super(WhiteConv2, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias, padding_mode=padding_mode, groups=groups)

        self.codebook = np.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=np.object)
        self.last_codebook = np.zeros(out_channels, dtype=np.object)
        self.previous_codebook = None

        self.white_table = np.zeros(out_channels, dtype=np.object)
        self.forward_mode = Forward_Mode.normal
        self.activate_function = activate_function
        self.is_last_layer = is_last_layer
        self.quantization_count = None

        self.kernel_size = (kernel_size, kernel_size)
        self.weight_mask = nn.Parameter(torch.ones(self.weight.shape).byte(), requires_grad=False)
        if bias:
            self.bias_mask = nn.Parameter(torch.ones(out_channels).byte(), requires_grad=False)
        self.groups = groups
        self.clusters_list = None

    def _whitebox_forward(self, input: torch.Tensor) -> torch.Tensor:
        input_arr = input.cpu().numpy().astype(np.uint8)
        expanded_padding = ((0,0),(0,0),(self.padding[1], self.padding[1]),
                            (self.padding[0], self.padding[0]))
        if self.padding_mode == "zeros":
            self.zero_pad_tabel = []
            for in_ch in range(self.in_channels):
                p = self.nearest_value(0, self.info.get_previous_codebook()[in_ch][1])
                self.zero_pad_tabel.append(p)
            x_padding = ((0, 0), (self.padding[1], self.padding[1]),(self.padding[0], self.padding[0]))
            input_arr_p = np.zeros(input_arr.shape)
            origin = np.pad(input_arr_p, expanded_padding, mode="edge")

            for in_ch in range(self.in_channels):
                origin[:, in_ch] = np.pad(input_arr[:,in_ch], x_padding, mode="constant", constant_values=self.zero_pad_tabel[in_ch])
            input_arr = origin
            input_arr = input_arr.astype(np.uint8)
        else:
            input_arr = np.pad(input_arr, expanded_padding, mode="edge")
        stride = self.stride[0]
        step = input_arr.shape[2] - self.kernel_size[0] + 1
        out = np.zeros((input.shape[0], self.out_channels, int((input_arr.shape[2] - self.kernel_size[0])/stride)+1, int((input_arr.shape[2] - self.kernel_size[0])/stride)+1), dtype=np.uint8)
        if self.is_last_layer:
            new_outputs = np.zeros_like(out, dtype=np.float32)
        for out_ch in range(self.out_channels):
            for (in_ch, i, j, table) in self.white_table[out_ch]:
                out[:, out_ch] = table[out[:, out_ch], input_arr[:, in_ch, i: i + step: stride, j: j + step: stride]]
                #这里out[:, out_ch]自动更新，table里的第一个参数其实就是上一次的out[]输出，即l的index

            if self.is_last_layer:
                new_outputs[:, out_ch] = self.last_codebook[out_ch][1][out[:, out_ch]]
        if self.is_last_layer:
            y = torch.Tensor(new_outputs)
        else:
            y = torch.Tensor(out)
        return y

    def _normal_forward(self, input) -> torch.Tensor:
        weight = self.weight * self.weight_mask.float()
        bias = None
        if self.bias is not None:
            bias = self.bias * self.bias_mask.float()
        expanded_padding = (self.padding[1], self.padding[1],
                            self.padding[0], self.padding[0])
        if self.padding_mode != 'zeros':
            input = F.pad(input, expanded_padding, self.padding_mode)
        else:
            input = F.pad(input, expanded_padding, 'constant') # zeros padding

        out = F.conv2d(input, weight, bias, self.stride,
                        (0,0), self.dilation, self.groups)

        if self.activate_function:
            out = self.activate_function(out)

        return out