import numpy as np
import torch

from modules.white_conv import WhiteConv2


class WhiteConv2Groups(WhiteConv2):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, padding_mode: str = 'zeros', bias: bool = True, activate_function=None, is_last_layer=False,
                 groups: int = 1):

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, padding_mode, bias, activate_function,
                         is_last_layer, groups)
        assert self.groups == self.in_channels
        assert self.groups == self.out_channels


    def _whitebox_forward(self, input: torch.Tensor) -> torch.Tensor:
        input_arr = input.cpu().numpy().astype(np.uint8)
        expanded_padding = ((0,0),(0,0),(self.padding[1], self.padding[1]),
                            (self.padding[0], self.padding[0]))
        if self.padding_mode == "zeros":
            self.zero_pad_tabel = []
            for in_ch in range(self.in_channels):
                p = self.nearest_value(0, self.previous_codebook[in_ch][1])
                self.zero_pad_tabel.append(p)
            x_padding = ((0, 0), (self.padding[1], self.padding[1]),(self.padding[0], self.padding[0]))
            input_arr_p = np.zeros(input_arr.shape)
            origin = np.pad(input_arr_p, expanded_padding, mode="edge")

            for in_ch in range(self.in_channels):
                origin[:, in_ch] = np.pad(input_arr[:,in_ch], x_padding, mode="constant", constant_values=self.zero_pad_tabel[in_ch])
            # input_arr = np.pad(input_arr, expanded_padding, mode="constant", constant_values=self.zero_pad_tabel)
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
        return super()._normal_forward(input)