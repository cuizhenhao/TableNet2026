import numpy as np
import torch
import torch.nn as nn

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode


class WhiteConvEmbedding(ClusteringLayer, nn.Module):
    def __init__(self, vocab_size:int, word_num: int, word_emb_dim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride: int = 1,
                 padding: int = 0,
                 padding_mode: str = 'replicate',
                 activate_function=None,
                 bias: bool = True,
                 groups: int = 1,
                 set_codebook=None) -> None:
        super(WhiteConvEmbedding, self).__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels,  self.kernel_size, stride, padding, bias=bias, padding_mode=padding_mode, groups= groups)
        self.embedding = nn.Embedding(vocab_size, word_emb_dim)

        self.word_num = word_num
        self.word_emb_dim = word_emb_dim
        self.vocab_size = vocab_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activate_function = activate_function

        self.codebook = np.zeros(out_channels, dtype=np.object)
        self.last_codebook = np.zeros(1, dtype=np.object)
        self.previous_codebook = None

        self.white_table = np.zeros(vocab_size, dtype=np.object)
        self.quantization_count = None
        self.forward_mode = Forward_Mode.normal
        self.set_codebook = set_codebook

    def _whitebox_forward(self, input) -> torch.Tensor:
        out = np.zeros((input.shape[0], self.out_channels ,self.word_num, 1), dtype=np.uint8)
        for out_ch in range(self.out_channels):
            for j in range(self.word_num):
                out[:, out_ch, j, 0] = self.white_table[out_ch][input[:, j]]

        return torch.Tensor(out)

    def _normal_forward(self, input: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input)
        emb = emb.unsqueeze(1)
        out = self.conv(emb)
        if self.activate_function is not None:
            out = self.activate_function(out)
        return out
