import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode

class WhiteEmbedding(ClusteringLayer, nn.Embedding):
    def __init__(self, vocab_size:int, in_features: int, out_features: int, set_codebook=None) -> None:
        super(WhiteEmbedding, self).__init__(vocab_size,out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.vocab_size = vocab_size

        self.codebook = np.zeros((1), dtype=np.object)
        self.last_codebook = np.zeros(1, dtype=np.object)
        self.previous_codebook = None

        self.white_table = np.zeros(vocab_size, dtype=np.object)
        self.quantization_count = None
        self.forward_mode = Forward_Mode.normal
        self.set_codebook = set_codebook


    def _whitebox_forward(self, input) -> torch.Tensor:
        out = np.zeros((input.shape[0], self.in_features, self.out_features), dtype=np.uint8)
        for i in range(input.shape[0]):
            for j in range(self.in_features):
                out[i, j] = self.white_table[input[i, j]]

        return torch.Tensor(out)

    def _normal_forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input,  F.sigmoid(self.weight))
