import numpy as np
import torch
import torch.nn as nn

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode

class NoneZeroActivationFunction(ClusteringLayer, nn.Module):
    def __init__(self,  num_features: int,threshold_positive=0.1, threshold_negative=-0.1):
        super(NoneZeroActivationFunction, self).__init__()
        self.threshold_positive = threshold_positive
        self.threshold_negative = threshold_negative
        self.num_features = num_features
        self.last_codebook = np.zeros(num_features, dtype=np.object)
        self.codebook = np.zeros(num_features, dtype=np.object)
        self.previous_codebook = None
        self.white_table = np.zeros(num_features, dtype=np.object)
        self.quantization_count = None
        self.forward_mode = Forward_Mode.normal


    def forward(self, x):
        x = torch.where((x>=0) & (x < self.threshold_positive) , self.threshold_positive, x )
        x = torch.where((x<=0) & (x > self.threshold_negative),  self.threshold_negative, x)
        return x

    def _whitebox_forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input.cpu().numpy().astype(np.uint8)
        for i in range(self.num_features):
            out[:, i] = self.white_table[i][out[:, i]]

        return torch.Tensor(out)
