import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

from .white_bn import WhiteBatchNorm2d
from .white_conv import WhiteConv2
from .white_pooling import WhitePool
from .white_net import WhiteNet
from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode


class WhiteDenseBlock(WhiteNet):
    def __init__(self, in_planes, growth_rate):
        super(WhiteDenseBlock, self).__init__()
        self.bn1 = WhiteBatchNorm2d(in_planes, activate_function=F.relu)
        self.conv1 = WhiteConv2(in_planes, 4*growth_rate, kernel_size=1, bias=False, padding_mode='zeros')
        self.bn2 = WhiteBatchNorm2d(4*growth_rate, activate_function=F.relu)
        self.conv2 = WhiteConv2(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False, padding_mode='zeros')
        self.cat = WhiteCat(in_planes, growth_rate)
        self.previous_codebook = None

    def init_info(self, layer_name, previous):
        self.bn1.init_info(layer_name + ".bn1", previous=previous)
        self.conv1.init_info(layer_name + ".conv1",previous=self.bn1)
        self.bn2.init_info(layer_name + ".bn2", previous=self.conv1)
        self.conv2.init_info(layer_name + ".conv2",previous=self.bn2)
        self.cat.init_info(layer_name + ".cat",previous=(previous, self.conv2))
        return [self.bn1,self.conv1,self.bn2,self.conv2,self.cat]

    def forward(self, x):
        out = self.conv1(self.bn1(x))
        out = self.conv2(self.bn2(out))
        out = self.cat((x, out))
        return out

    def get_size(self) -> int:
        return self.bn1.get_size() + self.conv1.get_size() + self.bn2.get_size() + self.conv2.get_size()

class WhiteTransition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(WhiteTransition, self).__init__()
        self.bn = WhiteBatchNorm2d(in_planes, activate_function=F.relu)
        self.conv = WhiteConv2(in_planes, out_planes, kernel_size=1, bias=False, padding_mode='zeros')
        self.pool = WhitePool(out_planes, 2)
        self.previous_codebook = None

    def forward(self, x):
        out = self.conv(self.bn(x))
        out = self.pool(out)
        return out

    def init_info(self, layer_name, previous):
        self.bn.init_info(layer_name+".bn",previous=previous)
        self.conv.init_info(layer_name+".conv",previous=self.bn)
        self.pool.init_info(layer_name+".pool",previous=self.conv)
        return [self.bn,self.conv,self.pool]

    def get_size(self) -> int:
        return self.bn.get_size() + self.conv.get_size() + self.pool.get_size()

class WhiteCat(ClusteringLayer, nn.Module):
    def __init__(self, left_in_channels: int, right_in_channels: int):
        super(WhiteCat, self).__init__()
        self.in_channels = left_in_channels + right_in_channels
        self.left_in_channels = left_in_channels
        self.right_in_channels = right_in_channels
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)
        self.previous_codebook = None
        
        self.last_codebook = np.zeros(self.in_channels, dtype=np.object)
        self.quantization_count = None
        self.forward_mode = Forward_Mode.normal
        self.is_last_layer = False

    def _whitebox_forward(self, input) -> torch.Tensor:
        left, right = input
        out = torch.cat([left, right], 1)
        
        if self.is_last_layer:
            out_arr = out.cpu().numpy().astype(np.uint8)
            new_outputs = np.zeros_like(out_arr, dtype=np.float32)
            for inch in range(self.in_channels):
                new_outputs[:, inch] = self.last_codebook[inch][1][out_arr[:, inch]]
            return torch.Tensor(new_outputs)
        return out

    def _normal_forward(self, input) -> torch.Tensor:
        left, right = input
        out = torch.cat([left, right], 1)
        return out



